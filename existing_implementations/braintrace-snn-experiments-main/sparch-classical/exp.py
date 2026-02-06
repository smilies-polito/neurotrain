#
# SPDX-FileCopyrightText: Copyright © 2022 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Alexandre Bittar <abittar@idiap.ch>
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This file is part of the sparch package
#
"""
This is to define the experiment class used to perform training and testing
of ANNs and SNNs on all speech command recognition datasets.
"""

import datetime
import errno
import os
import time
from datetime import timedelta

import brainscale
import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from args import print_training_options, print_model_options
from general_utils import setup_logging, load_model_states, save_model_states
from snns import SNN, SNNExtractSpikes
from spiking_datasets import load_dataset


class Experiment(brainstate.util.PrettyObject):
    """
    Class for training and testing models (ANNs and SNNs) on all four
    datasets for speech command recognition (shd, ssc, hd and sc).
    """

    def __init__(self, args):
        self.args = args

        # New model config
        self.net_type = args.model_type
        self.nb_layers = args.nb_layers
        self.nb_hiddens = args.nb_hiddens
        self.pdrop = args.pdrop
        self.normalization = args.normalization
        self.use_bias = args.use_bias

        # Training config
        self.load_exp_folder = args.load_exp_folder
        self.new_exp_folder = args.new_exp_folder
        self.dataset_name = args.dataset_name
        self.data_folder = args.data_folder
        self.save_best = args.save_best
        self.batch_size = args.batch_size
        self.nb_epochs = args.nb_epochs
        self.start_epoch = args.start_epoch
        self.lr = args.lr
        self.use_augm = args.use_augm

        # Initialize logging and output folders
        self.init_exp_folders()
        self.logger = setup_logging(os.path.join(self.log_dir, 'exp.log'))
        print_model_options(self.logger, args)
        print_training_options(self.logger, args)

        # Initialize dataloaders and model
        self.init_dataset()
        self.init_model()

        # Define optimizer
        self.trainable_weights = self.net.states(brainstate.ParamState)
        lr = braintools.optim.StepLR(self.lr, step_size=args.lr_step_size, gamma=args.lr_step_gamma)
        self.optimizer = braintools.optim.Adam(lr)
        self.optimizer.register_trainable_weights(self.trainable_weights)

    def f_train(self):
        """
        This function performs model training with the configuration
        specified by the class initialization.
        """
        # Initialize best accuracy
        best_epoch, best_acc = 0, 0

        # Loop over epochs (training + validation)
        self.logger.warning("\n------ Begin training ------\n")
        for e in range(best_epoch + 1, best_epoch + self.nb_epochs + 1):
            self.train_one_epoch(e)
            best_epoch, best_acc = self.valid_one_epoch(e, best_epoch, best_acc)
            self.optimizer.lr.step_epoch()
        self.logger.warning(f"\nBest valid acc at epoch {best_epoch}: {best_acc}\n")
        self.logger.warning("\n------ Training finished ------\n")

        # Loading best model
        if self.save_best:
            load_model_states(f"{self.checkpoint_dir}/best_model.pth", self.net)
            self.logger.warning(f"Loading best model, epoch={best_epoch}, valid acc={best_acc}")
        else:
            self.logger.warning(
                "Cannot load best model because save_best option is "
                "disabled. Model from last epoch is used for testing."
            )

        # Test trained model
        self.test_one_epoch(self.valid_loader)
        self.logger.warning("\nThis dataset uses the same split for validation and testing.\n")

    def f_test(self, n_fig=5):

        data = iter(self.valid_loader)

        for _ in range(5):
            x, y = next(data)

            # validation
            x = jnp.asarray(x)
            print(x.shape)
            outs = self._validate(x)
            outs = jax.tree.map(np.asarray, outs)

            # visualization
            fig, gs = braintools.visualize.get_figure(len(outs), n_fig, 3, 3)
            for i, out in enumerate(outs):
                for i_img in range(n_fig):
                    fig.add_subplot(gs[i, i_img])
                    spikes = out[:, i_img]
                    spikes = np.reshape(spikes, (spikes.shape[0], -1))
                    # Create a raster plot of spikes
                    neuron_indices = np.where(spikes > 0)
                    plt.scatter(neuron_indices[0], neuron_indices[1], s=1, c='black', marker='|')
                    plt.ylabel('Neuron Index')
                    plt.xlabel('Time Step')
                    plt.title(f'Sample {i_img}, Layer {i}')
            plt.show()
            plt.close()

    def _validate(self, inputs):
        inputs = self._process_input(inputs)

        # add environment context
        model = brainstate.nn.EnvironContext(
            SNNExtractSpikes(self.net),
            fit=False
        )

        # assume the inputs have shape (time, batch, features, ...)
        n_time, n_batch = inputs.shape[:2]
        brainstate.nn.vmap_init_all_states(
            model,
            state_tag='hidden',
            axis_size=n_batch,
        )
        model = brainstate.nn.Vmap(
            model,
            vmap_states='hidden',
            axis_name='batch' if self.normalization == 'batchnorm' else None
        )

        # forward propagation
        outs = brainstate.transform.for_loop(model, inputs)

        return outs

    def _loss(self, predictions, targets):
        return braintools.metric.softmax_cross_entropy_with_integer_labels(predictions, targets).mean()

    def _acc(self, predictions, target):
        return jnp.mean(jnp.equal(target, jnp.argmax(predictions, axis=1)))

    def _process_input(self, inputs):
        inputs = u.math.flatten(jnp.asarray(inputs), start_axis=2)
        inputs = inputs.transpose((1, 0, 2))  # [n_time, n_batch, n_feature]
        return inputs

    @brainstate.transform.jit(static_argnums=0)
    def predict(self, inputs: jax.Array, targets: jax.Array):
        inputs = self._process_input(inputs)

        # add environment context
        model = brainstate.nn.EnvironContext(self.net, fit=False)

        # assume the inputs have shape (time, batch, features, ...)
        n_time, n_batch = inputs.shape[:2]
        brainstate.nn.vmap_init_all_states(
            model,
            state_tag='hidden',
            axis_size=n_batch,
        )
        model = brainstate.nn.Vmap(
            model,
            vmap_states='hidden',
            axis_name='batch' if self.normalization == 'batchnorm' else None
        )

        # forward propagation
        outs = brainstate.transform.for_loop(model, inputs)
        outs = outs.sum(axis=0)
        # outs = outs[-1]

        # loss
        loss = self._loss(outs, targets)

        # accuracy
        acc = self._acc(outs, targets)
        return acc, loss

    @brainstate.transform.jit(static_argnums=0)
    def bptt_train(self, inputs, targets):
        inputs = self._process_input(inputs)

        brainstate.nn.vmap_init_all_states(self.net, state_tag='hidden', axis_size=inputs.shape[1])
        model = brainstate.nn.EnvironContext(self.net, fit=True)
        model = brainstate.nn.Vmap(
            model,
            vmap_states='hidden',
            axis_name='batch' if self.normalization == 'batchnorm' else None
        )

        def _bptt_grad_step():
            outs = brainstate.transform.for_loop(model, inputs)
            outs = outs.sum(axis=0)
            # outs = outs[-1]
            loss = self._loss(outs, targets)
            return loss, outs

        # gradients
        grads, loss, outs = brainstate.transform.grad(
            _bptt_grad_step,
            self.trainable_weights,
            has_aux=True,
            return_value=True
        )()

        # optimization
        self.optimizer.update(grads)

        # accuracy
        acc = self._acc(outs, targets)
        return acc, loss

    @brainstate.transform.jit(static_argnums=0)
    def online_train(self, inputs, targets):
        inputs = self._process_input(inputs)

        # assume the inputs have shape (time, batch, features, ...)
        n_time, n_batch = inputs.shape[:2]

        # initialize the online learning model
        model = brainstate.nn.EnvironContext(self.net, fit=True)
        if self.args.method == 'esd-rtrl':
            model = brainscale.IODimVjpAlgorithm(model, self.args.etrace_decay, vjp_method=self.args.vjp_method)
        elif self.args.method == 'd-rtrl':
            model = brainscale.ParamDimVjpAlgorithm(model, vjp_method=self.args.vjp_method)
        else:
            raise ValueError(f'Unknown online learning methods: {self.args.method}.')

        @brainstate.transform.vmap_new_states(state_tag='new', axis_size=n_batch)
        def init():
            inp = jax.ShapeDtypeStruct(inputs.shape[2:], inputs.dtype)
            brainstate.nn.init_all_states(self.net)
            model.compile_graph(inp)
            model.show_graph()

        init()
        model = brainstate.nn.Vmap(
            model,
            vmap_states='new',
            axis_name='batch' if self.normalization == 'batchnorm' else None
        )

        def _etrace_grad(inp):
            out = model(inp)
            loss = self._loss(out, targets)
            return loss, out

        def _etrace_step(prev_grads, x):
            f_grad = brainstate.transform.grad(
                _etrace_grad,
                self.trainable_weights,
                has_aux=True,
                return_value=True
            )
            cur_grads, local_loss, out = f_grad(x)
            next_grads = jax.tree.map(lambda a, b: a + b, prev_grads, cur_grads)
            return next_grads, (out, local_loss)

        def _etrace_train(inputs_):
            grads = jax.tree.map(lambda a: jnp.zeros_like(a), self.trainable_weights.to_dict_values())
            grads, (outs, losses) = brainstate.transform.scan(_etrace_step, grads, inputs_)
            self.optimizer.update(grads)
            return losses.mean(), outs.sum(axis=0)

        loss, out_sum = _etrace_train(inputs)

        # accuracy
        acc = self._acc(out_sum, targets)

        # returns
        return acc, loss

    def init_exp_folders(self):
        """
        This function defines the output folders for the experiment.
        """

        # Use given path for new model folder
        if self.new_exp_folder is not None:
            exp_folder = self.new_exp_folder

        else:
            # Generate a path for new model from chosen config
            if self.args.method == 'esd-rtrl':
                outname = f'{self.args.method}_{self.args.etrace_decay}_{self.dataset_name}/'
            else:
                outname = f'{self.args.method}_{self.dataset_name}/'
            outname = outname + self.net_type + "_"
            outname += str(self.nb_layers) + "lay" + str(self.nb_hiddens)
            outname += "_drop" + str(self.pdrop) + "_" + str(self.normalization)
            outname += "_bias" if self.use_bias else "_nobias"
            outname += "_lr" + str(self.lr)
            exp_folder = f"{outname.replace('.', '_')}/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}/"

        # For a new model check that out path does not exist
        if os.path.exists(exp_folder):
            raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), exp_folder)

        # Create folders to store experiment
        self.log_dir = exp_folder
        self.checkpoint_dir = exp_folder
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.exp_folder = exp_folder

    def init_dataset(self):
        """
        This function prepares dataloaders for the desired dataset.
        """
        results = load_dataset(self.args)

        self.nb_inputs = results['in_shape']
        self.nb_outputs = results['out_shape']

        self.train_loader = results['train_loader']
        self.valid_loader = results['test_loader']
        if self.use_augm:
            self.logger.warning("\nWarning: Data augmentation not implemented for SHD and SSC.\n")

    def init_model(self):
        """
        This function either loads pretrained model or builds a
        new model (ANN or SNN) depending on chosen config.
        """
        layer_sizes = [self.nb_hiddens] * (self.nb_layers - 1) + [self.nb_outputs]

        if self.net_type in ["LIF", "adLIF", "RLIF", "RadLIF"]:
            self.net = SNN(
                input_shape=self.nb_inputs,
                layer_sizes=layer_sizes,
                neuron_type=self.net_type,
                dropout=self.pdrop,
                normalization=self.normalization,
                use_bias=self.use_bias,
                use_readout_layer=True,
                inp_scale=self.args.inp_scale,
                rec_scale=self.args.rec_scale,
            )
            self.logger.warning(f"\nCreated new spiking model:\n {self.net}\n")

        else:
            raise ValueError(f"Invalid model type {self.net_type}")

        table, _ = brainstate.nn.count_parameters(self.net, return_table=True)
        self.logger.warning('\n' + str(table))

    def train_one_epoch(self, e):
        """
        This function trains the model with a single pass over the
        training split of the dataset.
        """
        start = time.time()
        losses, accs = [], []

        # Loop over batches from train set
        for step, (x, y) in enumerate(self.train_loader):
            # Forward pass through network
            x = jnp.asarray(x)  # images:[bs, 1, 28, 28]
            y = jnp.asarray(y)
            if self.args.method == 'bptt':
                acc, loss = self.bptt_train(x, y)
            else:
                acc, loss = self.online_train(x, y)
            losses.append(loss)
            accs.append(acc)

        # Learning rate of whole epoch
        current_lr = self.optimizer.current_lr
        self.logger.warning(f"Epoch {e}: lr={current_lr}")

        # Train loss of whole epoch
        train_loss = np.mean(losses)
        self.logger.warning(f"Epoch {e}: train loss={train_loss}")

        # Train accuracy of whole epoch
        train_acc = np.mean(accs)
        self.logger.warning(f"Epoch {e}: train acc={train_acc}")

        end = time.time()
        elapsed = str(timedelta(seconds=end - start))
        self.logger.warning(f"Epoch {e}: train elapsed time={elapsed}")

    def valid_one_epoch(self, e, best_epoch, best_acc):
        """
        This function tests the model with a single pass over the
        validation split of the dataset.
        """
        losses, accs = [], []

        # Loop over batches from validation set
        for step, (x, y) in enumerate(self.valid_loader):
            # Forward pass through network
            x = jnp.asarray(x)  # images:[bs, 1, 28, 28]
            y = jnp.asarray(y)
            acc, loss = self.predict(x, y)
            losses.append(loss)
            accs.append(acc)

        # Validation loss of whole epoch
        valid_loss = np.mean(losses)
        self.logger.warning(f"Epoch {e}: valid loss={valid_loss}")

        # Validation accuracy of whole epoch
        valid_acc = np.mean(accs)
        self.logger.warning(f"Epoch {e}: valid acc={valid_acc}")

        # # Update learning rate
        # self.scheduler.step(valid_acc)

        # Update the best epoch and accuracy
        if valid_acc > best_acc:
            best_acc = valid_acc
            best_epoch = e

            # Save best model
            if self.save_best:
                save_model_states(
                    f"{self.checkpoint_dir}/best_model.pth", self.net, valid_acc=best_acc, epoch=best_epoch)
                self.logger.warning(f"\nBest model saved with valid acc={valid_acc}")

        self.logger.warning("\n-----------------------------\n")

        return best_epoch, best_acc

    def test_one_epoch(self, test_loader):
        """
        This function tests the model with a single pass over the
        testing split of the dataset.
        """
        losses, accs = [], []
        epoch_spike_rate = 0

        self.logger.warning("\n------ Begin Testing ------\n")

        # Loop over batches from test set
        for step, (x, y) in enumerate(test_loader):
            # Forward pass through network
            x = jnp.asarray(x)  # images:[bs, 1, 28, 28]
            y = jnp.asarray(y)
            acc, loss = self.predict(x, y)
            losses.append(loss)
            accs.append(acc)

        # Test loss
        test_loss = np.mean(losses)
        self.logger.warning(f"Test loss={test_loss}")

        # Test accuracy
        test_acc = np.mean(accs)
        self.logger.warning(f"Test acc={test_acc}")

        self.logger.warning("\n-----------------------------\n")
