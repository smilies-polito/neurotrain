# Copyright (c) 2023  Khaleelulla Khan Nazeer, Anand Subramoney, Mark Schöne, David Kappel
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import glob
import os
import sys
import time
from datetime import datetime

from args import parse_args, store_args

args = parse_args()
os.environ['JAX_TRACEBACK_FILTERING'] = 'off'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.99'

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import jax
import jax.numpy as jnp
import tonic

import braintools
import braintrace
import brainstate
from data import get_dvs128_test_dataset, get_dvs128_train_val
from model import Network, FiringRateState
from general_utils import copy_files, save_model_states, AverageMeter, setup_logging, ProgressMeter


def resume_model(
    args,
    model: brainstate.nn.Module,
    optimizer: braintools.optim.Optimizer,
):
    assert args.resume_path is not None, "Resume path is not provided"
    assert os.path.exists(args.resume_path), "Resume path does not exist"
    assert isinstance(model, brainstate.nn.Module), "Model is not a brainstate.nn.Module"
    assert isinstance(optimizer, braintools.optim.Optimizer), "Optimizer is not a braintools.optim.Optimizer"

    targets = {
        'epoch': 0.,
        'accuracy': 0.,
        'state_dict': model.states(brainstate.ParamState),
        'optimizer_state_dict': brainstate.graph.states(optimizer),
    }

    model_path = sorted(glob.glob(os.path.join(args.resume_path, '*-Epoch-*.msgpack')))[-1]
    targets = braintools.file.msgpack_load(model_path, target=targets)
    print(f"Model Restored from Epoch {targets[0]} at path {model_path}")
    start_epoch = targets['epoch'] + 1
    accuracy = targets['accuracy']

    return start_epoch, accuracy


class Trainer(brainstate.util.PrettyObject):
    def __init__(self, args):
        self.args = args
        if self.args.seed is not None:
            brainstate.random.seed(self.args.seed)

        # tensorboard
        filename = f'./dvs_gesture-{self.args.rnn_type}-{self.args.num_layers}-{self.args.method}'
        if self.args.method == 'esd-rtrl':
            filename += '-{self.args.etrace_decay}'
        filename += f'-{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'
        tensorboard_dir = os.path.join(self.args.logdir, filename)
        os.makedirs(tensorboard_dir, exist_ok=True)
        print(f'Tensorboard logged to {os.path.abspath(tensorboard_dir)}')
        print('Backing up current file')
        copy_files(os.path.join(os.path.dirname(os.path.abspath(__file__))), tensorboard_dir)
        store_args(os.path.join(tensorboard_dir, 'args.json'), self.args)
        self.tensorboard_dir = tensorboard_dir
        self.logger = setup_logging(os.path.join(tensorboard_dir, 'log.txt'))

        # defining model
        model = Network(
            input_size=tonic.datasets.DVSGesture.sensor_size,
            frame_size=args.frame_size,
            n_rnn_layer=args.num_layers,
            n_rnn_hidden=args.units,
            n_class=len(tonic.datasets.DVSGesture.classes),
            zoneout=args.zoneout,
            layer_dropout=args.dropout,
            pseudo_derivative_width=args.pseudo_derivative_width,
            threshold_mean=args.threshold_mean,
            rnn_type=args.rnn_type,
            event_agg_method=args.event_agg_method,
            use_cnn=args.use_cnn,
        )
        self.model = model
        table, _ = brainstate.nn.count_parameters(self.model, return_table=True)
        self.trainable_weights = self.model.states(brainstate.ParamState)

        self.logger.warning(str(self.model))
        self.logger.warning(str(table))

        # optimizer and lr scheduler
        scheduler = braintools.optim.StepLR(
            args.learning_rate,
            step_size=args.lr_decay_epochs,
            gamma=args.lr_gamma
        )
        if args.use_rmsprop:
            optimizer = braintools.optim.RMSProp(lr=scheduler, weight_decay=0.9)
        else:
            optimizer = braintools.optim.Adam(lr=scheduler)
        optimizer.register_trainable_weights(self.trainable_weights)
        self.optimizer = optimizer

        # resume model
        if args.resume_path:
            self.start_epoch, self.best_acc = resume_model(args, model, optimizer)

        else:
            self.start_epoch = 1
            self.best_acc = float('-inf')

    def _loss(self, predictions, targets):
        return braintools.metric.softmax_cross_entropy_with_integer_labels(predictions, targets).mean()

    def _acc(self, predictions, target):
        return jnp.mean(jnp.equal(target, jnp.argmax(jnp.mean(predictions, axis=0), axis=1)))

    def _spike_sparsity(self, n_time: int, n_batch: int):
        fr_states = brainstate.graph.states(self.model, FiringRateState)
        if len(fr_states) == 1:
            mean_activity = jax.tree.map(
                lambda a: jnp.mean(a) / n_time / n_batch,
                fr_states.to_dict_values()
            )
            mean_activity = jax.tree.leaves(mean_activity)
            assert len(mean_activity) == 1, str(mean_activity)
            return mean_activity[0]
        else:
            return 1.

    @brainstate.transform.jit(static_argnums=0)
    def predict(self, inputs: jax.Array, targets: jax.Array):
        inputs = jnp.asarray(inputs)
        inputs = jnp.transpose(inputs, (0, 1, 3, 4, 2))

        # add environment context
        model = brainstate.nn.EnvironContext(self.model, fit=False)

        # assume the inputs have shape (time, batch, features, ...)
        n_time, n_batch = inputs.shape[:2]
        brainstate.nn.vmap_init_all_states(model, state_tag='hidden', axis_size=n_batch)

        # forward propagation
        def _step(inp):
            out = brainstate.transform.vmap(model, in_states=model.states('hidden'))(inp)
            loss = self._loss(out, targets)
            return loss, out

        losses, outs = brainstate.transform.for_loop(_step, inputs)

        # firing rate
        spike_sparsity = self._spike_sparsity(n_time, n_batch)

        # accuracy
        acc = self._acc(outs, targets)
        return losses.mean(), acc, spike_sparsity

    @brainstate.transform.jit(static_argnums=(0,))
    def etrace_train(self, inputs, targets):
        inputs = jnp.asarray(inputs)
        inputs = jnp.transpose(inputs, (0, 1, 3, 4, 2))

        # assume the inputs have shape (time, batch, features, ...)
        n_time, n_batch = inputs.shape[:2]

        # initialize the online learning model
        model = brainstate.nn.EnvironContext(self.model, fit=True)
        if self.args.method == 'es-d-rtrl':
            model = braintrace.IODimVjpAlgorithm(model, self.args.etrace_decay, vjp_method=self.args.vjp_method)
        elif self.args.method == 'd-rtrl':
            model = braintrace.ParamDimVjpAlgorithm(model, vjp_method=self.args.vjp_method)
        else:
            raise ValueError(f'Unknown online learning methods: {self.args.method}.')

        @brainstate.transform.vmap_new_states(state_tag='new', axis_size=n_batch)
        def init():
            """
            Initialize the model states and compile the computation graph.

            This function performs the following tasks:
            1. Creates a shape and dtype structure for the input.
            2. Initializes all states of the model.
            3. Compiles the computation graph.
            4. Displays the compiled graph.

            The function is decorated with `vmap_new_state`, which vectorizes the function
            across a new state axis with the tag 'new' and size `n_batch`.
            """
            inp = jax.ShapeDtypeStruct(inputs.shape[2:], inputs.dtype)
            brainstate.nn.init_all_states(self.model)
            model.compile_graph(inp)
            model.show_graph()

        init()
        model = brainstate.nn.Vmap(model, vmap_states='new')

        def _etrace_grad(inp):
            out = model(inp)
            loss = self._loss(out, targets)
            return loss, out

        def _etrace_step(prev_grads, x):
            # no need to return weights and states, since they are generated then no longer needed
            f_grad = brainstate.transform.grad(_etrace_grad, self.trainable_weights, has_aux=True, return_value=True)
            cur_grads, local_loss, out = f_grad(x)
            next_grads = jax.tree.map(lambda a, b: a + b, prev_grads, cur_grads)
            return next_grads, (out, local_loss)

        def _etrace_train(inputs_):
            # forward propagation
            grads = jax.tree.map(lambda a: jnp.zeros_like(a), self.trainable_weights.to_dict_values())
            grads, (outs, losses) = brainstate.transform.scan(_etrace_step, grads, inputs_)
            # gradient updates
            if self.args.use_grad_clipping:
                grads = brainstate.functional.clip_grad_norm(grads, self.args.grad_clip_norm)
            self.optimizer.update(grads)
            return losses.mean(), outs

        # running indices
        if self.args.warmup_ratio > 0:
            n_sim = brainstate.util.split_total(inputs.shape[0], self.args.warmup_ratio)
            brainstate.transform.for_loop(model, inputs[:n_sim])
            loss, outs = _etrace_train(inputs[n_sim:])
        else:
            loss, outs = _etrace_train(inputs)

        # firing rate
        spike_sparsity = self._spike_sparsity(n_time, n_batch)

        # accuracy
        acc = self._acc(outs, targets)
        # returns
        return loss, acc, spike_sparsity

    @brainstate.transform.jit(static_argnums=(0,))
    def bptt_train(self, inputs, targets):
        inputs = jnp.asarray(inputs)
        inputs = jnp.transpose(inputs, (0, 1, 3, 4, 2))

        # initialize the states
        brainstate.nn.vmap_init_all_states(self.model, state_tag='hidden', axis_size=inputs.shape[1])

        model = brainstate.nn.EnvironContext(self.model, fit=True)
        model = brainstate.nn.Vmap(model, vmap_states='hidden')

        def _run_step_train(inp):
            out = model(inp)
            return out, self._loss(out, targets)

        def _bptt_grad_step():
            if self.args.warmup_ratio > 0:
                n_sim = brainstate.util.split_total(inputs.shape[0], self.args.warmup_ratio)
                _ = brainstate.transform.for_loop(model, inputs[:n_sim])
                outs, losses = brainstate.transform.for_loop(_run_step_train, inputs[n_sim:])
            else:
                outs, losses = brainstate.transform.for_loop(_run_step_train, inputs)
            return losses.mean(), outs

        # gradients
        grads, loss, outs = brainstate.transform.grad(
            _bptt_grad_step, self.trainable_weights, has_aux=True, return_value=True)()

        # optimization
        if self.args.use_grad_clipping:
            grads = brainstate.functional.clip_grad_norm(grads, self.args.grad_clip_norm)
        self.optimizer.update(grads)

        # firing rate
        spike_sparsity = self._spike_sparsity(inputs.shape[0], inputs.shape[1])

        # accuracy
        acc = self._acc(outs, targets)

        return loss, acc, spike_sparsity

    def val_epoch(self, data_loader):
        losses = AverageMeter('Loss', ':.4e')
        accuracies = AverageMeter('Acc', ':6.8f')
        mean_activities = AverageMeter('Sparsity', ':.8f')
        progress = ProgressMeter(
            len(data_loader),
            [losses, accuracies, mean_activities],
            prefix="Test: "
        )

        for i, (data, targets) in enumerate(data_loader):
            data, targets = jnp.asarray(data), jnp.asarray(targets)
            n_time, batch_size = data.shape[:2]
            loss, acc, mean_activity = self.predict(data, targets)

            losses.update(float(loss), batch_size)
            accuracies.update(float(acc), batch_size)
            mean_activities.update(float(mean_activity), batch_size)

            self.logger.warning(progress.display(i * self.args.batch_size))

        # show info
        self.logger.warning(
            f'Validation set ({len(data_loader.dataset):d} samples): '
            f'Average loss: {losses.avg:.8f}\t'
            f'Acc: {accuracies.avg * 100:.8f}%'
        )
        return losses.avg, accuracies.avg, mean_activities.avg

    def train_epoch(self, data_loader, epoch: int):
        log_interval = self.args.log_interval
        num_batches = len(data_loader)
        start_time = time.time()

        train_loss = 0.0
        losses = AverageMeter('Loss', ':.4e')
        accuracies = AverageMeter('Acc', ':6.8f')
        mean_activities = AverageMeter('Sparsity', ':.8f')
        progress = ProgressMeter(
            len(data_loader),
            [accuracies, losses, mean_activities],
            prefix="Epoch: [{}]".format(epoch)
        )

        for i, (data, targets) in enumerate(data_loader):
            batch_size = data.shape[1]
            data, targets = jnp.asarray(data), jnp.asarray(targets)
            if self.args.method == 'bptt':
                loss, acc, spike_sparsity = map(float, self.bptt_train(data, targets))
            else:
                loss, acc, spike_sparsity = map(float, self.etrace_train(data, targets))

            train_loss += loss
            losses.update(loss, batch_size)
            accuracies.update(acc, batch_size)
            mean_activities.update(spike_sparsity, batch_size)

            self.logger.warning(progress.display(i * self.args.batch_size))

            if (i + 1) % log_interval == 0:
                lr = self.optimizer.lr()
                ms_per_batch = (time.time() - start_time) * 1000 / log_interval
                avg_loss = train_loss / log_interval
                print(
                    f'| epoch {epoch:3d} '
                    f'| {i:5d}/{num_batches:5d} batches '
                    f'| lr {lr:02.8f} '
                    f'| ms/batch {ms_per_batch:5.8f} '
                    f'| loss {avg_loss:5.8f} '
                    f'| Acc {accuracies.avg * 100:8.8f} '
                    f'| mean activity {mean_activities.avg:.8f}'
                )
                start_time = time.time()
                train_loss = 0.0

        self.logger.warning(
            f'Train set ({len(data_loader.dataset):d} samples): '
            f'Average loss: {losses.avg:.8f}\t'
            f'Acc: {accuracies.avg * 100:.8f}%'
        )

        return losses.avg, accuracies.avg

    def f_train(self, train_loader, test_loader):
        if self.args.resume_path:
            best_loss, best_acc, _ = self.val_epoch(test_loader)
        # start training
        for epoch in range(self.start_epoch, self.args.train_epochs + 1):
            epoch_start_time = time.time()
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            val_loss, val_acc, mean_activity = self.val_epoch(test_loader)
            elapsed = time.time() - epoch_start_time

            # write summary
            self.logger.warning(f'Epoch {epoch}, train loss {train_loss:.8f}, val loss = {val_loss:.8f}')
            self.logger.warning(f'Epoch {epoch}, train acc {train_acc:.8f}, val acc = {val_acc:.8f}')

            # # saving weights to checkpoint
            # if epoch % self.args.log_interval == 0:
            #     path = os.path.join(self.tensorboard_dir, f'{self.args.rnn_type}-Epoch-{epoch}-Acc-{val_acc}.msgpack')
            #     save_model_states(path, self.model, self.optimizer, epoch=epoch, accuracy=val_acc)

            lr = self.optimizer.lr()
            self.logger.warning('-' * 89)
            self.logger.warning(
                f'| end of epoch {epoch:3d} '
                f'| time: {elapsed:5.8f}s '
                f'| lr {lr:02.8f} '
                f'| valid loss {val_loss:5.8f} '
                f'| valid acc {val_acc:5.8f} '
                f'| mean activity {mean_activity:.8f}'
            )
            self.logger.warning('-' * 89)

            if val_acc > self.best_acc:
                self.best_acc = val_acc
                path = os.path.join(self.tensorboard_dir, 'best_model.msgpack')
                save_model_states(path, self.model, self.optimizer, epoch=epoch, accuracy=val_acc)

            self.optimizer.lr.step_epoch()

        self.logger.warning('-' * 89)
        self.logger.warning(
            f'| end of epochs '
            f'| best test acc {self.best_acc:5.8f} '
        )
        self.logger.warning('-' * 89)


def main():
    # dataset loading
    train_loader, _ = get_dvs128_train_val(args, split=1, augmentation=args.augment_data)
    test_loader = get_dvs128_test_dataset(args)

    # trainer and training
    trainer = Trainer(args)
    trainer.f_train(train_loader, test_loader)


if __name__ == "__main__":
    main()
