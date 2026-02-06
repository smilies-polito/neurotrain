# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
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

import os
import platform
import sys
import time
from typing import Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from args import parse_args

global_args = parse_args()

if not platform.platform().startswith('Windows'):
    import matplotlib

    matplotlib.use('agg')

import brainstate
import braintools
import braintrace
import brainunit as u
import jax
import jax.numpy as jnp

from data import EvidenceAccumulation
from model import _SNNEINet, SNNCubaNet, SNNCobaNet


class ExponentialSmooth(object):
    def __init__(self, decay: float = 0.8):
        self.decay = decay
        self.value = None

    def update(self, value):
        if self.value is None:
            self.value = value
        else:
            self.value = self.decay * self.value + (1 - self.decay) * value
        return self.value

    def __call__(self, value, i: int = None):
        return self.update(value)  # / (1. - self.decay ** (i + 1))


class Trainer:
    def __init__(
        self,
        target_net: _SNNEINet,
        task_loader: EvidenceAccumulation,
        args: brainstate.util.DotDict,
        lr: float,
        filepath: str | None = None
    ):
        # the network
        self.target = target_net

        # the dataset
        self.task_loader = task_loader

        # parameters
        self.args = args
        self.filepath = filepath

        # optimizer
        self.trainable_weights = self.target.states().subset(brainstate.ParamState)
        lr = braintools.optim.StepLR(lr, step_size=args.epoch_per_step, gamma=0.9)
        self.optimizer = braintools.optim.Adam(lr=lr)
        self.optimizer.register_trainable_weights(self.trainable_weights)

        self.smoother = ExponentialSmooth()

    def print(self, msg, file=None):
        if file is not None:
            print(msg, file=file)
        print(msg)

    def _loss(self, out, target):
        # MSE loss
        mse_loss = braintools.metric.softmax_cross_entropy_with_integer_labels(out, target).mean()
        # L1 regularization loss
        l1_loss = 0.
        if self.args.weight_L1 != 0.:
            leaves = self.trainable_weights.to_dict_values()
            for leaf in leaves.values():
                l1_loss += self.args.weight_L1 * jnp.sum(jnp.abs(leaf))
        return mse_loss, l1_loss

    def _acc(self, outs, target):
        pred = jnp.argmax(jnp.sum(outs, 0), 1)  # [T, B, N] -> [B, N] -> [B]
        acc = jnp.asarray(pred == target, dtype=brainstate.environ.dftype()).mean()
        return acc

    @brainstate.transform.jit(static_argnums=(0,))
    def etrace_train(self, inputs, targets):
        inputs = jnp.asarray(inputs, dtype=brainstate.environ.dftype())
        # inputs: [n_seq, n_batch, n_feat]
        n_batch = inputs.shape[1]

        # initialize the online learning model
        if self.args.method == 'esd-rtrl':
            model = braintrace.IODimVjpAlgorithm(
                self.target,
                self.args.etrace_decay,
                vjp_method=self.args.vjp_method,
            )
        elif self.args.method == 'd-rtrl':
            model = braintrace.ParamDimVjpAlgorithm(
                self.target,
                vjp_method=self.args.vjp_method,
            )
        elif self.args.method == 'hybrid':
            model = braintrace.HybridDimVjpAlgorithm(
                self.target,
                self.args.etrace_decay,
                vjp_method=self.args.vjp_method,
            )
        else:
            raise ValueError(f'Unknown online learning methods: {self.args.method}.')

        @brainstate.transform.vmap_new_states(state_tag='new', axis_size=n_batch)
        def init():
            # init target network
            brainstate.nn.init_all_states(self.target)
            # init etrace algorithm
            inp = jax.ShapeDtypeStruct(inputs.shape[2:], inputs.dtype)
            model.compile_graph(inp)
            model.show_graph()

        init()
        model = brainstate.nn.Vmap(model, vmap_states='new')

        warmup = self.args.warmup + inputs.shape[0] if self.args.warmup < 0 else self.args.warmup
        n_sim = int(warmup) if warmup > 0 else 0

        def _etrace_grad(inp):
            # call the model
            out = model(inp)
            # calculate the loss
            me, l1 = self._loss(out, targets)
            return me + l1, (out, me, l1)

        def _etrace_step(prev_grads, inp):
            # no need to return weights and states, since they are generated then no longer needed
            f_grad = brainstate.transform.grad(
                _etrace_grad,
                grad_states=self.trainable_weights,
                has_aux=True,
                return_value=True
            )
            cur_grads, local_loss, (out, mse_l, l1) = f_grad(inp)
            next_grads = jax.tree.map(lambda a, b: a + b, prev_grads, cur_grads)
            return next_grads, (out, mse_l, l1)

        def _etrace_train(inputs_):
            # forward propagation
            grads = jax.tree.map(jnp.zeros_like, self.trainable_weights.to_dict_values())
            grads, (outs, mse_ls, l1_ls) = brainstate.transform.scan(_etrace_step, grads, inputs_)
            acc = self._acc(outs, targets)

            grads = brainstate.functional.clip_grad_norm(grads, 1.)
            self.optimizer.update(grads)
            # accuracy
            return mse_ls.mean(), l1_ls.mean(), acc

        # running indices
        if n_sim > 0:
            brainstate.transform.for_loop(model, inputs[:n_sim])
            r = _etrace_train(inputs[n_sim:])
        else:
            r = _etrace_train(inputs)
        mem = jax.pure_callback(
            lambda: jax.devices()[0].memory_stats()['bytes_in_use'] / 1024 / 1024 / 1024,
            jax.ShapeDtypeStruct((), brainstate.environ.dftype())
        )
        return r + (mem,)

    @brainstate.transform.jit(static_argnums=(0,))
    def bptt_train(self, inputs, targets) -> Tuple:
        inputs = jnp.asarray(inputs, dtype=brainstate.environ.dftype())
        # inputs: [n_seq, n_batch, n_feat]
        brainstate.nn.vmap_init_all_states(self.target, axis_size=inputs.shape[1], state_tag='new')
        model = brainstate.nn.Vmap(self.target, vmap_states='new')

        warmup = self.args.warmup + inputs.shape[0] if self.args.warmup < 0 else self.args.warmup
        n_sim = int(warmup) if warmup > 0 else 0

        def _step_run(inp):
            out = model(inp)
            return self._loss(out, targets), out

        def _bptt_grad():
            (mse_losses, l1_losses), outs = brainstate.transform.for_loop(_step_run, inputs)
            mse_losses = mse_losses[n_sim:].mean()
            l1_losses = l1_losses[n_sim:].mean()
            acc = self._acc(outs[n_sim:], targets)
            return mse_losses + l1_losses, (mse_losses, l1_losses, acc)

        f_grad = brainstate.transform.grad(
            _bptt_grad,
            grad_states=self.trainable_weights,
            has_aux=True,
            return_value=True
        )
        grads, loss, (mse_losses, l1_losses, acc) = f_grad()
        grads = brainstate.functional.clip_grad_norm(grads, 1.)
        self.optimizer.update(grads)
        mem = jax.pure_callback(
            lambda: jax.devices()[0].memory_stats()['bytes_in_use'] / 1024 / 1024 / 1024,
            jax.ShapeDtypeStruct((), brainstate.environ.dftype())
        )
        return mse_losses, l1_losses, acc, mem

    def f_sim(self):
        inputs, outputs = next(iter(self.task_loader))
        inputs = jnp.asarray(inputs, dtype=brainstate.environ.dftype()).transpose(1, 0, 2)
        self.target.visualize(inputs)

    def f_train(self):
        file = None
        if self.filepath is not None:
            os.makedirs(self.filepath, exist_ok=True)
            file = open(f'{self.filepath}/loss.txt', 'w')
        self.print(self.args, file=file)

        acc_max = 0.
        t0 = time.time()
        for bar_idx, (inputs, outputs) in enumerate(self.task_loader):
            if bar_idx > self.args.epochs:
                break

            inputs = jnp.asarray(inputs, dtype=brainstate.environ.dftype()).transpose(1, 0, 2)
            outputs = jnp.asarray(outputs, dtype=brainstate.environ.ditype())

            fun = (self.bptt_train if self.args.method == 'bptt' else self.etrace_train)
            mse_ls, l1_ls, acc, mem = fun(inputs, outputs)
            self.optimizer.lr.step_epoch()
            desc = (
                f'Batch {bar_idx:2d}, '
                f'CE={float(mse_ls):.8f}, '
                f'L1={float(l1_ls):.6f}, '
                f'acc={float(acc):.6f}, '
                f'lr={self.optimizer.lr():.6f}, '
                f'mem={mem:.6f} GB, '
                f'time={time.time() - t0:.2f} s'
            )
            self.print(desc, file=file)
            self.smoother.update(acc)

            t0 = time.time()
            if self.smoother.value >= self.args.threshold:
                if self.filepath is not None:
                    braintools.file.msgpack_save(
                        f'{self.filepath}/best_model.msgpack',
                        self.target.states(brainstate.ParamState).to_nest()
                    )
                    self.target.visualize(
                        inputs,
                        filename=f'{self.filepath}/train-results-{bar_idx}.png'
                    )
                print(f'Accuracy reaches {self.args.threshold * 100}% at {bar_idx}th epoch. Stop training.')
                break
        if file is not None:
            file.close()


def training():
    gargs = global_args
    brainstate.environ.set(dt=gargs.dt)

    # filepath
    now = time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(int(round(time.time() * 1000)) / 1000))
    exp_name = 'results/'
    if gargs.exp_name:
        exp_name += gargs.exp_name + '/'
    filepath = f'{exp_name}/{gargs.method}'
    if gargs.method != 'bptt':
        filepath += f'-{gargs.vjp_method}'
    if gargs.method == 'esd-rtrl':
        filepath = f'{filepath}-etrace={gargs.etrace_decay}'
    filepath = (
        f'{filepath}/'
        f'tau_I1={gargs.tau_I1}-A1={gargs.A1}-tau_I2={gargs.tau_I2}-A2={gargs.A2}-'
        f'tau_neu={gargs.tau_neu}-tau_syn={gargs.tau_syn}-{now}'
    )

    # data
    with brainstate.environ.context(dt=brainstate.environ.get_dt() * u.ms):
        task_loader = EvidenceAccumulation(batch_size=gargs.batch_size)
    gargs.warmup = -(task_loader.t_recall / u.ms / brainstate.environ.get_dt())

    # network
    cls = SNNCobaNet if gargs.net == 'coba' else SNNCubaNet
    net = cls(
        task_loader.num_inputs,
        gargs.n_rec,
        task_loader.num_outputs,
        args=gargs,
        filepath=filepath,
    )

    # trainer
    trainer = Trainer(net, task_loader, gargs, lr=gargs.lr, filepath=filepath)
    if gargs.mode == 'sim':
        trainer.f_sim()
    else:
        trainer.f_train()


if __name__ == '__main__':
    training()
