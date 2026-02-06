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


import argparse
import copy
import os
import pickle
import time
from functools import partial
from typing import Callable, List, Union, Sequence


def _set_gpu_preallocation(mode: float):
    """
    GPU memory allocation.

    If preallocation is enabled, this makes JAX preallocate ``percent`` of the total GPU memory,
    instead of the default 75%. Lowering the amount preallocated can fix OOMs that occur when the JAX program starts.
    """
    assert isinstance(mode, float) and 0. <= mode < 1., f'GPU memory preallocation must be in [0., 1.]. But got {mode}.'
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(mode)


def _set_gpu_device(device_ids: Union[str, int, Sequence[int]]):
    if isinstance(device_ids, int):
        device_ids = str(device_ids)
    elif isinstance(device_ids, (tuple, list)):
        device_ids = ','.join([str(d) for d in device_ids])
    elif isinstance(device_ids, str):
        device_ids = device_ids
    else:
        raise ValueError
    os.environ['CUDA_VISIBLE_DEVICES'] = device_ids


def define_device_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--devices', type=str, default='0', help='The GPU device ids.')
    args, _ = parser.parse_known_args()
    _set_gpu_device(args.devices)
    _set_gpu_preallocation(0.99)
    return args


define_device_args()

import braintrace
import braintools
import jax

import brainstate
import jax.numpy as jnp
import numpy as np
from torch.utils.data import DataLoader

from rsnn_models_and_data import (
    LIF_STPExpCu_Dense_Layer,
    LIF_STDExpCu_Dense_Layer,
    LIF_ExpCu_Dense_Layer,
    ALIF_STPExpCu_Dense_Layer,
    ALIF_STDExpCu_Dense_Layer,
    ALIF_Delta_Dense_Layer,
    LIF_Delta_Dense_Layer,
    ALIF_ExpCu_Dense_Layer,
    ETraceDenseNet,
    NetWithMemSpkRegularize,
    visualize,
    get_snn_data,
    RandomDataset,
    cosine_similarity,
)
from rsnn_hidden_jacobian_cosine import model_setting_1, model_setting_2

brainstate.environ.set(dt=1.0)


def compare_gradient_of_five_layer_net_by_random_data():
    brainstate.random.seed(1)

    # sets = dict(n_layer=1, model='lif_std_expcu_dense', rec_wscale=1.0, ff_wscale=1.0, spk_reset='soft', n_rank=80, tau_std=200.0, tau_syn=10.0, tau_v=10.,)
    # sets = dict(n_layer=2, model='lif_std_expcu_dense', rec_wscale=1.0, ff_wscale=1.0, spk_reset='soft', n_rank=80, tau_std=200.0, tau_syn=10.0, tau_v=10.,)
    # sets = dict(n_layer=3, model='lif_std_expcu_dense', rec_wscale=2.0, ff_wscale=2.0, spk_reset='soft', n_rank=80, tau_std=200.0, tau_syn=10.0, tau_v=10.,)
    sets = dict(
        n_layer=5,
        model='lif_std_expcu_dense',
        rec_wscale=2.0,
        ff_wscale=2.0,
        spk_reset='soft',
        n_rank=80,
        tau_std=200.0,
        tau_syn=10.0,
        tau_v=10.,
    )

    sets = dict(
        n_layer=5,
        model='alif_std_expcu_dense',
        rec_wscale=2.0,
        ff_wscale=2.0,
        spk_reset='soft',
        n_rank=80,
        tau_std=200.0,
        tau_syn=10.0,
        tau_v=10.,
    )

    # sets = dict(n_layer=1, model='lif_expcu_dense', rec_wscale=1.0, ff_wscale=1.0, spk_reset='soft', n_rank=80, tau_syn=10.0, tau_v=10.,)
    # sets = dict(n_layer=2, model='lif_expcu_dense', rec_wscale=1.0, ff_wscale=1.0, spk_reset='soft', n_rank=80, tau_syn=10.0, tau_v=10.,)
    # sets = dict(n_layer=3, model='lif_expcu_dense', rec_wscale=1.0, ff_wscale=1.0, spk_reset='soft', n_rank=80, tau_syn=10.0, tau_v=10.,)
    sets = dict(
        n_layer=5,
        model='lif_expcu_dense',
        rec_wscale=1.0,
        ff_wscale=1.0,
        spk_reset='soft',
        n_rank=80,
        tau_syn=10.0,
        tau_v=10.,
    )

    # sets = dict(n_layer=1, model='alif_expcu_dense', rec_wscale=1, ff_wscale=1, spk_reset='soft', n_rank=80, tau_syn=10.0, tau_v=10., tau_a=200.0, )
    # sets = dict(n_layer=2, model='alif_expcu_dense', rec_wscale=1, ff_wscale=1, spk_reset='soft', n_rank=80, tau_syn=10.0, tau_v=10., tau_a=200.0, )
    # sets = dict(n_layer=3, model='alif_expcu_dense', rec_wscale=1, ff_wscale=1, spk_reset='soft', n_rank=80, tau_syn=10.0, tau_v=10., tau_a=200.0, )
    sets = dict(
        n_layer=5,
        model='alif_expcu_dense',
        rec_wscale=1,
        ff_wscale=1,
        spk_reset='soft',
        n_rank=80,
        tau_syn=10.0,
        tau_v=10.,
        tau_a=200.0,
    )

    # sets = dict(n_layer=3, model='if_delta_dense', rec_wscale=4.0, ff_wscale=4.0, spk_reset='hard', n_rank=80, )
    # sets = dict(n_layer=4, model='if_delta_dense', rec_wscale=4.0, ff_wscale=4.0, spk_reset='hard', n_rank=80, )
    # sets = dict(n_layer=4, model='if_delta_dense', rec_wscale=2., ff_wscale=2., spk_reset='hard', n_rank=80, tau_v=10.)
    sets = dict(n_layer=5, model='if_delta_dense', rec_wscale=1., ff_wscale=1., spk_reset='soft', n_rank=80, tau_v=10.)
    # sets = dict(n_layer=8, model='if_delta_dense', rec_wscale=1., ff_wscale=1., spk_reset='soft', n_rank=80, tau_v=10.)

    sets = dict(n_layer=5, model='alif_delta_dense', rec_wscale=1., ff_wscale=1., spk_reset='soft', n_rank=80,
                tau_v=10.)

    print(sets)
    args = brainstate.util.DotDict(n_layer=2, model='lif_expcu_dense',
                                   lr=0.0005, epochs=200, rec_wscale=10.0, ff_wscale=10.0,
                                   dt=0.1, loss='cel', mode='train', dataset='N-MNIST',
                                   tau_v=80.0, tau_v_sigma=1.0, tau_a=200.0, beta=1.0, V_th=1.0, spk_reset='hard',
                                   tau_std=200.0, tau_syn=10.0, tau_o=80.0, spk_reg=0.0,
                                   spk_reg_rate=10.0, v_reg=0.0, v_reg_low=-0.4, v_reg_high=1.0, weight_L1=0.0,
                                   weight_L2=0.0, method='expsm_diag', n_rank=5, warmup_ratio=0.0,
                                   record_period=10, data_length=200, devices='0')
    args.update(sets)

    n_in = 100
    n_rec = 200
    n_out = 10
    n_batch = 16
    n_seq = 1000
    n_rank = 10
    model = ETraceDenseNet(n_in, n_rec, n_out, args, spk_fun=braintools.surrogate.ReluGrad(width=0.5))
    weights = model.states().subset(brainstate.ParamState)

    def step_to_visualize(i, inp):
        with brainstate.environ.context(i=i, t=i * brainstate.environ.get_dt(), fit=True):
            out = model(inp)
        return model.visualize_variables()

    def loss_fun(i, inp, target):
        with brainstate.environ.context(i=i, t=i * brainstate.environ.get_dt(), fit=True):
            out = model(inp)
        return braintools.metric.softmax_cross_entropy_with_integer_labels(out, target).mean()

    def compute_etrace_grad(inputs, targets):
        # etrace = nn.DiagExpSmOnAlgorithm(partial(loss_fun, target=targets), decay_or_rank=n_rank)
        etrace = braintrace.ParamDimVjpAlgorithm(partial(loss_fun, target=targets), )
        etrace.compile_graph(0, jax.ShapeDtypeStruct((n_batch, n_in), brainstate.environ.dftype()))
        f_grad = brainstate.transform.grad(etrace, grad_states=weights)

        def step(prev_grad, inp):
            i, inp = inp
            cur_grads = f_grad(i, inp, running_index=i)
            new_grads = jax.tree.map(jnp.add, cur_grads, prev_grad)
            return new_grads, None

        grads = jax.tree.map(jnp.zeros_like, weights.to_dict_values())
        grads, _ = brainstate.transform.scan(step, grads, (indices, inputs))
        return grads

    def compute_bptt_grad(inputs, targets):
        def global_loss():
            losses = brainstate.transform.for_loop(partial(loss_fun, target=targets), indices, inputs)
            return losses.sum()

        return brainstate.transform.grad(global_loss, grad_states=weights)()

    indices = jnp.arange(n_seq)
    inp_spks = brainstate.random.random((n_seq, n_batch, n_in)) < 0.1
    inp_spks = jnp.asarray(inp_spks, dtype=brainstate.environ.dftype())
    target_outs = brainstate.random.randint(0, n_out, (n_batch,))

    brainstate.nn.init_all_states(model)
    visualize_outs = brainstate.transform.for_loop(step_to_visualize, indices, inp_spks)
    visualize.plot_multilayer_lif(inp_spks,
                                  visualize_outs['rec_s'],
                                  visualize_outs['rec_v'],
                                  visualize_outs['out_v'],
                                  show=True,
                                  num_vis=1)

    brainstate.nn.init_all_states(model, n_batch)
    etrace_grads = compute_etrace_grad(inp_spks, target_outs)
    etrace_grads = jax.tree.map(lambda x: x, etrace_grads)

    brainstate.nn.init_all_states(model, n_batch)
    bptt_grads = compute_bptt_grad(inp_spks, target_outs)

    from pprint import pprint
    pprint(jax.tree.map(cosine_similarity, etrace_grads, bptt_grads))


class ETraceDenseNetV2(NetWithMemSpkRegularize):
    def __init__(self, n_in, n_rec, n_out, model_cls, args, spk_fun: Callable = braintools.surrogate.ReluGrad()):
        super().__init__()

        # arguments
        self.n_in = n_in
        self.n_rec = n_rec
        self.n_out = n_out
        self.n_layer = args.pop('n_layer')
        tau_o = args.pop('tau_o')
        kwargs = args.pop('kwargs', dict())

        rec_init = braintools.init.KaimingNormal(scale=args.pop('rec_wscale'))
        ff_init = braintools.init.KaimingNormal(scale=args.pop('ff_wscale'))

        # recurrent layers
        self.rec_layers = []
        for layer_idx in range(self.n_layer):
            rec = model_cls(
                n_rec=n_rec,
                n_in=n_in,
                spk_fun=spk_fun,
                rec_init=rec_init,
                ff_init=ff_init,
                **args,
                **kwargs,
            )
            n_in = n_rec
            self.rec_layers.append(rec)

        # output layer
        self.out = braintrace.nn.LeakyRateReadout(
            in_size=n_rec,
            out_size=n_out,
            tau=tau_o,
            w_init=braintools.init.KaimingNormal(),
            name='out',
        )

    def update(self, x):
        for i in range(self.n_layer):
            x = self.rec_layers[i](x)
        return self.out(x)

    def visualize_variables(self) -> dict:
        neurons = tuple(self.nodes().subset(brainpy.state.Neuron).unique().values())
        outs = {
            'out_v': self.out.r.value,
            'rec_v': [l.V.value for l in neurons],
            'rec_s': [l.spike for l in neurons],
        }
        return outs


def _compare(
    dataloader,
    model: ETraceDenseNetV2,
    decay_or_rank: List[int | float],
    x_func: Callable,
    y_func: Callable,
    n_data: int = 500,
    show: bool = False,
    frac_sim: float = 0.,
):
    weights = model.states().subset(brainstate.ParamState)

    def step_to_visualize(i, inp):
        with brainstate.environ.context(i=i, t=i * brainstate.environ.get_dt(), fit=True):
            out = model(inp)
        return model.visualize_variables()

    def run_to_visualize(inputs):
        brainstate.nn.init_all_states(model)
        indices = np.arange(inputs.shape[0])
        visualize_outs = brainstate.transform.for_loop(step_to_visualize, indices, inputs)
        visualize.plot_multilayer_lif(
            inputs,
            visualize_outs['rec_s'],
            visualize_outs['rec_v'],
            visualize_outs['out_v'],
            show=True,
            num_vis=1
        )

    def sim_step(i, inp):
        with brainstate.environ.context(i=i, t=i * brainstate.environ.get_dt(), fit=True):
            out = model(inp)
        return out

    class Model(brainstate.nn.Module):
        def __init__(self, target):
            super().__init__()
            self.model = model
            self.target = target

        def update(self, i, inp):
            with brainstate.environ.context(i=i, t=i * brainstate.environ.get_dt(), fit=True):
                out = model(inp)
            return braintools.metric.softmax_cross_entropy_with_integer_labels(out, self.target).mean()

    @brainstate.transform.jit(static_argnums=(0, 1, 2))
    def compute_etrace_grad(method: str, n_rank, n_sim, inputs, targets):
        brainstate.nn.init_all_states(model)

        loss_fn = Model(target=targets)

        if method == 'diag_expsm':
            etrace = braintrace.IODimVjpAlgorithm(loss_fn, decay_or_rank=n_rank)
        elif method == 'diag':
            etrace = braintrace.ParamDimVjpAlgorithm(loss_fn)
        else:
            raise ValueError(f'Unknown method {method}')
        etrace.compile_graph(0, jax.ShapeDtypeStruct([model.n_in], brainstate.environ.dftype()))
        f_grad = brainstate.transform.grad(etrace, grad_states=weights)

        def train_step(prev_grad, inp):
            i, inp = inp
            cur_grads = f_grad(i, inp)
            new_grads = jax.tree.map(jnp.add, cur_grads, prev_grad)
            return new_grads, None

        grads = jax.tree.map(jnp.zeros_like, weights.to_dict_values())
        indices = np.arange(inputs.shape[0])
        if n_sim > 0:
            _ = brainstate.transform.for_loop(sim_step, indices[:n_sim], inputs[:n_sim])
            grads, _ = brainstate.transform.scan(train_step, grads, (indices[n_sim:], inputs[n_sim:]))
        else:
            grads, _ = brainstate.transform.scan(train_step, grads, (indices, inputs))
        return grads

    @brainstate.transform.jit(static_argnums=(0,))
    def compute_bptt_grad(n_sim, inputs, targets):
        def global_loss():
            loss_fn = Model(target=targets)

            indices = np.arange(inputs.shape[0])
            if n_sim > 0:
                _ = brainstate.transform.for_loop(sim_step, indices[:n_sim], inputs[:n_sim])
                losses = brainstate.transform.for_loop(loss_fn,
                                                       indices[n_sim:],
                                                       inputs[n_sim:])
            else:
                losses = brainstate.transform.for_loop(loss_fn, indices, inputs)
            return losses.sum()

        brainstate.nn.init_all_states(model)
        return brainstate.transform.grad(global_loss, grad_states=weights)()

    def flatten(tree):
        leaves = jax.tree.leaves(tree)
        return jnp.concatenate([jnp.ravel(x) for x in leaves])

    def compare_cosine(grads1, grads2):
        r = dict()
        for k in grads1:
            r[k] = cosine_similarity(flatten(grads1[k]), flatten(grads2[k]))
        return r

    def append(all_res, res):
        for k, v in res.items():
            for kk, vv in v.items():
                if kk not in all_res[k]:
                    all_res[k][kk] = []
                all_res[k][kk].append(vv)
        return all_res

    final_results = {'diag': dict()}
    for rank in decay_or_rank:
        final_results[f'diag_expsm_{rank}'] = dict()
    i = 0
    for inp_spks, target_outs in dataloader:
        i += 1
        if i > n_data:
            break

        inp_spks = x_func(inp_spks)
        target_outs = y_func(target_outs)
        inp_spks = jnp.squeeze(inp_spks)
        target_outs = jnp.squeeze(target_outs)
        num_sim = int(frac_sim * inp_spks.shape[0]) if frac_sim > 0 else 0

        if show:
            run_to_visualize(inp_spks)

        bptt_grads = compute_bptt_grad(num_sim, inp_spks, target_outs)

        rr = {}
        diag_grads = compute_etrace_grad('diag', None, num_sim, inp_spks, target_outs)
        diag_r = compare_cosine(diag_grads, bptt_grads)
        rr['diag'] = diag_r

        for rank in decay_or_rank:
            diag_expsm_grads = compute_etrace_grad('diag_expsm', rank, num_sim, inp_spks, target_outs)
            diag_exp_r = compare_cosine(diag_expsm_grads, bptt_grads)
            rr[f'diag_expsm_{rank}'] = diag_exp_r
        final_results = append(final_results, rr)

    final_results = {k: {kk: np.asarray(v)
                         for kk, v in final_results[k].items()}
                     for k in final_results}
    return final_results


def compare_gradient_by_neuromorphic_data(fn='analysis/jac_cosine_sim', frac_sim=0.99):
    if not os.path.exists(fn):
        os.makedirs(fn)
    # brainstate.random.seed(1)

    n_rec = 200
    n_layer = 1
    results = dict()

    for data_name, _, model_cls, args in model_setting_1:
        datalengths = [200, 400, 600, 1000] if data_name == 'gesture' else [None]
        for datalength in datalengths:
            data_args = brainstate.util.DotDict(
                batch_size=1,
                n_data_worker=1,
                drop_last=False,
                dataset=data_name,
                data_length=datalength,
                shuffle=False
            )
            dataset = get_snn_data(data_args)

            print(f'Processing {data_name} {datalength} with {model_cls.__name__}')

            args__ = brainstate.util.DotDict(n_layer=n_layer, tau_o=10, )
            for k, v in args.items():
                args__[k] = v
            model = ETraceDenseNetV2(
                n_in=int(np.prod(dataset.in_shape)),
                n_rec=n_rec,
                n_out=dataset.out_shape,
                model_cls=model_cls,
                args=copy.copy(args__)
            )
            r = _compare(
                dataset.train_loader,
                model,
                decay_or_rank=[4, 10, 20, 40, 100],
                x_func=dataset.input_process,
                y_func=dataset.label_process,
                n_data=200,
                frac_sim=frac_sim,
            )

            print(f'{data_name} {datalength} {model_cls.__name__}')
            print(r)
            results[(data_name, model_cls.__name__, datalength)] = r

    now = time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(int(round(time.time() * 1000)) / 1000))
    filename = f'gradient-cosine-n_rec={n_rec}-frac_sim={frac_sim}-{now}.pkl'
    with open(os.path.join(fn, filename), 'wb') as fout:
        pickle.dump(results, fout)


def compare_gradient_multi_layer_by_neuromorphic_data(
    fn='analysis/jac_cosine_sim',
    frac_sim: float = 0.99,
    n_layer: int = 3,
):
    if not os.path.exists(fn):
        os.makedirs(fn)
    # brainstate.random.seed(1)

    n_rec = 200
    results = dict()

    model_setting_1 = [
        ('SHD', None, LIF_Delta_Dense_Layer, dict(rec_wscale=0.1, ff_wscale=0.8, tau_mem=10.)),
        ('SHD', None, LIF_ExpCu_Dense_Layer,
         dict(rec_wscale=4., ff_wscale=20. * 8., tau_mem=10., kwargs=dict(tau_syn=10.0))),
        ('SHD', None, LIF_STDExpCu_Dense_Layer,
         dict(rec_wscale=8., ff_wscale=20. * 8., tau_mem=10., kwargs=dict(tau_std=200.0, tau_syn=10.0))),
        ('SHD', None, LIF_STPExpCu_Dense_Layer,
         dict(rec_wscale=5., ff_wscale=20. * 8., tau_mem=10., kwargs=dict(tau_syn=10, tau_f=10, tau_d=100))),
        ('SHD', None, ALIF_Delta_Dense_Layer, dict(rec_wscale=0.5, ff_wscale=0.5 * 4., tau_mem=10.)),
        ('SHD', None, ALIF_ExpCu_Dense_Layer,
         dict(rec_wscale=10., ff_wscale=20. * 8., tau_mem=10., kwargs=dict(tau_syn=10.0, tau_a=100.0))),
        ('SHD', None, ALIF_STDExpCu_Dense_Layer,
         dict(rec_wscale=10., ff_wscale=30. * 8., tau_mem=10., kwargs=dict(tau_std=200., tau_syn=10))),
        ('SHD', None, ALIF_STPExpCu_Dense_Layer,
         dict(rec_wscale=6, ff_wscale=40 * 8., tau_mem=5, kwargs=dict(tau_syn=5, tau_f=10, tau_d=100, tau_a=100))),
    ]

    for data_name, _, model_cls, args in model_setting_1:
        # if data_name in ['N-MNIST', 'gesture']:
        #   continue
        datalengths = [200, ] if data_name == 'gesture' else [None]
        for datalength in datalengths:
            data_args = brainstate.util.DotDict(
                batch_size=1,
                n_data_worker=1,
                drop_last=False,
                dataset=data_name,
                data_length=datalength,
                shuffle=True
            )
            dataset = get_snn_data(data_args)

            print(f'Processing {data_name} {datalength} with {model_cls.__name__} {n_layer} layers')

            args__ = brainstate.util.DotDict(n_layer=n_layer, tau_o=10, )
            for k, v in args.items():
                args__[k] = v
            # if data_name == 'N-MNIST':
            #   args__['ff_wscale'] *= 2.
            # elif data_name == 'gesture':
            #   args__['ff_wscale'] *= 4.
            # # elif data_name == 'SHD':
            # #   args__['ff_wscale'] *= 4.

            model = ETraceDenseNetV2(
                n_in=int(np.prod(dataset.in_shape)),
                n_rec=n_rec,
                n_out=dataset.out_shape,
                model_cls=model_cls,
                args=copy.copy(args__)
            )
            r = _compare(
                dataset.train_loader,
                model,
                # decay_or_rank=[4, 10, 20, 40, 100],
                decay_or_rank=[4, 10, 20, 40],
                x_func=dataset.input_process,
                y_func=dataset.label_process,
                n_data=100,
                # show=True,
                frac_sim=frac_sim,
            )

            print(f'{data_name} {datalength} {model_cls.__name__}')
            print(r)
            results[(data_name, model_cls.__name__, datalength)] = r
            brainstate.util.clear_buffer_memory(compilation=True, array=True)

    now = time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(int(round(time.time() * 1000)) / 1000))
    filename = f'data/gradient-cosine-n_layer={n_layer}-n_rec={n_rec}-frac_sim={frac_sim}-{now}.pkl'
    with open(os.path.join(fn, filename), 'wb') as fout:
        pickle.dump(results, fout)


def compare_gradient_on_random_data():
    n_in = 100
    n_rec = 200
    n_out = 2
    n_layer = 1
    n_seq = 1000

    args = brainstate.util.DotDict(rec_wscale=0.1, ff_wscale=0.1, tau_mem=10.)
    model_cls = LIF_Delta_Dense_Layer

    args = brainstate.util.DotDict(rec_wscale=40., ff_wscale=100., tau_mem=10., kwargs=dict(tau_syn=10.0, tau_a=100.0))
    model_cls = ALIF_ExpCu_Dense_Layer

    args__ = brainstate.util.DotDict(n_layer=n_layer, tau_o=10)
    for k, v in args.items():
        args__[k] = v
    model = ETraceDenseNetV2(n_in=n_in,
                             n_rec=n_rec,
                             n_out=n_out,
                             model_cls=model_cls,
                             args=copy.copy(args__))

    data = RandomDataset(n_seq, n_in, n_out, prob=0.01)
    data_loader = DataLoader(data, batch_size=1, shuffle=False)

    r = _compare(
        data_loader,
        model,
        decay_or_rank=[4, 10, 20, 40, 100],
        x_func=lambda x: jax.numpy.transpose(jnp.asarray(x), (1, 0, 2)),
        y_func=lambda x: jnp.asarray(x),
        show=False,
        n_data=2,
        # n_sim=0,
        frac_sim=0.5,
    )
    from pprint import pprint
    pprint(r)


def compare_gradient_approx_when_recurrent_size_increases(fn='analysis/jac_cosine_sim', frac_sim=0.99):
    brainstate.random.seed(1)
    spk_fun = braintools.surrogate.ReluGrad(width=1.)

    rec_sizes = [10, 50, 100, 300, 500, 1000, 2000, 4000, 8000, 10000, 20000, 40000]
    rec_sizes = [10, 50, 100, 300, 500, 1000, 2000, 4000]

    n_layer = 1
    final_results = dict()

    for data_name, data_len, model_cls, model_args in model_setting_2:
        for n_rec in rec_sizes:
            # datalengths = [200, 400, 600, 1000] if data_name == 'gesture' else [None]
            datalengths = [200, ] if data_name == 'gesture' else [None]
            for datalength in datalengths:
                # data
                data_args = brainstate.util.DotDict(
                    batch_size=1,
                    n_data_worker=1,
                    drop_last=False,
                    dataset=data_name,
                    shuffle=False,
                    data_length=datalength
                )
                dataset = get_snn_data(data_args)

                # model
                args__ = brainstate.util.DotDict(n_layer=n_layer, tau_o=10)
                for k, v in model_args.items():
                    args__[k] = v
                model = ETraceDenseNetV2(
                    n_in=int(np.prod(dataset.in_shape)),
                    n_rec=n_rec,
                    n_out=dataset.out_shape,
                    model_cls=model_cls,
                    args=copy.copy(args__),
                    spk_fun=spk_fun,
                )

                print(f'{data_name} {datalength} {model_cls.__name__} {n_rec}')

                # computing
                print(data_args)
                try:
                    r = _compare(
                        dataset.train_loader,
                        model,
                        decay_or_rank=[4, 10, 20, 40, 100],
                        x_func=dataset.input_process,
                        y_func=dataset.label_process,
                        n_data=200,
                        frac_sim=frac_sim,
                        show=False
                    )
                    key = (data_name, datalength, n_rec, model_cls.__name__)
                    final_results[key] = r
                    print(r)
                    brainstate.util.clear_buffer_memory()
                except Exception as e:
                    print(e)
                print()

    now = time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(int(round(time.time() * 1000)) / 1000))
    with open(os.path.join(fn, f'gradient-frac_sim={frac_sim}-cosine-rec-size-increase-{now}.pkl'), 'wb') as fout:
        pickle.dump(final_results, fout)


def compare_gradient_approx_when_ff_conn_increases(fn='analysis/jac_cosine_sim', frac_sim=0.99):
    brainstate.random.seed(1)
    spk_fun = braintools.surrogate.ReluGrad(width=1.)
    n_rec = 200
    n_layer = 1

    final_results = dict()
    for data_name, data_len, model_cls, model_args in model_setting_2:
        for ff_frac in [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0]:
            datalengths = [200, 400, 600, 1000] if data_name == 'gesture' else [None]
            # datalengths = [200,] if data_name == 'gesture' else [None]
            for datalength in datalengths:
                # data
                data_args = brainstate.util.DotDict(
                    batch_size=1,
                    n_data_worker=1,
                    drop_last=False,
                    dataset=data_name,
                    shuffle=False,
                    data_length=datalength
                )
                dataset = get_snn_data(data_args)

                # model
                args__ = brainstate.util.DotDict(n_layer=n_layer, tau_o=10)
                for k, v in model_args.items():
                    args__[k] = v
                args__['ff_wscale'] = args__['ff_wscale'] * ff_frac
                model = ETraceDenseNetV2(
                    n_in=int(np.prod(dataset.in_shape)),
                    n_rec=n_rec,
                    n_out=dataset.out_shape,
                    model_cls=model_cls,
                    args=copy.copy(args__),
                    spk_fun=spk_fun,
                )

                print(f'{data_name} {datalength} {model_cls.__name__} {ff_frac}')

                # computing
                print(data_args)
                try:
                    r = _compare(
                        dataset.train_loader,
                        model,
                        decay_or_rank=[4, 10, 20, 40, 100],
                        x_func=dataset.input_process,
                        y_func=dataset.label_process,
                        n_data=200,
                        frac_sim=frac_sim,
                        show=False
                    )
                    key = (data_name, datalength, ff_frac, model_cls.__name__)
                    final_results[key] = r
                    print(r)
                    brainstate.util.clear_buffer_memory()
                except Exception as e:
                    print(e)
                print()

    now = time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(int(round(time.time() * 1000)) / 1000))
    with open(os.path.join(fn, f'gradient-frac_sim={frac_sim}-cosine-ff-conn-increase-{now}.pkl'), 'wb') as fout:
        pickle.dump(final_results, fout)


def compare_gradient_approx_when_rec_conn_increases(fn='analysis/jac_cosine_sim', frac_sim=0.99):
    brainstate.random.seed(1)
    spk_fun = braintools.surrogate.ReluGrad(width=1.)
    n_rec = 200
    n_layer = 1

    final_results = dict()
    for data_name, data_len, model_cls, model_args in model_setting_2:
        for rec_frac in [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0]:
            datalengths = [200, 400, 600, 1000] if data_name == 'gesture' else [None]
            # datalengths = [200,] if data_name == 'gesture' else [None]
            for datalength in datalengths:
                # data
                data_args = brainstate.util.DotDict(
                    batch_size=1,
                    n_data_worker=1,
                    drop_last=False,
                    dataset=data_name,
                    shuffle=False,
                    data_length=datalength
                )
                dataset = get_snn_data(data_args)

                # model
                args__ = brainstate.util.DotDict(n_layer=n_layer, tau_o=10)
                for k, v in model_args.items():
                    args__[k] = v
                args__['rec_wscale'] = args__['rec_wscale'] * rec_frac
                model = ETraceDenseNetV2(
                    n_in=int(np.prod(dataset.in_shape)),
                    n_rec=n_rec,
                    n_out=dataset.out_shape,
                    model_cls=model_cls,
                    args=copy.copy(args__),
                    spk_fun=spk_fun,
                )

                print(f'{data_name} {datalength} {model_cls.__name__} {n_layer}')

                # computing
                print(data_args)
                try:
                    r = _compare(
                        dataset.train_loader,
                        model,
                        decay_or_rank=[4, 10, 20, 40, 100],
                        x_func=dataset.input_process,
                        y_func=dataset.label_process,
                        n_data=200,
                        frac_sim=frac_sim,
                        show=False
                    )
                    key = (data_name, datalength, rec_frac, model_cls.__name__)
                    final_results[key] = r
                    print(r)
                    brainstate.util.clear_buffer_memory()
                except Exception as e:
                    print(e)
                print()

    now = time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(int(round(time.time() * 1000)) / 1000))
    with open(os.path.join(fn, f'gradient-frac_sim={frac_sim}-cosine-rec-conn-increase-{now}.pkl'), 'wb') as fout:
        pickle.dump(final_results, fout)


if __name__ == '__main__':
    pass

    # compare_gradient_of_five_layer_net_by_random_data()

    compare_gradient_multi_layer_by_neuromorphic_data(frac_sim=0.0, n_layer=5)

    # compare_gradient_by_neuromorphic_data(frac_sim=0.)

    # compare_gradient_multi_layer_by_neuromorphic_data(frac_sim=0., n_layer=3)
    # compare_gradient_multi_layer_by_neuromorphic_data(frac_sim=0., n_layer=4)

    # compare_gradient_multi_layer_by_neuromorphic_data(frac_sim=0.9, n_layer=3)
    # compare_gradient_multi_layer_by_neuromorphic_data(frac_sim=0.9, n_layer=4)

    # compare_gradient_on_random_data()

    # compare_gradient_approx_when_recurrent_size_increases(frac_sim=0.)
    # compare_gradient_approx_when_recurrent_size_increases(frac_sim=0.99)

    # compare_gradient_approx_when_ff_conn_increases(frac_sim=0.)
    # compare_gradient_approx_when_ff_conn_increases(frac_sim=0.99)

    # compare_gradient_approx_when_rec_conn_increases(frac_sim=0.)
    # compare_gradient_approx_when_rec_conn_increases(frac_sim=0.99)
