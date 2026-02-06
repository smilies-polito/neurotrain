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
import os
import platform

os.environ['JAX_TRACEBACK_FILTERING'] = 'off'


def _set_gpu_preallocation(mode: float):
    """GPU memory allocation.

    If preallocation is enabled, this makes JAX preallocate ``percent`` of the total GPU memory,
    instead of the default 75%. Lowering the amount preallocated can fix OOMs that occur when the JAX program starts.
    """
    assert isinstance(mode, float) and 0. <= mode < 1., f'GPU memory preallocation must be in [0., 1.]. But got {mode}.'
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(mode)


def _set_gpu_device(device_ids):
    if isinstance(device_ids, int):
        device_ids = str(device_ids)
    elif isinstance(device_ids, (tuple, list)):
        device_ids = ','.join([str(d) for d in device_ids])
    elif isinstance(device_ids, str):
        device_ids = device_ids
    else:
        raise ValueError
    os.environ['CUDA_VISIBLE_DEVICES'] = device_ids


_parser = argparse.ArgumentParser()
_parser.add_argument('--devices', type=str, default='0', help='The GPU device ids.')
_args, _ = _parser.parse_known_args()
_set_gpu_device(_args.devices)
_set_gpu_preallocation(0.99)

import os.path
import pickle
import time
import matplotlib

if platform.system() == 'Linux':
    matplotlib.use('Agg')

import jax
from fast_histogram import histogram1d
import braintrace
import brainstate
import braintools
import numpy as np
import jax.numpy as jnp
from scipy import stats
import matplotlib.pyplot as plt

from rsnn_models_and_data import (
    LIF_STPExpCu_Dense_Layer,
    LIF_STDExpCu_Dense_Layer,
    LIF_ExpCu_Dense_Layer,
    LIF_Delta_Dense_Layer,
    ALIF_STPExpCu_Dense_Layer,
    ALIF_STDExpCu_Dense_Layer,
    ALIF_ExpCu_Dense_Layer,
    ALIF_Delta_Dense_Layer,
    get_snn_data,
    cosine_similarity,
)


def _compute_confidence_interval(data, confidence=0.95):
    n_seq, n_batch = data.shape

    # Calculate the mean along the sequence dimension
    means = np.mean(data, axis=1)

    # Calculate the standard error of the mean
    sem = stats.sem(data, axis=1)

    # Calculate the confidence interval
    ci = stats.t.interval(confidence, n_batch - 1, loc=means, scale=sem)

    return ci


def _compute_cosine_similarity(left, rights, num: int):
    cosines = dict()
    for key, right in rights.items():
        cos = jax.vmap(cosine_similarity)(
            jnp.einsum('rbi,rbj->bij', left.value, right),
            jax.vmap(jnp.outer)(jnp.sum(left.value, 0) / num, jnp.sum(right, 0))
        )
        cosines[key] = cos
    return cosines


class Model(brainstate.nn.Module):
    def __init__(self, module):
        super().__init__()

        self.module = module

    def update(self, i, inp):
        with brainstate.environ.context(i=i, t=brainstate.environ.get_dt() * i, fit=True):
            return self.module(inp)


def _compare(
    dataloader,
    num_in: int,
    num_rec: int,
    model_cls: type = LIF_STDExpCu_Dense_Layer,
    tau_mem: float = 10.,
    ff_wscale: float = 4.,
    rec_wscale: float = 4.,
    spk_fun=braintools.surrogate.ReluGrad(),
    # spk_fun=lambda x: jax.lax.stop_gradient(braintools.surrogate.relu_grad(x)),
    kwargs: dict = None,
    num_data=16,
):
    @brainstate.transform.jit
    def _run_the_model(inputs):
        model = Model(
            model_cls(
                n_in=num_in,
                n_rec=num_rec,
                ff_init=braintools.init.KaimingNormal(scale=ff_wscale),
                rec_init=braintools.init.KaimingNormal(scale=rec_wscale),
                spk_fun=spk_fun,
                tau_mem=tau_mem,
                **(kwargs or {})
            )
        )
        brainstate.nn.init_all_states(model, inputs.shape[1])
        etrace_model = braintrace.DiagTruncatedAlgorithm(model, n_truncation=inputs.shape[0])
        etrace_model.compile_graph(0, inputs[0])

        def _step_to_run(ri, inp):
            out = etrace_model(ri, inp)
            xs = tuple(etrace_model.etrace_xs.values())
            assert len(xs) == 1, f'len(xs) must be 1, but got {etrace_model.etrace_xs.keys()}'
            dfs = {
                etrace_model.graph.hidden_outvar_to_hidden[k[1]].name: v.value
                for k, v in etrace_model.etrace_dfs.items()
            }
            return _compute_cosine_similarity(xs[0], dfs, ri + 1)

        indices = np.arange(inputs.shape[0])
        cosines = brainstate.transform.for_loop(_step_to_run, indices, inputs)
        etrace_xs = [a.value for a in etrace_model.etrace_xs.values()]
        etrace_dfs = {
            etrace_model.graph.hidden_outvar_to_hidden[k[1]].name: v.value
            for k, v in etrace_model.etrace_dfs.items()
        }

        return cosines, etrace_xs, etrace_dfs

    final_results = []
    i_data = 0
    for xs, _ in dataloader:
        if i_data >= num_data:
            break
        i_data += 1
        xs = jnp.reshape(jnp.asarray(xs, dtype=brainstate.environ.dftype()), xs.shape[:2] + (-1,))
        res = _run_the_model(xs)
        final_results.append(res)
    final_results = braintools.tree.concat(final_results, axis=1)
    return final_results


def compare_jacobian_approx_on_real_dataset(fn='analysis/jac_cosine_sim_theorem3'):
    if not os.path.exists(fn):
        os.makedirs(fn)

    brainstate.environ.set(dt=1.0)
    n_rec = 100

    final_results = dict()

    for data_name, _, model, args in [
        ('N-MNIST', None, LIF_Delta_Dense_Layer, dict(rec_wscale=0.2, ff_wscale=1.5, tau_mem=10.)),
        ('N-MNIST', None, LIF_ExpCu_Dense_Layer,
         dict(rec_wscale=10., ff_wscale=100., tau_mem=10., kwargs=dict(tau_syn=10.0))),
        ('N-MNIST', None, LIF_STDExpCu_Dense_Layer,
         dict(rec_wscale=40., ff_wscale=100., tau_mem=10., kwargs=dict(tau_std=200.0, tau_syn=10.0))),
        ('N-MNIST', None, LIF_STPExpCu_Dense_Layer,
         dict(rec_wscale=5., ff_wscale=100., tau_mem=10., kwargs=dict(tau_syn=10, tau_f=10, tau_d=100, ))),
        ('N-MNIST', None, ALIF_Delta_Dense_Layer, dict(rec_wscale=1., ff_wscale=1., tau_mem=10.)),
        ('N-MNIST', None, ALIF_ExpCu_Dense_Layer,
         dict(rec_wscale=20., ff_wscale=100., tau_mem=10., kwargs=dict(tau_syn=10.0, tau_a=100.0))),
        ('N-MNIST', None, ALIF_STDExpCu_Dense_Layer,
         dict(rec_wscale=10., ff_wscale=100., tau_mem=10., kwargs=dict(tau_std=100., tau_syn=10))),
        ('N-MNIST', None, ALIF_STPExpCu_Dense_Layer,
         dict(rec_wscale=6, ff_wscale=100, tau_mem=5, kwargs=dict(tau_syn=5, tau_f=10, tau_d=100, tau_a=100))),

        ('gesture', None, LIF_Delta_Dense_Layer, dict(rec_wscale=0.1, ff_wscale=0.1, tau_mem=10.)),
        ('gesture', None, LIF_ExpCu_Dense_Layer,
         dict(rec_wscale=4., ff_wscale=20., tau_mem=10., kwargs=dict(tau_syn=10.0))),
        ('gesture', None, LIF_STDExpCu_Dense_Layer,
         dict(rec_wscale=8., ff_wscale=20., tau_mem=10., kwargs=dict(tau_std=200.0, tau_syn=10.0))),
        ('gesture', None, LIF_STPExpCu_Dense_Layer,
         dict(rec_wscale=5., ff_wscale=20., tau_mem=10., kwargs=dict(tau_syn=10, tau_f=10, tau_d=100))),
        ('gesture', None, ALIF_Delta_Dense_Layer, dict(rec_wscale=0.5, ff_wscale=0.5, tau_mem=10.)),
        ('gesture', None, ALIF_ExpCu_Dense_Layer,
         dict(rec_wscale=10., ff_wscale=20., tau_mem=10., kwargs=dict(tau_syn=10.0, tau_a=100.0))),
        ('gesture', None, ALIF_STDExpCu_Dense_Layer,
         dict(rec_wscale=10., ff_wscale=30., tau_mem=10., kwargs=dict(tau_std=200., tau_syn=10))),
        ('gesture', None, ALIF_STPExpCu_Dense_Layer,
         dict(rec_wscale=6, ff_wscale=40, tau_mem=5, kwargs=dict(tau_syn=5, tau_f=10, tau_d=100, tau_a=100))),

        ('SHD', None, LIF_Delta_Dense_Layer, dict(rec_wscale=0.1, ff_wscale=0.1, tau_mem=10.)),
        ('SHD', None, LIF_ExpCu_Dense_Layer,
         dict(rec_wscale=4., ff_wscale=20., tau_mem=10., kwargs=dict(tau_syn=10.0))),
        ('SHD', None, LIF_STDExpCu_Dense_Layer,
         dict(rec_wscale=8., ff_wscale=20., tau_mem=10., kwargs=dict(tau_std=200.0, tau_syn=10.0))),
        ('SHD', None, LIF_STPExpCu_Dense_Layer,
         dict(rec_wscale=5., ff_wscale=20., tau_mem=10., kwargs=dict(tau_syn=10, tau_f=10, tau_d=100))),
        ('SHD', None, ALIF_Delta_Dense_Layer, dict(rec_wscale=0.5, ff_wscale=0.5, tau_mem=10.)),
        ('SHD', None, ALIF_ExpCu_Dense_Layer,
         dict(rec_wscale=10., ff_wscale=20., tau_mem=10., kwargs=dict(tau_syn=10.0, tau_a=100.0))),
        ('SHD', None, ALIF_STDExpCu_Dense_Layer,
         dict(rec_wscale=10., ff_wscale=30., tau_mem=10., kwargs=dict(tau_std=200., tau_syn=10))),
        ('SHD', None, ALIF_STPExpCu_Dense_Layer,
         dict(rec_wscale=6, ff_wscale=40, tau_mem=5, kwargs=dict(tau_syn=5, tau_f=10, tau_d=100, tau_a=100))),
    ]:

        if data_name == 'gesture':
            datalengths = [100, 200, 400, 600, 1000]
            datalengths = [100, 200, 400, ]
        elif data_name == 'N-MNIST':
            datalengths = [200]
        else:
            datalengths = [None]

        for datalength in datalengths:
            data_args = brainstate.util.DotDict(batch_size=2, n_data_worker=10,
                                                drop_last=False, dataset=data_name,
                                                data_length=datalength, shuffle=True)
            dataset = get_snn_data(data_args)

            print(f'Processing {data_name} {datalength} with {model.__name__}')
            cosines, etrace_xs, etrace_dfs = _compare(
                dataset.train_loader,
                np.prod(dataset.in_shape),
                n_rec,
                model_cls=model,
                num_data=100,
                # spk_fun=lambda x: jnp.asarray(x > 0., dtype=x.dtype),
                spk_fun=braintools.surrogate.ReluGrad(),
                **args,
            )

            key = (data_name, datalength, model.__name__)
            final_results[key] = jax.tree.map(np.asarray, cosines)

            def visualize_hist(data):
                range = (np.min(data), np.max(data))
                if range[0] >= range[1]:
                    range = (range[0] - 1, range[1] + 1)
                hists = histogram1d(data.flatten(), bins=100, range=range)
                bins = np.linspace(range[0], range[1], 100)
                bins2 = np.zeros(101)
                bins2[:-1] = bins
                bins2[-1] = bins[-1] + (bins[1] - bins[0])
                plt.hist(bins, bins2, weights=hists, alpha=0.5)
                plt.yscale('log')

            fig, gs = braintools.visualize.get_figure(2, 1 + len(etrace_dfs), 4.5, 6.0)
            fig.add_subplot(gs[0, 0])
            visualize_hist(etrace_xs[0])
            for j, (key, data) in enumerate(etrace_dfs.items()):
                fig.add_subplot(gs[0, j + 1])
                visualize_hist(data)
                plt.title(f'{key}')
            final_cos = 0.
            for j, (key, data) in enumerate(cosines.items()):
                # print(data.shape)
                fig.add_subplot(gs[1, j + 1])
                cos = np.asarray(data).mean(1)
                final_cos = final_cos + cos
                plt.plot(cos)
                plt.title(f'Cosine Similarity {key}')
            fig.add_subplot(gs[1, 0])
            plt.plot(final_cos / len(cosines))
            plt.title('Mean of All Cosine Similarity')
            plt.suptitle(f'{data_name}-{datalength}-{model.__name__}')
            plt.savefig(f'{fn}/{data_name}-{datalength}-{model.__name__}.png')
            plt.close()

            for arr in jax.live_arrays():
                if arr.ndim != 0:
                    arr.delete()
            jax.clear_caches()

            # fig, gs = bts.visualize.get_figure(1, 1, 4.5, 6.0)
            # fig.add_subplot(gs[0, 0])
            # plt.plot(np.asarray(r))
            # plt.suptitle(f'{data_name}-{datalength}-{model.__name__}')
            # plt.savefig('analysis/jac_cosine_sim_theorem3/' + f'{data_name}-{datalength}-{model.__name__}.png')
            # plt.close()
            # plt.show()

    now = time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(int(round(time.time() * 1000)) / 1000))
    with open(os.path.join(fn, f'hidden-jacobian-cosine-n_rec={n_rec}-{now}.pkl'), 'wb') as fout:
        pickle.dump(final_results, fout)


def visualize_results():
    import seaborn
    seaborn.set_theme(font_scale=1.2, style=None)
    plt.style.use(['science', 'nature', 'notebook'])

    with open(
        'analysis/jac_cosine_sim_theorem3/t2/'
        'hidden-jacobian-cosine-n_rec=100-2024-08-02 12-31-35.pkl', 'rb'
    ) as fin:
        results = pickle.load(fin)

    model_mapping = {
        'LIF_Delta_Dense_Layer': 'LIF+Delta',
        'LIF_ExpCu_Dense_Layer': 'LIF+Expon',
        'LIF_STDExpCu_Dense_Layer': 'LIF+STD+Expon',
        'LIF_STPExpCu_Dense_Layer': 'LIF+STP+Expon',
        'ALIF_Delta_Dense_Layer': 'ALIF+Delta',
        'ALIF_ExpCu_Dense_Layer': 'ALIF+Expon',
        'ALIF_STDExpCu_Dense_Layer': 'ALIF+STD+Expon',
        'ALIF_STPExpCu_Dense_Layer': 'ALIF+STP+Expon',
    }

    for key, value in results.items():
        data_name, datalength, model_name = key
        if data_name != 'gesture' or datalength != 200:
            continue

        print(data_name, datalength, model_name)
        fig, gs = braintools.visualize.get_figure(1, 1, 3., 4.)
        ax = fig.add_subplot(gs[0, 0])
        for cosine_key, cosine in value.items():
            cosine_lower, cosine_upper = _compute_confidence_interval(cosine)
            cosine_mean = cosine.mean(1)
            if cosine_mean[datalength // 2:].sum() != 0.:
                # plt.plot(cosine_mean, label="$\mathbf{x} \otimes f_{%s}$" % (cosine_key.split('.')[1].lower()))
                plt.plot(cosine_mean, label="$%s$" % (cosine_key.split('.')[1].lower()))
                plt.fill_between(
                    np.arange(cosine_mean.shape[0]),
                    cosine_lower,
                    cosine_upper,
                    alpha=0.2,
                )
        plt.ylabel('Cosine Similarity')
        plt.xlabel('Time Step')
        if model_name in ['LIF_Delta_Dense_Layer', 'LIF_ExpCu_Dense_Layer']:
            plt.legend(loc='best')
        else:
            plt.legend(loc='lower right')
        plt.xlim(-2, datalength + 2)
        plt.ylim(0., 1.05)
        plt.suptitle(model_mapping[model_name], fontsize=16)
        # plt.suptitle(model_mapping[model_name])

        # ax.set_rasterized(True)
        plt.savefig(
            f'analysis/jac_cosine_sim_theorem3/t2/{data_name}-{datalength}-{model_name}.svg',
            dpi=300,
            transparent=True
        )
        # seaborn.despine()
        plt.show()
        plt.close()


if __name__ == '__main__':
    pass
    # compare_jacobian_approx_on_real_dataset('analysis/jac_cosine_sim_theorem3/t2')
    # compare_jacobian_approx_on_real_dataset('analysis/jac_cosine_sim_theorem3/no_surrogate_grad/')

    visualize_results()
