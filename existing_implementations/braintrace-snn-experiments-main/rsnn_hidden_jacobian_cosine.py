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


def define_device_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--devices', type=str, default='0', help='The GPU device ids.')
    args, _ = parser.parse_known_args()
    _set_gpu_device(args.devices)
    _set_gpu_preallocation(0.99)
    return args


define_device_args()

import os.path
import pickle
import time
from functools import partial
from typing import Callable

import braintrace
import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
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
    cosine_similarity,
    RandomDataset,
    get_snn_data,
)

model_setting_1 = [
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
    (
        'gesture', None, LIF_ExpCu_Dense_Layer,
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
    ('SHD', None, LIF_ExpCu_Dense_Layer, dict(rec_wscale=4., ff_wscale=20., tau_mem=10., kwargs=dict(tau_syn=10.0))),
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
]

model_setting_1_v2 = [
    ('N-MNIST', None, LIF_Delta_Dense_Layer, dict(rec_wscale=0.2, ff_wscale=1.5, tau_mem=10.)),
    ('N-MNIST', None, LIF_ExpCu_Dense_Layer,
     dict(rec_wscale=10., ff_wscale=100., tau_mem=10., kwargs=dict(tau_syn=10.0))),
    ('N-MNIST', None, LIF_STDExpCu_Dense_Layer,
     dict(rec_wscale=40., ff_wscale=100., tau_mem=10., kwargs=dict(tau_std=200.0, tau_syn=10.0))),
    ('N-MNIST', None, LIF_STPExpCu_Dense_Layer,
     dict(rec_wscale=5., ff_wscale=100., tau_mem=10., kwargs=dict(tau_syn=10, tau_f=10, tau_d=100, ))),
    ('N-MNIST', None, ALIF_Delta_Dense_Layer, dict(rec_wscale=1., ff_wscale=1., tau_mem=10., beta=1.)),
    ('N-MNIST', None, ALIF_ExpCu_Dense_Layer,
     dict(rec_wscale=20., ff_wscale=100., tau_mem=10., kwargs=dict(tau_syn=10.0, tau_a=100.0, beta=1.))),
    ('N-MNIST', None, ALIF_STDExpCu_Dense_Layer,
     dict(rec_wscale=10., ff_wscale=100., tau_mem=10., kwargs=dict(tau_std=100., tau_syn=10, beta=1.))),
    ('N-MNIST', None, ALIF_STPExpCu_Dense_Layer,
     dict(rec_wscale=6, ff_wscale=100, tau_mem=5, kwargs=dict(tau_syn=5, tau_f=10, tau_d=100, tau_a=100, beta=1.))),

    ('gesture', None, LIF_Delta_Dense_Layer, dict(rec_wscale=0.1, ff_wscale=0.1, tau_mem=10.)),
    (
        'gesture', None, LIF_ExpCu_Dense_Layer,
        dict(rec_wscale=4., ff_wscale=20., tau_mem=10., kwargs=dict(tau_syn=10.0))),
    ('gesture', None, LIF_STDExpCu_Dense_Layer,
     dict(rec_wscale=8., ff_wscale=20., tau_mem=10., kwargs=dict(tau_std=200.0, tau_syn=10.0))),
    ('gesture', None, LIF_STPExpCu_Dense_Layer,
     dict(rec_wscale=5., ff_wscale=20., tau_mem=10., kwargs=dict(tau_syn=10, tau_f=10, tau_d=100))),
    ('gesture', None, ALIF_Delta_Dense_Layer, dict(rec_wscale=0.5, ff_wscale=0.5, tau_mem=10., beta=1.)),
    ('gesture', None, ALIF_ExpCu_Dense_Layer,
     dict(rec_wscale=10., ff_wscale=20., tau_mem=10., kwargs=dict(tau_syn=10.0, tau_a=100.0, beta=1.))),
    ('gesture', None, ALIF_STDExpCu_Dense_Layer,
     dict(rec_wscale=10., ff_wscale=30., tau_mem=10., kwargs=dict(tau_std=200., tau_syn=10, beta=1.))),
    ('gesture', None, ALIF_STPExpCu_Dense_Layer,
     dict(rec_wscale=6, ff_wscale=40, tau_mem=5, kwargs=dict(tau_syn=5, tau_f=10, tau_d=100, tau_a=100, beta=1.))),

    ('SHD', None, LIF_Delta_Dense_Layer, dict(rec_wscale=0.1, ff_wscale=0.1, tau_mem=10.)),
    ('SHD', None, LIF_ExpCu_Dense_Layer, dict(rec_wscale=4., ff_wscale=20., tau_mem=10., kwargs=dict(tau_syn=10.0))),
    ('SHD', None, LIF_STDExpCu_Dense_Layer,
     dict(rec_wscale=8., ff_wscale=20., tau_mem=10., kwargs=dict(tau_std=200.0, tau_syn=10.0))),
    ('SHD', None, LIF_STPExpCu_Dense_Layer,
     dict(rec_wscale=5., ff_wscale=20., tau_mem=10., kwargs=dict(tau_syn=10, tau_f=10, tau_d=100))),
    ('SHD', None, ALIF_Delta_Dense_Layer, dict(rec_wscale=0.5, ff_wscale=0.5, tau_mem=10., beta=1.)),
    ('SHD', None, ALIF_ExpCu_Dense_Layer,
     dict(rec_wscale=10., ff_wscale=20., tau_mem=10., kwargs=dict(tau_syn=10.0, tau_a=100.0, beta=1.))),
    ('SHD', None, ALIF_STDExpCu_Dense_Layer,
     dict(rec_wscale=10., ff_wscale=30., tau_mem=10., kwargs=dict(tau_std=200., tau_syn=10, beta=1.))),
    ('SHD', None, ALIF_STPExpCu_Dense_Layer,
     dict(rec_wscale=6, ff_wscale=40, tau_mem=5, kwargs=dict(tau_syn=5, tau_f=10, tau_d=100, tau_a=100, beta=1.))),
]

model_setting_2 = [
    ('gesture', None, LIF_Delta_Dense_Layer, dict(rec_wscale=0.1, ff_wscale=0.1, tau_mem=10.)),
    (
        'gesture', None, LIF_ExpCu_Dense_Layer,
        dict(rec_wscale=4., ff_wscale=20., tau_mem=10., kwargs=dict(tau_syn=10.0))),
    ('gesture', None, ALIF_Delta_Dense_Layer, dict(rec_wscale=0.5, ff_wscale=0.5, tau_mem=10.)),
    ('gesture', None, ALIF_ExpCu_Dense_Layer,
     dict(rec_wscale=10., ff_wscale=20., tau_mem=10., kwargs=dict(tau_syn=10.0, tau_a=100.0))),

    ('N-MNIST', None, LIF_Delta_Dense_Layer, dict(rec_wscale=0.2, ff_wscale=1.5, tau_mem=10.)),
    ('N-MNIST', None, LIF_ExpCu_Dense_Layer,
     dict(rec_wscale=10., ff_wscale=100., tau_mem=10., kwargs=dict(tau_syn=10.0))),
    ('N-MNIST', None, ALIF_Delta_Dense_Layer, dict(rec_wscale=1., ff_wscale=1., tau_mem=10.)),
    ('N-MNIST', None, ALIF_ExpCu_Dense_Layer,
     dict(rec_wscale=20., ff_wscale=100., tau_mem=10., kwargs=dict(tau_syn=10.0, tau_a=100.0))),

    ('SHD', None, LIF_Delta_Dense_Layer, dict(rec_wscale=0.1, ff_wscale=0.1, tau_mem=10.)),
    ('SHD', None, LIF_ExpCu_Dense_Layer, dict(rec_wscale=4., ff_wscale=20., tau_mem=10., kwargs=dict(tau_syn=10.0))),
    ('SHD', None, ALIF_Delta_Dense_Layer, dict(rec_wscale=0.5, ff_wscale=0.5, tau_mem=10.)),
    ('SHD', None, ALIF_ExpCu_Dense_Layer,
     dict(rec_wscale=10., ff_wscale=20., tau_mem=10., kwargs=dict(tau_syn=10.0, tau_a=100.0))),
]


def approx_LR(W, iteration=100):
    m, n = W.shape
    # v = np.random.randn(n, 1)
    u = np.random.randn(m, 1)
    for i in range(iteration):
        v = (W.T @ u) / (u.T @ u)
        u = (W @ v) / (v.T @ v)
    return u @ v.T


def _check_jacobian(
    n: int,
    p_rec_spk: float = 0.1,
    p_inp_spk: float = 0.1,
    model_cls: type = LIF_STDExpCu_Dense_Layer,
    tau_mem: float = 10.,
    ff_scale: float = 4.,
    rec_scale: float = 4.,
    width: float = 0.3
):
    ff_init = braintools.init.KaimingNormal(scale=ff_scale)
    rec_init = braintools.init.KaimingNormal(scale=rec_scale)
    model = model_cls(n_in=n, n_rec=n,
                      ff_init=ff_init, rec_init=rec_init,
                      spk_fun=braintools.surrogate.ReluGrad(width=width),
                      tau_mem=tau_mem)
    brainstate.nn.init_all_states(model, 1)

    # initial membrane potential
    v_shape = model.neu.V.value.shape
    v0 = jnp.where(brainstate.random.random(v_shape) < p_rec_spk,
                   brainstate.random.uniform(1. - width, 1 + width, v_shape),
                   brainstate.random.truncated_normal(-5, 1 - width, v_shape))
    model.neu.V.value = v0

    # states and variables
    states = model.states()
    state_vals = states.to_dict_values()
    etrace_vars = states.subset(brainstate.HiddenState)
    get_state = lambda: jnp.concat([v.value.flatten() for v in etrace_vars.values()])

    def f(state, x):
        with brainstate.environ.context(i=0, t=0., fit=False):
            i = 0
            for k in etrace_vars:
                j = i + etrace_vars[k].value.size
                etrace_vars[k].value = state[i:j].reshape(etrace_vars[k].value.shape)
                i = j
            model(x)
            return get_state()

    # inputs and initial states
    inputs = jnp.asarray(brainstate.random.rand(1, n) < p_inp_spk, dtype=brainstate.environ.dftype())
    s = get_state()

    # jacobian
    Jacobian = np.asarray(jax.jacrev(f)(s, inputs))
    states.assign_dict_values(state_vals)

    # diagonal
    # diagonal1 = brainstate.transform.vector_grad(f, argnums=0)(s, inputs)
    # states.assign_dict_values(state_vals)
    with braintrace.stop_param_gradients():
        D = np.asarray(jax.jacrev(f, argnums=0)(s, inputs))
        states.assign_dict_values(state_vals)
        # diagonal = brainstate.transform.vector_grad(f, argnums=0)(s, inputs)
        # states.assign_dict_values(state_vals)
        # D = np.diag(diagonal)
    # diff = diagonal1 - diagonal

    # approximations
    # approximated_jac = low_rank_diag_approx(Jacobian, D)
    # approximated_jac = approx_DLR(Jacobian)
    DD = np.diag(Jacobian)
    jac = Jacobian - np.diag(DD)
    approximated_jac = approx_LR(jac, iteration=1) + D

    print((np.sum(np.abs(jac).sum(0) < DD)) / s.size)

    # similarity
    sim = np.asarray(cosine_similarity(Jacobian, approximated_jac))
    sim_d = np.asarray(cosine_similarity(Jacobian, D))
    print(f'{model_cls.__name__}, '
          f'p_rec_spk={p_rec_spk}, p_inp_spk={p_inp_spk}, '
          f'rec_scale={rec_scale}, ff_scale={ff_scale}, width={width}, '
          f'tau_mem={tau_mem}, n={s.size}, '
          f'sim<diag>={sim_d}, sim<diag+LORA>={sim}')
    return sim_d


def compare_jacobian_approximation_on_artificial_data():
    n = 1000
    setting = dict(n=n, width=0.5, rec_scale=20., tau_mem=10., ff_scale=4.)
    setting = dict(n=n, width=0.5, rec_scale=10., tau_mem=10., ff_scale=4.)
    brainstate.environ.set(
        mode=brainstate.mixin.JointMode(brainstate.mixin.Batching(), brainstate.mixin.Training()),
        dt=0.1
    )
    for model_cls in [
        LIF_STPExpCu_Dense_Layer,
        LIF_STDExpCu_Dense_Layer,
        LIF_ExpCu_Dense_Layer,
        LIF_Delta_Dense_Layer,
        ALIF_STPExpCu_Dense_Layer,
        ALIF_STDExpCu_Dense_Layer,
        ALIF_ExpCu_Dense_Layer,
        ALIF_Delta_Dense_Layer,
        LIF_STDExpCu_Dense_Layer,
        ALIF_STDExpCu_Dense_Layer,
    ]:
        rr = []
        # for p_rec_spk in [0.1]:
        for p_rec_spk in [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,  # 0.7, 0.8, 0.9, 1.0
                          ]:
            # for p_rec_spk in [0.2, 0.3, 0.4, 0.5, 0.6]:
            # for p_inp_spk in [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
            for p_inp_spk in [0.1]:
                # for p_inp_spk in [0.6]:
                r = _check_jacobian(p_rec_spk=p_rec_spk, p_inp_spk=p_inp_spk, model_cls=model_cls, **setting)
                rr.append(r)
        print(f'{model_cls.__name__}, {np.asarray(rr).tolist()}')
        print()


def _compare_jac_one_step(model: brainstate.nn.Module, idx, inp):
    states = model.states()
    etrace_states = states.subset(brainstate.HiddenState)
    state_vals = states.to_dict_values()
    hidden = u.math.concatenate([v.value.flatten() for v in etrace_states.values()])

    def step_run(hid, i, x):
        with brainstate.environ.context(i=i, t=brainstate.environ.get_dt() * i, fit=True):
            i = 0
            for k in etrace_states:
                j = i + etrace_states[k].value.size
                etrace_states[k].value = hid[i:j].reshape(etrace_states[k].value.shape)
                i = j
            model(x)
            return jnp.concatenate([v.value.flatten() for v in etrace_states.values()])

    def jacobian():
        jac = jax.jacrev(step_run, argnums=0)(hidden, idx, inp)
        states.assign_dict_values(state_vals)
        return jac

    # compute jacobian and diagonal
    jac = jacobian()
    with braintrace.stop_param_gradients():
        diag = jacobian()
    cos = cosine_similarity(jac, diag)

    # update the model
    with brainstate.environ.context(i=idx, t=brainstate.environ.get_dt() * idx, fit=True):
        spk = model(inp)

    return cos.mean(), spk


def _compare_jac_all_steps(model: brainstate.nn.Module, inputs):
    def step_run(i, x):
        with brainstate.environ.context(i=i, t=brainstate.environ.get_dt() * i, fit=True):
            return model(x)

    def all_run(hid, xs):
        etrace_states = model.states().subset(brainstate.HiddenState)

        i = 0
        for k in etrace_states:
            j = i + etrace_states[k].value.size
            etrace_states[k].value = hid[i:j].reshape(etrace_states[k].value.shape)
            i = j

        indices = np.arange(xs.shape[0])
        spks = brainstate.transform.for_loop(step_run, indices, xs)
        new_hidden = jnp.concat([v.value.flatten() for v in etrace_states.values()])
        return new_hidden, spks

    def jacobian():
        etrace_states = model.states().subset(brainstate.HiddenState)
        hidden = jnp.concat([v.value.flatten() for v in etrace_states.values()])
        jac, spks = jax.jacrev(all_run, argnums=0, has_aux=True)(hidden, inputs)
        return jac, spks

    # compute jacobian and diagonal
    brainstate.nn.init_all_states(model)
    jac, spks = jacobian()
    with braintrace.stop_param_gradients():
        brainstate.nn.init_all_states(model)
        diag, _ = jacobian()
    cos = cosine_similarity(jac, diag)
    return cos, spks


def raster_plot(
    ts,
    sp_matrix,
    ax=None,
    marker='.',
    markersize=2,
    color='k',
    xlabel='Time (ms)',
    ylabel='Neuron index',
    xlim=None,
    ylim=None,
    title=None,
    show=False,
    **kwargs
):
    """Show the rater plot of the spikes.
  
    Parameters
    ----------
    ts : np.ndarray
        The run times.
    sp_matrix : np.ndarray
        The spike matrix which records the spike information.
        It can be easily accessed by specifying the ``monitors``
        of NeuGroup by: ``neu = NeuGroup(..., monitors=['spike'])``
    ax : Axes
        The figure.
    markersize : int
        The size of the marker.
    color : str
        The color of the marker.
    xlim : list, tuple
        The xlim.
    ylim : list, tuple
        The ylim.
    xlabel : str
        The xlabel.
    ylabel : str
        The ylabel.
    show : bool
        Show the figure.
    """

    sp_matrix = np.asarray(sp_matrix)
    ts = np.asarray(ts)

    # get index and time
    elements = np.where(sp_matrix > 0.)
    index = elements[1]
    time = ts[elements[0]]

    # plot rater
    if ax is None:
        ax = plt
    ax.plot(time, index, marker + color, markersize=markersize, **kwargs)

    # xlable
    if xlabel:
        plt.xlabel(xlabel)

    # ylabel
    if ylabel:
        plt.ylabel(ylabel)

    if xlim:
        plt.xlim(xlim[0], xlim[1])

    if ylim:
        plt.ylim(ylim[0], ylim[1])

    if title:
        plt.title(title)

    if show:
        plt.show()


def _compare(
    dataloader,
    num_in: int,
    num_rec: int,
    model_cls: type = LIF_STDExpCu_Dense_Layer,
    tau_mem: float = 10.,
    ff_wscale: float = 4.,
    rec_wscale: float = 4.,
    spk_fun: Callable = braintools.surrogate.ReluGrad(),
    kwargs: dict = None,
    num_data: int = 50,
    show: bool = False
):
    model = model_cls(
        n_in=num_in,
        n_rec=num_rec,
        ff_init=braintools.init.KaimingNormal(scale=ff_wscale),
        rec_init=braintools.init.KaimingNormal(scale=rec_wscale),
        spk_fun=spk_fun,
        tau_mem=tau_mem,
        **(kwargs or {})
    )

    cosine_singles, cosine_all, firing_rates = [], [], []
    i = 0
    for xs, _ in dataloader:
        i += 1
        if i > num_data:
            break

        xs = u.math.flatten(jnp.asarray(xs, dtype=brainstate.environ.dftype()), 1)

        brainstate.nn.init_all_states(model)
        indices = np.arange(xs.shape[0])
        cosine, spikes1 = brainstate.transform.for_loop(partial(_compare_jac_one_step, model), indices, xs)
        fr = jnp.sum(spikes1) / (xs.shape[0] * brainstate.environ.get_dt()) * 1000 / num_rec

        cosine2, spikes2 = _compare_jac_all_steps(model, xs)
        print(f'Cosine Similarity = {cosine.mean()},  Cosine all = {cosine2}, firing rate = {fr} Hz')
        if show:
            raster_plot(indices, spikes1, show=True)

        cosine_singles.extend(list(cosine))
        cosine_all.append(cosine2)
        firing_rates.append(fr)

    return cosine_singles, cosine_all, firing_rates


def compare_jacobian_approx_on_random_data():
    brainstate.environ.set(dt=1.0)
    n_in = 100
    n_out = 2
    num_rec = 100
    n_seq = 1000

    data = RandomDataset(n_seq, n_in, n_out, prob=0.01)
    # args = brainstate.util.DotDict(rec_wscale=0.1, ff_wscale=0.1, tau_mem=10.)
    # model_cls = LIF_Delta_Dense_Layer
    args = brainstate.util.DotDict(rec_wscale=40., ff_wscale=100., tau_mem=10., kwargs=dict(tau_syn=10.0, tau_a=100.0))
    model_cls = ALIF_ExpCu_Dense_Layer

    for n_rec in [10, 20, 40, 60, 80, 100, 200, 300, 500]:
        print('n_rec = ', n_rec)
        r = _compare(data, n_in, num_rec, model_cls=model_cls, **args, show=False, num_data=2)
        print()


def compare_jacobian_approx_on_real_dataset():
    brainstate.environ.set(dt=1.0)
    n_rec = 100
    spk_fun = braintools.surrogate.ReluGrad(width=0.3)
    spk_fun = braintools.surrogate.ReluGrad(width=1.)
    # spk_fun = braintools.surrogate.S2NN(alpha=8.0, beta=2.0)
    # spk_fun = braintools.surrogate.LeakyRelu(alpha=0.01)
    # spk_fun = braintools.surrogate.MultiGaussianGrad()

    for n_rec in [10, 50, 100, 200, 300, 400, 500]:
        # data
        args = brainstate.util.DotDict(batch_size=1, n_data_worker=1, drop_last=False, dataset='N-MNIST', shuffle=False)
        # args = brainstate.util.DotDict(batch_size=1, n_data_worker=1, drop_last=False, dataset='gesture', data_length=200)
        # args = brainstate.util.DotDict(batch_size=1, n_data_worker=1, drop_last=False, dataset='gesture', data_length=500)
        # args = brainstate.util.DotDict(batch_size=1, n_data_worker=1, drop_last=False, dataset='SHD')
        dataset = get_snn_data(args)

        # model
        args = brainstate.util.DotDict(rec_wscale=0.1, ff_wscale=0.1, tau_mem=10.)
        args = brainstate.util.DotDict(rec_wscale=1, ff_wscale=10, tau_mem=10.)
        # args = brainstate.util.DotDict(rec_wscale=2, ff_wscale=2, tau_mem=10.)
        # args = brainstate.util.DotDict(rec_wscale=1, ff_wscale=2, tau_mem=10.)
        # args = brainstate.util.DotDict(rec_wscale=0.1, ff_wscale=2, tau_mem=10.)
        model_cls = LIF_Delta_Dense_Layer
        # args = brainstate.util.DotDict(rec_wscale=10., ff_wscale=100., tau_mem=10., kwargs=dict(tau_syn=10.0, ))
        # args = brainstate.util.DotDict(rec_wscale=20., ff_wscale=40., tau_mem=10., kwargs=dict(tau_syn=10.0, ))
        # args = brainstate.util.DotDict(rec_wscale=40., ff_wscale=100., tau_mem=10., kwargs=dict(tau_syn=10.0, ))
        # args = brainstate.util.DotDict(rec_wscale=100., ff_wscale=100., tau_mem=10., kwargs=dict(tau_syn=10.0, ))
        # model_cls = LIF_ExpCu_Dense_Layer
        # args = brainstate.util.DotDict(rec_wscale=40., ff_wscale=100., tau_mem=10., kwargs=dict(tau_std=200.0, tau_syn=10.0))
        # model_cls = LIF_STDExpCu_Dense_Layer
        # args = brainstate.util.DotDict(rec_wscale=50., ff_wscale=80., tau_mem=10., kwargs=dict(tau_syn=10, tau_f=10, tau_d=100,))
        # args = brainstate.util.DotDict(rec_wscale=5., ff_wscale=20., tau_mem=10., kwargs=dict(tau_syn=10, tau_f=10, tau_d=100))
        # model_cls = LIF_STPExpCu_Dense_Layer

        # args = brainstate.util.DotDict(rec_wscale=0.1, ff_wscale=0.1, tau_mem=10., kwargs=dict())
        # model_cls = ALIF_Delta_Dense_Layer
        # args = brainstate.util.DotDict(rec_wscale=40., ff_wscale=100., tau_mem=10., kwargs=dict(tau_syn=10.0, tau_a=100.0))
        # model_cls = ALIF_ExpCu_Dense_Layer
        # args = brainstate.util.DotDict(rec_wscale=10., ff_wscale=100., tau_mem=10., kwargs=dict(tau_std=100., tau_syn=10))
        # model_cls = ALIF_STDExpCu_Dense_Layer
        # args = brainstate.util.DotDict(rec_wscale=6, ff_wscale=40, tau_mem=5, kwargs=dict(tau_syn=5, tau_f=10, tau_d=100, tau_a=100))
        # model_cls = ALIF_STPExpCu_Dense_Layer

        args.update(spk_fun=spk_fun, num_rec=n_rec, model_cls=model_cls)

        # computing
        print(n_rec, args)
        r = _compare(dataset.train_loader, int(np.prod(dataset.in_shape)), **args, num_data=10)


def compare_jacobian_approx_on_real_dataset_v2(fn='analysis/jac_cosine_sim'):
    if not os.path.exists(fn):
        os.makedirs(fn)

    brainstate.environ.set(dt=1.0)
    n_rec = 100

    final_results = dict()

    for data_name, _, model, args in [
        ('N-MNIST', None, LIF_Delta_Dense_Layer, dict(rec_wscale=2, ff_wscale=15, tau_mem=10.)),
        ('N-MNIST', None, LIF_ExpCu_Dense_Layer,
         dict(rec_wscale=10., ff_wscale=30., tau_mem=10., kwargs=dict(tau_syn=10.0))),
        ('N-MNIST', None, LIF_STDExpCu_Dense_Layer,
         dict(rec_wscale=10., ff_wscale=20., tau_mem=10., kwargs=dict(tau_std=200.0, tau_syn=10.0))),
        ('N-MNIST', None, LIF_STPExpCu_Dense_Layer,
         dict(rec_wscale=5., ff_wscale=100., tau_mem=10., kwargs=dict(tau_syn=10, tau_f=10, tau_d=100, ))),
        ('N-MNIST', None, ALIF_Delta_Dense_Layer, dict(rec_wscale=1., ff_wscale=1., tau_mem=10.)),
        ('N-MNIST', None, ALIF_ExpCu_Dense_Layer,
         dict(rec_wscale=10., ff_wscale=20., tau_mem=10., kwargs=dict(tau_syn=10.0, tau_a=100.0))),
        ('N-MNIST', None, ALIF_STDExpCu_Dense_Layer,
         dict(rec_wscale=5., ff_wscale=20., tau_mem=10., kwargs=dict(tau_std=100., tau_syn=10))),
        ('N-MNIST', None, ALIF_STPExpCu_Dense_Layer,
         dict(rec_wscale=3, ff_wscale=20, tau_mem=5, kwargs=dict(tau_syn=5, tau_f=10, tau_d=100, tau_a=100))),

        ('gesture', None, LIF_Delta_Dense_Layer, dict(rec_wscale=0.1, ff_wscale=0.1, tau_mem=10.)),
        ('gesture', None, LIF_ExpCu_Dense_Layer,
         dict(rec_wscale=0.2, ff_wscale=1., tau_mem=10., kwargs=dict(tau_syn=10.0))),
        ('gesture', None, LIF_STDExpCu_Dense_Layer,
         dict(rec_wscale=0.2, ff_wscale=1., tau_mem=10., kwargs=dict(tau_std=200.0, tau_syn=10.0))),
        ('gesture', None, LIF_STPExpCu_Dense_Layer,
         dict(rec_wscale=1., ff_wscale=4., tau_mem=10., kwargs=dict(tau_syn=10, tau_f=10, tau_d=100))),
        ('gesture', None, ALIF_Delta_Dense_Layer, dict(rec_wscale=0.5, ff_wscale=0.5, tau_mem=10.)),
        ('gesture', None, ALIF_ExpCu_Dense_Layer,
         dict(rec_wscale=1., ff_wscale=5., tau_mem=10., kwargs=dict(tau_syn=10.0, tau_a=100.0))),
        ('gesture', None, ALIF_STDExpCu_Dense_Layer,
         dict(rec_wscale=1., ff_wscale=4., tau_mem=10., kwargs=dict(tau_std=200., tau_syn=10))),
        ('gesture', None, ALIF_STPExpCu_Dense_Layer,
         dict(rec_wscale=1, ff_wscale=4, tau_mem=5, kwargs=dict(tau_syn=5, tau_f=10, tau_d=100, tau_a=100))),

        ('SHD', None, LIF_Delta_Dense_Layer, dict(rec_wscale=10, ff_wscale=100, tau_mem=10.)),
        ('SHD', None, LIF_ExpCu_Dense_Layer,
         dict(rec_wscale=0.1, ff_wscale=100., tau_mem=10., kwargs=dict(tau_syn=10.0))),
        ('SHD', None, LIF_STDExpCu_Dense_Layer,
         dict(rec_wscale=0.2, ff_wscale=1., tau_mem=10., kwargs=dict(tau_std=200.0, tau_syn=10.0))),
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
            datalengths = [200, 400, 600, 1000]
        else:
            datalengths = [None]

        for datalength in datalengths:
            data_args = brainstate.util.DotDict(batch_size=1, n_data_worker=1,
                                                drop_last=False, dataset=data_name,
                                                data_length=datalength, shuffle=False)
            dataset = get_snn_data(data_args)

            print(f'Processing {data_name} {datalength} with {model.__name__}')
            r = _compare(
                dataset.train_loader,
                int(np.prod(dataset.in_shape)),
                n_rec,
                model_cls=model,
                **args,
                num_data=100,
            )

            key = (data_name, datalength, model.__name__)
            final_results[key] = {
                'cos_single_steps': np.asarray(r[0]),
                'cos_all_steps': np.asarray(r[1]),
                'firing_rates': np.asarray(r[2])
            }

    now = time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(int(round(time.time() * 1000)) / 1000))
    with open(os.path.join(fn, f'hidden-jacobian-cosine-n_rec={n_rec}-{now}.pkl'), 'wb') as fout:
        pickle.dump(final_results, fout)


def compare_jacobian_approx_when_recurrent_size_increases(fn='analysis/jac_cosine_sim'):
    brainstate.environ.set(dt=1.0)
    spk_fun = braintools.surrogate.ReluGrad(width=1.)

    rec_sizes = [10, 50, 100, 300, 500, 1000, 2000, 4000, 8000, 10000, 20000, 40000]
    rec_sizes = [10, 50, 100, 300, 500, 1000, 2000, 4000]

    final_results = dict()

    for data_name, data_len, model_cls, args in model_setting_2:
        for n_rec in rec_sizes:
            datalengths = [200, 400, 600, 1000] if data_name == 'gesture' else [None]
            # datalengths = [200,] if data_name == 'gesture' else [None]
            for datalength in datalengths:
                # data
                data_args = brainstate.util.DotDict(batch_size=1, n_data_worker=1,
                                                    drop_last=False, dataset=data_name,
                                                    shuffle=False, data_length=datalength)
                dataset = get_snn_data(data_args)

                args.update(spk_fun=spk_fun, num_rec=n_rec, model_cls=model_cls)

                # computing
                print(data_args)
                print(args)
                try:
                    r = _compare(dataset.train_loader, int(np.prod(dataset.in_shape)), **args, num_data=200)
                    final_results[(data_name, datalength, n_rec, model_cls.__name__)] = {
                        'cos_single_steps': np.asarray(r[0]),
                        'cos_all_steps': np.asarray(r[1]),
                        'firing_rates': np.asarray(r[2]),
                        'net_sizes': np.ones(len(r[0]), dtype=int) * n_rec,
                    }
                    brainstate.util.clear_buffer_memory()
                except Exception as e:
                    print(e)
                print()

    now = time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(int(round(time.time() * 1000)) / 1000))
    with open(os.path.join(fn, f'hidden-jacobian-cosine-rec-size-increase-{now}.pkl'), 'wb') as fout:
        pickle.dump(final_results, fout)


def compare_jacobian_approx_when_ff_conn_increases(fn='analysis/jac_cosine_sim'):
    brainstate.environ.set(dt=1.0)
    spk_fun = braintools.surrogate.ReluGrad(width=1.)
    n_rec = 200

    final_results = dict()
    for data_name, data_len, model_cls, model_args in model_setting_2:
        for ff_frac in [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0]:
            datalengths = [200, 400, 600, 1000] if data_name == 'gesture' else [None]
            # datalengths = [200, ] if data_name == 'gesture' else [None]
            for data_length in datalengths:
                args = dict(model_args)
                args['ff_wscale'] = args['ff_wscale'] * ff_frac

                # data
                data_args = brainstate.util.DotDict(batch_size=1, n_data_worker=1, drop_last=False,
                                                    dataset=data_name, shuffle=False, data_length=data_length)
                dataset = get_snn_data(data_args)

                args.update(spk_fun=spk_fun, num_rec=n_rec, model_cls=model_cls)

                # computing
                print(data_args)
                print(args)
                try:
                    r = _compare(dataset.train_loader, int(np.prod(dataset.in_shape)), **args, num_data=200)
                    final_results[(data_name, data_length, ff_frac, model_cls.__name__)] = {
                        'cos_single_steps': np.asarray(r[0]),
                        'cos_all_steps': np.asarray(r[1]),
                        'firing_rates': np.asarray(r[2]),
                        'ff_frac': ff_frac,
                    }
                    brainstate.util.clear_buffer_memory()
                except Exception as e:
                    print(e)
                print()

    now = time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(int(round(time.time() * 1000)) / 1000))
    with open(os.path.join(fn, f'hidden-jacobian-cosine-ff-conn-increase-{now}.pkl'), 'wb') as fout:
        pickle.dump(final_results, fout)


def compare_jacobian_approx_when_rec_conn_increases(fn='analysis/jac_cosine_sim'):
    brainstate.environ.set(dt=1.0)
    spk_fun = braintools.surrogate.ReluGrad(width=1.)
    n_rec = 200

    final_results = dict()
    for data_name, data_len, model_cls, model_args in model_setting_2:
        for rec_frac in [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0]:
            datalengths = [200, 400, 600, 1000] if data_name == 'gesture' else [None]
            # datalengths = [200, ] if data_name == 'gesture' else [None]
            for data_length in datalengths:
                # data
                data_args = brainstate.util.DotDict(batch_size=1, n_data_worker=1, drop_last=False,
                                                    dataset=data_name, shuffle=False, data_length=data_length)
                dataset = get_snn_data(data_args)

                args = dict(model_args)
                args['rec_wscale'] = args['rec_wscale'] * rec_frac
                args.update(spk_fun=spk_fun, num_rec=n_rec, model_cls=model_cls)

                # computing
                print(data_args)
                print(args)
                try:
                    r = _compare(dataset.train_loader, int(np.prod(dataset.in_shape)), **args, num_data=200)
                    final_results[(data_name, data_length, rec_frac, model_cls.__name__)] = {
                        'cos_single_steps': np.asarray(r[0]),
                        'cos_all_steps': np.asarray(r[1]),
                        'firing_rates': np.asarray(r[2]),
                        'rec_frac': rec_frac,
                    }
                    brainstate.util.clear_buffer_memory()
                except Exception as e:
                    print(e)
                print()

    now = time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(int(round(time.time() * 1000)) / 1000))
    with open(os.path.join(fn, f'hidden-jacobian-cosine-rec-conn-increase-{now}.pkl'), 'wb') as fout:
        pickle.dump(final_results, fout)


if __name__ == '__main__':
    pass
    # compare_jacobian_approx_on_random_data()
    # compare_jacobian_approximation_on_artificial_data()
    # compare_jacobian_approx_on_real_dataset()

    compare_jacobian_approx_on_real_dataset_v2()

    # compare_jacobian_approx_when_recurrent_size_increases()

    # compare_jacobian_approx_when_ff_conn_increases()
    # compare_jacobian_approx_when_rec_conn_increases()
