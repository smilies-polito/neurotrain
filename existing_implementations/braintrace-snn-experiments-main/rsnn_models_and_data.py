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
from typing import Callable, Dict, Sequence

import brainpy
import braintrace
import brainstate
import braintools
import brainunit as u
import h5py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import tonic
import torch
from tonic.collation import PadTensors
from tonic.datasets import SHD, NMNIST, DVSGesture
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.data import Dataset


def cosine_similarity(x, y):
    x = x.flatten()
    y = y.flatten()
    deno = jnp.linalg.norm(x) * jnp.linalg.norm(y)
    return jnp.nan_to_num(jnp.inner(x, y) / deno, nan=0.)


class _ExpCo_Dense_Layer(brainstate.nn.Module):
    """
    The RTRL layer with dense connected exponential conductance-based synapses.
    """

    def __init__(
        self,
        neu, n_in, n_rec,
        input_ei_sep=False,
        tau_syn=10.,
        rec_init: Callable = braintools.init.KaimingNormal(),
        ff_init: Callable = braintools.init.KaimingNormal()
    ):
        super().__init__()

        self.input_ei_sep = input_ei_sep
        self.n_exc_rec = int(n_rec * 0.8)
        self.n_inh_rec = n_rec - self.n_exc_rec

        self.neu = neu

        if input_ei_sep:
            self.n_exc_in = int(n_in * 0.8)
            self.n_inh_in = n_in - self.n_exc_in

            weight = jnp.concat([ff_init([self.n_exc_in, n_rec]), rec_init([self.n_exc_rec, n_rec])], axis=0)
            self.exe_syn = brainpy.state.AlignPostProj(
                comm=braintrace.nn.SignedWLinear(self.n_exc_in + self.n_exc_rec, n_rec, w_init=weight),
                syn=brainpy.state.Expon.desc(n_rec, tau=tau_syn, g_initializer=braintools.init.ZeroInit()),
                out=brainpy.state.COBA.desc(E=3.5),
                post=self.neu
            )

            weight = jnp.concat([4 * ff_init([self.n_inh_in, n_rec]), 4 * rec_init([self.n_inh_rec, n_rec])], axis=0)
            self.inh_syn = brainpy.state.AlignPostProj(
                comm=braintrace.nn.SignedWLinear(self.n_inh_in + self.n_inh_rec, n_rec, w_init=weight),
                syn=brainpy.state.Expon.desc(n_rec, tau=tau_syn, g_initializer=braintools.init.ZeroInit()),
                out=brainpy.state.COBA.desc(E=-0.5),
                post=self.neu
            )
        else:

            self.inp_syn = brainpy.state.AlignPostProj(
                comm=braintrace.nn.Linear(n_in, n_rec, w_init=ff_init([n_in, n_rec])),
                syn=brainpy.state.Expon.desc(n_rec, tau=tau_syn, g_initializer=braintools.init.ZeroInit()),
                out=brainpy.state.CUBA.desc(1.),
                post=self.neu
            )

            self.exe_syn = brainpy.state.AlignPostProj(
                comm=braintrace.nn.SignedWLinear(self.n_exc_rec, n_rec, w_init=rec_init([self.n_exc_rec, n_rec])),
                syn=brainpy.state.Expon.desc(n_rec, tau=tau_syn, g_initializer=braintools.init.ZeroInit()),
                out=brainpy.state.COBA.desc(E=1.5),
                post=self.neu
            )

            self.inh_syn = brainpy.state.AlignPostProj(
                comm=braintrace.nn.SignedWLinear(self.n_inh_rec, n_rec, w_init=4 * rec_init([self.n_inh_rec, n_rec])),
                syn=brainpy.state.Expon.desc(n_rec, tau=tau_syn, g_initializer=braintools.init.ZeroInit()),
                out=brainpy.state.COBA.desc(E=-0.5),
                post=self.neu
            )

    def update(self, spk):
        rec_exe_spk, rec_inh_spk = jnp.split(self.neu.get_spike(), [self.n_exc_rec], axis=-1)
        if self.input_ei_sep:
            in_exe_spk, in_inh_spk = jnp.split(spk, [self.n_exc_in], axis=-1)
            self.exe_syn(jnp.concat([in_exe_spk, rec_exe_spk], axis=-1))
            self.inh_syn(jnp.concat([in_inh_spk, rec_inh_spk], axis=-1))
            self.neu()
        else:
            self.inp_syn(spk)
            self.exe_syn(rec_exe_spk)
            self.inh_syn(rec_inh_spk)
            self.neu()
        # only output excitatory spikes
        # return self.neu.spike[..., :self.n_exc]
        return self.neu.get_spike()


class LIF_ExpCo_Dense_Layer(_ExpCo_Dense_Layer):
    """
    The RTRL layer with LIF neurons and dense connected exponential conductance-based synapses.
    """

    def __init__(
        self, n_in, n_rec, input_ei_sep=False, tau_mem=5., tau_syn=10., V_th=1.,
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'soft',
        rec_init: Callable = braintools.init.KaimingNormal(),
        ff_init: Callable = braintools.init.KaimingNormal()
    ):
        neu = brainpy.state.LIF(
            n_rec,
            tau=tau_mem,
            spk_fun=spk_fun,
            spk_reset=spk_reset,
            V_th=V_th,
            V_rest=0.,
            V_reset=0.,
            R=1.,
            V_initializer=braintools.init.ZeroInit()
        )
        super().__init__(
            n_in=n_in,
            n_rec=n_rec,
            input_ei_sep=input_ei_sep,
            tau_syn=tau_syn,
            rec_init=rec_init,
            ff_init=ff_init,
            neu=neu,
        )


class ALIF_ExpCo_Dense_Layer(_ExpCo_Dense_Layer):
    """
    The RTRL layer with ALIF neurons and dense connected exponential conductance-based synapses.
    """

    def __init__(
        self, n_in, n_rec, input_ei_sep=False,
        tau_a=100., beta=0.1, tau_mem=5., tau_syn=10., V_th=1.,
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'soft',
        rec_init: Callable = braintools.init.KaimingNormal(),
        ff_init: Callable = braintools.init.KaimingNormal()
    ):
        neu = brainpy.state.ALIF(
            n_rec,
            tau=tau_mem,
            tau_a=tau_a,
            beta=beta,
            spk_fun=spk_fun,
            spk_reset=spk_reset,
            V_th=V_th,
            R=1.,
            V_reset=0.,
            V_rest=0.,
            V_initializer=braintools.init.ZeroInit(),
            a_initializer=braintools.init.ZeroInit(),
        )
        super().__init__(
            neu=neu,
            n_in=n_in,
            n_rec=n_rec,
            input_ei_sep=input_ei_sep,
            tau_syn=tau_syn,
            rec_init=rec_init,
            ff_init=ff_init
        )


class LIF_ExpCu_Dense_Layer(brainstate.nn.Module):
    """
    The RTRL layer with LIF neurons and dense connected exponential current synapses.
    """

    def __init__(
        self, n_in, n_rec, tau_mem=5., tau_syn=10., V_th=1.,
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'soft',
        rec_init: Callable = braintools.init.KaimingNormal(),
        ff_init: Callable = braintools.init.KaimingNormal()
    ):
        super().__init__()
        self.neu = brainpy.state.LIF(
            n_rec,
            tau=tau_mem,
            spk_fun=spk_fun,
            spk_reset=spk_reset,
            V_th=V_th,
            V_rest=0.,
            V_reset=0.,
            R=1.,
            V_initializer=braintools.init.ZeroInit()
        )
        self.syn = brainpy.state.AlignPostProj(
            comm=braintrace.nn.Linear(
                n_in + n_rec, n_rec,
                jnp.concat([ff_init([n_in, n_rec]), rec_init([n_rec, n_rec])], axis=0)
            ),
            syn=brainpy.state.Expon.desc(n_rec, tau=tau_syn, g_initializer=braintools.init.ZeroInit()),
            out=brainpy.state.CUBA.desc(1.),
            post=self.neu
        )

    def update(self, spk):
        self.syn(jnp.concat([spk, self.neu.get_spike()], axis=-1))
        self.neu(0.)
        return self.neu.get_spike()


class LIF_STDExpCu_Dense_Layer(brainstate.nn.Module):
    """
    The RTRL layer with LIF neurons and dense connected STD-based exponential current synapses.
    """

    def __init__(
        self, n_in, n_rec, inp_std=False, tau_mem=5., tau_syn=10., V_th=1., tau_std=500.,
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'soft',
        rec_init: Callable = braintools.init.KaimingNormal(),
        ff_init: Callable = braintools.init.KaimingNormal()
    ):
        super().__init__()
        self.neu = brainpy.state.LIF(
            n_rec,
            tau=tau_mem,
            spk_fun=spk_fun,
            spk_reset=spk_reset,
            V_th=V_th,
            V_rest=0.,
            V_reset=0.,
            R=1.,
            V_initializer=braintools.init.ZeroInit()
        )
        self.std = brainpy.state.STD(n_rec, tau=tau_std, U=0.1)
        if inp_std:
            self.std_inp = brainpy.state.STD(n_in, tau=tau_std, U=0.1)
        else:
            self.std_inp = None

        self.syn = brainpy.state.AlignPostProj(
            comm=braintrace.nn.Linear(
                n_in + n_rec, n_rec,
                jnp.concat([ff_init([n_in, n_rec]), rec_init([n_rec, n_rec])], axis=0)
            ),
            syn=brainpy.state.Expon.desc(n_rec, tau=tau_syn, g_initializer=braintools.init.ZeroInit()),
            out=brainpy.state.CUBA.desc(1.),
            post=self.neu
        )

    def update(self, inp_spk):
        if self.std_inp is not None:
            inp_spk = self.std_inp(inp_spk) * inp_spk
        last_spk = self.neu.get_spike()
        inp = jnp.concat([inp_spk, last_spk * self.std(last_spk)], axis=-1)
        self.syn(inp)
        self.neu(0.)
        return self.neu.get_spike()


class LIF_STPExpCu_Dense_Layer(brainstate.nn.Module):
    def __init__(
        self,
        n_in, n_rec, inp_stp=False,
        tau_mem=5., tau_syn=10., V_th=1., tau_f=500., tau_d=100.,
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'soft',
        rec_init: Callable = braintools.init.KaimingNormal(),
        ff_init: Callable = braintools.init.KaimingNormal()
    ):
        super().__init__()
        self.inp_stp = inp_stp
        self.neu = brainpy.state.LIF(
            n_rec,
            tau=tau_mem,
            spk_fun=spk_fun,
            spk_reset=spk_reset,
            V_th=V_th,
            V_rest=0.,
            V_reset=0.,
            R=1.,
            V_initializer=braintools.init.ZeroInit()
        )
        self.stp = brainpy.state.STP(n_rec, tau_f=tau_f, tau_d=tau_d)
        if inp_stp:
            self.stp_inp = brainpy.state.STP(n_in, tau_f=tau_f, tau_d=tau_d)

        self.syn = brainpy.state.AlignPostProj(
            comm=braintrace.nn.Linear(
                n_in + n_rec, n_rec,
                jnp.concat([ff_init([n_in, n_rec]), rec_init([n_rec, n_rec])])
            ),
            syn=brainpy.state.Expon.desc(n_rec, tau=tau_syn, g_initializer=braintools.init.ZeroInit()),
            out=brainpy.state.CUBA.desc(1.),
            post=self.neu
        )

    def update(self, inp_spk):
        if self.inp_stp:
            inp_spk = self.stp_inp(inp_spk) * inp_spk
        last_spk = self.neu.get_spike()
        inp = jnp.concat([inp_spk, last_spk * self.stp(last_spk)], axis=-1)
        self.syn(inp)
        self.neu(0.)
        return self.neu.get_spike()


class LIF_Delta_Dense_Layer(brainstate.nn.Module):
    """
    The RTRL layer with LIF neurons and dense connected delta synapses.
    """

    def __init__(
        self,
        n_in,
        n_rec,
        tau_mem=5.,
        V_th=1.,
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'soft',
        rec_init: Callable = braintools.init.KaimingNormal(),
        ff_init: Callable = braintools.init.KaimingNormal()
    ):
        super().__init__()
        self.neu = brainpy.state.LIF(
            n_rec, tau=tau_mem, spk_fun=spk_fun, spk_reset=spk_reset, V_th=V_th,
            V_rest=0., V_reset=0., R=1., V_initializer=braintools.init.ZeroInit()
        )
        w_init = jnp.concat([ff_init([n_in, n_rec]), rec_init([n_rec, n_rec])], axis=0)
        self.syn = brainpy.state.DeltaProj(
            comm=braintrace.nn.Linear(n_in + n_rec, n_rec, w_init=w_init), post=self.neu
        )

    def update(self, spk):
        inp = jnp.concat([spk, self.neu.get_spike()], axis=-1)
        self.syn(inp)
        self.neu(0.)
        return self.neu.get_spike()


class IF_Delta_Dense_Layer(brainstate.nn.Module):
    """
    The RTRL layer with IF neurons and dense connected delta synapses.
    """

    def __init__(
        self,
        n_in, n_rec,
        tau_mem=5.,
        V_th=1.,
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'soft',
        rec_init: Callable = braintools.init.KaimingNormal(),
        ff_init: Callable = braintools.init.KaimingNormal()
    ):
        super().__init__()
        self.neu = brainpy.state.IF(n_rec, tau=tau_mem, spk_fun=spk_fun, spk_reset=spk_reset, V_th=V_th)
        w_init = jnp.concat([ff_init([n_in, n_rec]), rec_init([n_rec, n_rec])], axis=0)
        self.syn = brainpy.state.DeltaProj(comm=braintrace.nn.Linear(n_in + n_rec, n_rec, w_init=w_init), post=self.neu)

    def update(self, spk):
        spk = jnp.concat([spk, self.neu.get_spike()], axis=-1)
        self.syn(spk)
        self.neu(0.)
        return self.neu.get_spike()


class ALIF_ExpCu_Dense_Layer(brainstate.nn.Module):
    """
    The RTRL layer with LIF neurons and dense connected exponential current synapses.
    """

    def __init__(
        self, n_in, n_rec, tau_mem=5., tau_syn=10., V_th=1., tau_a=100., beta=0.1,
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'soft',
        rec_init: Callable = braintools.init.KaimingNormal(),
        ff_init: Callable = braintools.init.KaimingNormal()
    ):
        super().__init__()
        self.neu = brainpy.state.ALIF(
            n_rec, tau=tau_mem, tau_a=tau_a, beta=beta, spk_fun=spk_fun, spk_reset=spk_reset,
            V_th=V_th, R=1., V_reset=0., V_rest=0.,
            V_initializer=braintools.init.ZeroInit(),
            a_initializer=braintools.init.ZeroInit(),
        )
        self.syn = brainpy.state.AlignPostProj(
            comm=braintrace.nn.Linear(
                n_in + n_rec, n_rec,
                jnp.concat([ff_init([n_in, n_rec]), rec_init([n_rec, n_rec])], axis=0)
            ),
            syn=brainpy.state.Expon.desc(n_rec, tau=tau_syn, g_initializer=braintools.init.ZeroInit()),
            out=brainpy.state.CUBA.desc(1.),
            post=self.neu
        )

    def update(self, spk):
        self.syn(jnp.concat([spk, self.neu.get_spike()], axis=-1))
        self.neu(0.)
        return self.neu.get_spike()


class ALIF_Delta_Dense_Layer(brainstate.nn.Module):
    """
    The RTRL layer with LIF neurons and dense connected delta synapses.
    """

    def __init__(
        self,
        n_in, n_rec, tau_mem=5., tau_a=100., V_th=1., beta=0.1,
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'soft',
        rec_init: Callable = braintools.init.KaimingNormal(),
        ff_init: Callable = braintools.init.KaimingNormal()
    ):
        super().__init__()
        self.neu = brainpy.state.ALIF(
            n_rec,
            tau=tau_mem,
            spk_fun=spk_fun,
            spk_reset=spk_reset,
            V_th=V_th,
            tau_a=tau_a,
            beta=beta,
            R=1.,
            V_reset=0.,
            V_rest=0.,
            V_initializer=braintools.init.ZeroInit(),
            a_initializer=braintools.init.ZeroInit(),
        )
        w_init = jnp.concat([ff_init([n_in, n_rec]), rec_init([n_rec, n_rec])], axis=0)
        self.syn = brainpy.state.DeltaProj(
            comm=braintrace.nn.Linear(n_in + n_rec, n_rec, w_init=w_init),
            post=self.neu
        )

    def update(self, spk):
        inp = jnp.concat([spk, self.neu.get_spike()], axis=-1)
        self.syn(inp)
        self.neu(0.)
        return self.neu.get_spike()


class ALIF_STDExpCu_Dense_Layer(brainstate.nn.Module):
    """
    The RTRL layer with LIF neurons and dense connected STD-based exponential current synapses.
    """

    def __init__(
        self,
        n_in,
        n_rec,
        inp_std=False,
        tau_mem=5.,
        tau_syn=10.,
        V_th=1.,
        tau_std=500.,
        beta=0.1,
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'soft',
        rec_init: Callable = braintools.init.KaimingNormal(),
        ff_init: Callable = braintools.init.KaimingNormal()
    ):
        super().__init__()
        self.neu = brainpy.state.ALIF(
            n_rec,
            tau=tau_mem,
            tau_a=100.,
            spk_fun=spk_fun,
            spk_reset=spk_reset,
            V_th=V_th,
            beta=beta,
            R=1.,
            V_reset=0.,
            V_rest=0.,
            V_initializer=braintools.init.ZeroInit(),
            a_initializer=braintools.init.ZeroInit(),
        )
        self.std = brainpy.state.STD(n_rec, tau=tau_std, U=0.1)
        if inp_std:
            self.std_inp = brainpy.state.STD(n_in, tau=tau_std, U=0.1)
        else:
            self.std_inp = None

        self.syn = brainpy.state.AlignPostProj(
            comm=braintrace.nn.Linear(
                n_in + n_rec, n_rec,
                jnp.concat([ff_init([n_in, n_rec]), rec_init([n_rec, n_rec])], axis=0)
            ),
            syn=brainpy.state.Expon.desc(n_rec, tau=tau_syn, g_initializer=braintools.init.ZeroInit()),
            out=brainpy.state.CUBA.desc(1.),
            post=self.neu
        )

    def update(self, inp_spk):
        if self.std_inp is not None:
            inp_spk = self.std_inp(inp_spk) * inp_spk
        last_spk = self.neu.get_spike()
        inp = jnp.concat([inp_spk, last_spk * self.std(last_spk)], axis=-1)
        self.syn(inp)
        self.neu(0.)
        return self.neu.get_spike()


class ALIF_STPExpCu_Dense_Layer(brainstate.nn.Module):
    def __init__(
        self,
        n_in, n_rec, inp_stp=False, tau_mem=5., tau_syn=10., V_th=1., beta=0.1,
        tau_f=500., tau_d=100., tau_a=100.,
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'soft',
        rec_init: Callable = braintools.init.KaimingNormal(),
        ff_init: Callable = braintools.init.KaimingNormal()
    ):
        super().__init__()
        self.inp_stp = inp_stp
        self.neu = brainpy.state.ALIF(
            n_rec, tau=tau_mem, spk_fun=spk_fun, spk_reset=spk_reset, V_th=V_th, tau_a=tau_a,
            beta=beta, R=1., V_reset=0., V_rest=0.,
            V_initializer=braintools.init.ZeroInit(),
            a_initializer=braintools.init.ZeroInit(),
        )
        self.stp = brainpy.state.STP(n_rec, tau_f=tau_f, tau_d=tau_d)
        if inp_stp:
            self.stp_inp = brainpy.state.STP(n_in, tau_f=tau_f, tau_d=tau_d)

        self.syn = brainpy.state.AlignPostProj(
            comm=braintrace.nn.Linear(
                n_in + n_rec, n_rec,
                jnp.concat([ff_init([n_in, n_rec]), rec_init([n_rec, n_rec])])
            ),
            syn=brainpy.state.Expon.desc(n_rec, tau=tau_syn, g_initializer=braintools.init.ZeroInit()),
            out=brainpy.state.CUBA.desc(1.),
            post=self.neu
        )

    def update(self, inp_spk):
        if self.inp_stp:
            inp_spk = self.stp_inp(inp_spk) * inp_spk
        last_spk = self.neu.get_spike()
        inp = jnp.concat([inp_spk, last_spk * self.stp(last_spk)], axis=-1)
        self.syn(inp)
        self.neu(0.)
        return self.neu.get_spike()


class NetWithMemSpkRegularize(brainstate.nn.Module):
    """
    The class for the RTRL-based network.

    This class implements the bash function for the following regularization:

    - membrane_reg: regularize the membrane potential of the neurons.
    - spike_reg: regularize the firing rate of the neurons.

    """

    def membrane_reg(self, mem_low: float, mem_high: float, factor: float = 0.):
        loss = 0.
        if factor > 0.:
            # extract all Neuron models
            neurons = self.nodes().subset(brainpy.state.Neuron).unique().values()
            # evaluate the membrane potential
            for l in neurons:
                loss += jnp.square(jnp.mean(jax.nn.relu(l.V.value - mem_high) ** 2 +
                                            jax.nn.relu(mem_low - l.V.value) ** 2))
            loss = loss * factor
        return loss

    def spike_reg(self, target_fr: float, factor: float = 0.):
        # target_fr: Hz
        loss = 0.
        if factor > 0.:
            # extract all Neuron models
            neurons = self.nodes().subset(brainpy.state.Neuron).unique().values()
            # evaluate the spiking dynamics
            for l in neurons:
                loss += (jnp.mean(l.spike) - target_fr / 1e3 * brainstate.environ.get_dt()) ** 2
            loss = loss * factor
        return loss

    def visualize_variables(self) -> dict:
        raise NotImplementedError


class ETraceDenseNet(NetWithMemSpkRegularize):
    def __init__(self, n_in, n_rec, n_out, args, spk_fun: Callable = braintools.surrogate.ReluGrad()):
        super().__init__()

        # arguments
        self.n_in = n_in
        self.n_rec = n_rec
        self.n_out = n_out
        self.n_layer = args.n_layer

        # recurrent layers
        self.rec_layers = []
        for layer_idx in range(args.n_layer):
            tau_mem = brainstate.random.normal(args.tau_v, args.tau_v_sigma,
                                               [n_rec]) if args.tau_v_sigma > 0. else args.tau_v
            if args.model == 'lif_expco_dense':
                rec = LIF_ExpCo_Dense_Layer(
                    n_rec=n_rec,
                    n_in=n_in,
                    tau_mem=tau_mem,
                    tau_syn=args.tau_syn,
                    V_th=args.V_th,
                    input_ei_sep=layer_idx != 0,
                    spk_fun=spk_fun,
                    spk_reset=args.spk_reset,
                    rec_init=braintools.init.KaimingNormal(scale=args.rec_wscale),
                    ff_init=braintools.init.KaimingNormal(scale=args.ff_wscale)
                )
                n_in = n_rec
            elif args.model == 'alif_expco_dense':
                rec = ALIF_ExpCo_Dense_Layer(
                    n_rec=n_rec,
                    n_in=n_in,
                    tau_mem=tau_mem,
                    tau_syn=args.tau_syn,
                    V_th=args.V_th,
                    input_ei_sep=layer_idx != 0,
                    spk_fun=spk_fun,
                    spk_reset=args.spk_reset,
                    beta=args.beta, tau_a=args.tau_a,
                    rec_init=braintools.init.KaimingNormal(scale=args.rec_wscale),
                    ff_init=braintools.init.KaimingNormal(scale=args.ff_wscale)
                )
                n_in = n_rec
            elif args.model == 'lif_expcu_dense':
                rec = LIF_ExpCu_Dense_Layer(
                    n_rec=n_rec,
                    n_in=n_in,
                    tau_mem=tau_mem,
                    tau_syn=args.tau_syn,
                    V_th=args.V_th,
                    spk_fun=spk_fun,
                    spk_reset=args.spk_reset,
                    rec_init=braintools.init.KaimingNormal(scale=args.rec_wscale),
                    ff_init=braintools.init.KaimingNormal(scale=args.ff_wscale),
                )
                n_in = n_rec
            elif args.model == 'alif_expcu_dense':
                rec = ALIF_ExpCu_Dense_Layer(
                    n_rec=n_rec,
                    n_in=n_in,
                    tau_mem=tau_mem,
                    tau_syn=args.tau_syn,
                    V_th=args.V_th,
                    spk_fun=spk_fun,
                    spk_reset=args.spk_reset,
                    beta=args.beta,
                    tau_a=args.tau_a,
                    rec_init=braintools.init.KaimingNormal(scale=args.rec_wscale),
                    ff_init=braintools.init.KaimingNormal(scale=args.ff_wscale),
                )
                n_in = n_rec
            elif args.model == 'lif_std_expcu_dense':
                rec = LIF_STDExpCu_Dense_Layer(
                    n_rec=n_rec,
                    n_in=n_in,
                    tau_mem=tau_mem,
                    tau_syn=args.tau_syn,
                    V_th=args.V_th,
                    tau_std=args.tau_std,
                    spk_fun=spk_fun,
                    spk_reset=args.spk_reset,
                    inp_std=layer_idx == 0,
                    rec_init=braintools.init.KaimingNormal(scale=args.rec_wscale),
                    ff_init=braintools.init.KaimingNormal(scale=args.ff_wscale),
                )
                n_in = n_rec
            elif args.model == 'alif_std_expcu_dense':
                rec = ALIF_STDExpCu_Dense_Layer(
                    n_rec=n_rec,
                    n_in=n_in,
                    tau_mem=tau_mem,
                    tau_syn=args.tau_syn,
                    V_th=args.V_th,
                    tau_std=args.tau_std,
                    spk_fun=spk_fun,
                    spk_reset=args.spk_reset,
                    inp_std=layer_idx == 0,
                    rec_init=braintools.init.KaimingNormal(scale=args.rec_wscale),
                    ff_init=braintools.init.KaimingNormal(scale=args.ff_wscale),
                )
                n_in = n_rec
            elif args.model == 'alif_delta_dense':
                rec = ALIF_Delta_Dense_Layer(
                    n_rec=n_rec,
                    n_in=n_in,
                    tau_mem=tau_mem,
                    V_th=args.V_th,
                    spk_fun=spk_fun,
                    spk_reset=args.spk_reset,
                    rec_init=braintools.init.KaimingNormal(scale=args.rec_wscale),
                    ff_init=braintools.init.KaimingNormal(scale=args.ff_wscale),
                )
                n_in = n_rec
            elif args.model == 'lif_delta_dense':
                rec = LIF_Delta_Dense_Layer(
                    n_rec=n_rec,
                    n_in=n_in,
                    tau_mem=tau_mem,
                    V_th=args.V_th,
                    spk_fun=spk_fun,
                    spk_reset=args.spk_reset,
                    rec_init=braintools.init.KaimingNormal(scale=args.rec_wscale),
                    ff_init=braintools.init.KaimingNormal(scale=args.ff_wscale),
                )
                n_in = n_rec
            elif args.model == 'if_delta_dense':
                rec = IF_Delta_Dense_Layer(
                    n_rec=n_rec,
                    n_in=n_in,
                    tau_mem=tau_mem,
                    V_th=args.V_th,
                    spk_fun=spk_fun,
                    spk_reset=args.spk_reset,
                    rec_init=braintools.init.KaimingNormal(scale=args.rec_wscale),
                    ff_init=braintools.init.KaimingNormal(scale=args.ff_wscale),
                )
                n_in = n_rec
            else:
                raise ValueError('Unknown neuron model.')

            self.rec_layers.append(rec)

        # output layer
        self.out = braintrace.nn.LeakyRateReadout(
            in_size=n_rec,
            out_size=n_out,
            tau=args.tau_o,
            w_init=braintools.init.KaimingNormal()
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
            'rec_s': [l.get_spike() for l in neurons],
        }
        return outs


class visualize(object):
    @staticmethod
    def get_gs(gs, i, n_col):
        i_row, i_col = divmod(i, n_col)
        return gs[i_row, i_col]

    @staticmethod
    def plot_multilayer_lif(
        inp_s: jax.Array,
        hid_s: Sequence[jax.Array],
        hid_v: Sequence[jax.Array],
        out_v: jax.Array = None,
        fn: str = None,
        show: bool = False,
        n_col: int = 3,
        num_vis: int = 10
    ):
        n_layer = len(hid_s)
        n_panel = 2 + len(hid_s) + len(hid_v)
        n_row = (n_panel - 1) // n_col + 1
        inp_s = np.reshape(np.asarray(inp_s), (inp_s.shape[0], inp_s.shape[1], -1))
        times = np.arange(inp_s.shape[0]) * brainstate.environ.get_dt()

        for idx in np.arange(0, inp_s.shape[1], max(int(inp_s.shape[1] // num_vis), 1)):
            i = 0
            fig, gs = braintools.visualize.get_figure(n_row, n_col, 3, 4.5)

            # input spikes
            fig.add_subplot(visualize.get_gs(gs, i, n_col))
            spk = u.math.as_numpy(inp_s[:, idx])
            event_times, event_ids = np.where(spk)
            plt.scatter(x=event_times * brainstate.environ.get_dt(), y=event_ids, s=0.5)
            plt.xlim(-1, (inp_s.shape[0] + 1) * brainstate.environ.get_dt())
            plt.ylim(-1, inp_s.shape[2])
            plt.title('Input Spikes')
            plt.xlabel('Time [ms]')
            i += 1

            # hidden spikes
            for j in range(n_layer):
                fig.add_subplot(visualize.get_gs(gs, i, n_col))
                spk = u.math.as_numpy(hid_s[j][:, idx])
                event_times, event_ids = np.where(spk)
                plt.scatter(x=event_times * brainstate.environ.get_dt(), y=event_ids, s=0.5)
                plt.xlim(-1, (hid_s[j].shape[0] + 1) * brainstate.environ.get_dt())
                plt.ylim(-1, hid_s[j].shape[2])
                plt.title(f'Rec Layer {j} Spikes')
                plt.xlabel('Time [ms]')
                i += 1

            # recurrent membrane potentials
            for j in range(n_layer):
                fig.add_subplot(visualize.get_gs(gs, i, n_col))
                plt.plot(times, u.math.as_numpy(hid_v[j][:, idx]))
                plt.xlim(-1, (hid_v[j].shape[0] + 1) * brainstate.environ.get_dt())
                plt.title(f'Rec Layer {j} Potentials')
                plt.xlabel('Time [ms]')
                i += 1

            if out_v is not None:
                # output membrane potentials
                fig.add_subplot(visualize.get_gs(gs, i, n_col))
                plt.plot(times, u.math.as_numpy(out_v[:, idx]))
                plt.xlim(-1, (out_v.shape[0] + 1) * brainstate.environ.get_dt())
                plt.title('Output Activities')
                plt.xlabel('Time [ms]')

            if fn is not None:
                root_path = os.path.dirname(fn)
                if not os.path.exists(root_path):
                    os.makedirs(root_path)
                plt.savefig(fn + f'-idx{idx}.png', transparent=False)
            if show:
                plt.show()
            plt.close(fig)


def _label_processing(y_local):
    if len(y_local.shape) > 1:
        y_local = y_local[:, 0].max(1)[1]
    return jnp.asarray(y_local, dtype=brainstate.environ.ditype())  # (batch,)


class SpikingDataset(Dataset):
    """
    Dataset class for the Spiking Heidelberg Digits (SHD) or
    Spiking Speech Commands (SSC) dataset.

    Arguments
    ---------
    dataset_name : str
        Name of the dataset, either shd or ssc.
    data_folder : str
        Path to folder containing the dataset (h5py file).
    split : str
        Split of the SHD dataset, must be either "train" or "test".
    nb_steps : int
        Number of time steps for the generated spike trains.
    """

    def __init__(
        self,
        dataset_name: str,
        data_folder: str,
        split: str,
        nb_steps: int = 100,
    ):
        # Fixed parameters
        self.device = "cpu"  # to allow pin memory
        self.nb_steps = nb_steps
        self.nb_units = 700
        self.max_time = 1.4
        self.time_bins = np.linspace(0, self.max_time, num=self.nb_steps)

        # Read data from h5py file
        filename = f"{data_folder}/{dataset_name}_{split}.h5"
        self.h5py_file = h5py.File(filename, "r")
        self.firing_times = self.h5py_file["spikes"]["times"]
        self.units_fired = self.h5py_file["spikes"]["units"]
        self.labels = np.array(self.h5py_file["labels"], dtype=np.int_)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        times = np.digitize(self.firing_times[index], self.time_bins)
        units = self.units_fired[index]

        x_idx = torch.LongTensor(np.array([times, units])).to(self.device)
        x_val = torch.FloatTensor(np.ones(len(times))).to(self.device)
        x_size = torch.Size([self.nb_steps, self.nb_units])

        x = torch.sparse_coo_tensor(x_idx, x_val, x_size).to(self.device)
        y = self.labels[index]

        return x.to_dense(), y

    def generate_batch(self, batch):
        xs, ys = zip(*batch)
        xs = torch.nn.utils.rnn.pad_sequence(xs, batch_first=False)
        ys = torch.LongTensor(ys).to(self.device)
        return xs, ys


def _get_shd_data(args, cache_dir=os.path.expanduser("./data"), ):
    # The Spiking Heidelberg Digits (SHD) dataset consists of 20 classes of spoken digits (0-9) spoken by 50 speakers.
    # The SHD dataset is an audio-based classification dataset of 1k spoken digits ranging from zero to nine in
    # the English and German languages. The audio waveforms have been converted into spike trains using an
    # artificial model of the inner ear and parts of the ascending auditory pathway. The SHD dataset has 8,156
    # training and 2,264 test samples. A full description of the dataset and how it was created can be found
    # in the paper below. Please cite this paper if you make use of the dataset.

    data_length = 300

    train_dataset = SpikingDataset('shd', './data/SHD', 'train', data_length)
    test_dataset = SpikingDataset('shd', './data/SHD', 'test', data_length)

    in_shape = SHD.sensor_size
    out_shape = 20

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=train_dataset.generate_batch,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        collate_fn=train_dataset.generate_batch,
        shuffle=False,
    )
    return brainstate.util.DotDict(
        {
            'train_loader': train_loader,
            'test_loader': test_loader,
            'in_shape': in_shape,
            'out_shape': out_shape,
            'label_process': _label_processing,
            'input_process': _sequence_data_for_mlp,
        }
    )

    in_shape = SHD.sensor_size
    out_shape = 20
    transform = tonic.transforms.ToFrame(sensor_size=SHD.sensor_size, n_time_bins=300)
    train_set = SHD(save_to=cache_dir, train=True, transform=transform)
    test_set = SHD(save_to=cache_dir, train=False, transform=transform)
    train_loader = DataLoader(
        train_set,
        shuffle=args.shuffle,
        batch_size=args.batch_size,
        collate_fn=PadTensors(batch_first=False),
        num_workers=args.n_data_worker,
        drop_last=args.drop_last,
    )
    test_loader = DataLoader(
        test_set,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=PadTensors(batch_first=False),
        num_workers=args.n_data_worker
    )

    return brainstate.util.DotDict(
        {
            'train_loader': train_loader,
            'test_loader': test_loader,
            'in_shape': in_shape,
            'out_shape': out_shape,
            'label_process': _label_processing,
            'input_process': _sequence_data_for_mlp,
        }
    )


def _sequence_data_for_mlp(x_local):
    if x_local.ndim > 3:  # (sequence, batch, features, ...)
        x_local = x_local.reshape(x_local.shape[0], x_local.shape[1], -1)
    return jnp.asarray(x_local, dtype=brainstate.environ.dftype())


def _get_gesture_data(args, cache_dir):
    # The Dynamic Vision Sensor (DVS) Gesture (DVSGesture) dataset consists of 11 classes of hand gestures recorded
    # by a DVS sensor. The DVSGesture dataset is a spiking version of the MNIST dataset. The dataset consists of
    # 60k training and 10k test samples.

    out_shape = 11
    n_step = args.data_length
    in_shape = DVSGesture.sensor_size
    transform = tonic.transforms.ToFrame(sensor_size=in_shape, n_time_bins=n_step)
    train_set = DVSGesture(save_to=cache_dir, train=True, transform=transform)
    test_set = DVSGesture(save_to=cache_dir, train=False, transform=transform)
    train_loader = DataLoader(
        train_set,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=PadTensors(batch_first=False),
    )
    test_loader = DataLoader(
        test_set,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=PadTensors(batch_first=False),
    )

    return brainstate.util.DotDict(
        {
            'train_loader': train_loader,
            'test_loader': test_loader,
            'in_shape': in_shape,
            'out_shape': out_shape,
            'label_process': _label_processing,
            'input_process': _sequence_data_for_mlp,
        }
    )


def _get_nmnist_data(args, cache_dir=os.path.expanduser("./data"), first_saccade_only=True):
    # The Neuromorphic-MNIST (N-MNIST) dataset consists of 10 classes of handwritten digits (0-9) recorded by a
    # Dynamic Vision Sensor (DVS) sensor. The N-MNIST dataset is a spiking version of the MNIST dataset. The
    # dataset consists of 60k training and 10k test samples.

    in_shape = NMNIST.sensor_size
    out_shape = 10
    data_length = args.data_length if args.data_length is not None else 1000
    # transform = tonic.transforms.ToFrame(sensor_size=in_shape, time_window=brainstate.environ.get_dt() * 1000)
    transform = tonic.transforms.ToFrame(sensor_size=in_shape, n_time_bins=data_length)
    train_set = NMNIST(save_to=cache_dir, train=True, transform=transform, first_saccade_only=first_saccade_only)
    test_set = NMNIST(save_to=cache_dir, train=False, transform=transform, first_saccade_only=first_saccade_only)
    train_loader = DataLoader(
        train_set,
        shuffle=args.shuffle,
        batch_size=args.batch_size,
        collate_fn=PadTensors(batch_first=False),
        num_workers=args.n_data_worker,
        drop_last=args.drop_last,

    )
    test_loader = DataLoader(
        test_set,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=PadTensors(batch_first=False),
        num_workers=args.n_data_worker
    )

    return brainstate.util.DotDict(
        {
            'train_loader': train_loader,
            'test_loader': test_loader,
            'in_shape': in_shape,
            'out_shape': out_shape,
            'label_process': _label_processing,
            'input_process': _sequence_data_for_mlp
        }
    )


class DataDict(Dict):
    train_loader: DataLoader
    test_loader: DataLoader
    in_shape: tuple
    out_shape: int
    data_type: str
    input_process: callable
    label_process: callable


def get_snn_data(args, cache_dir=os.path.expanduser("./data")) -> DataDict:
    """
    Get the data loader for the specified dataset.
  
    Args:
      args: The arguments.
      cache_dir: The cache directory.
  
    Returns:
      A dictionary with the following keys:
  
      - train_loader: The training data loader.
      - test_loader: The test data loader.
      - in_shape: The input shape.
      - out_shape: The output shape.
      - data_type: The data type ('sequence' or 'static').
      - conv_process: The function to process the data for convolutional networks.
      - mlp_process: The function to process the data for MLPs.
  
    """

    data_to_fun = {
        'SHD': _get_shd_data,
        'N-MNIST': _get_nmnist_data,
        'gesture': _get_gesture_data,
    }
    ret = data_to_fun[args.dataset](args, cache_dir)
    return ret


class RandomDataset(IterableDataset):
    def __init__(self, n_seq: int, n_in: int, n_out: int, prob: float = 0.1):
        super().__init__()
        self.seq_length = n_seq
        self.n_in = n_in
        self.n_out = n_out
        self.prob = prob

    def __iter__(self):
        while True:
            x = np.asarray(np.random.rand(self.seq_length, self.n_in) < self.prob, dtype=np.float32)
            y = np.random.randint(0, self.n_out)
            yield x, y
