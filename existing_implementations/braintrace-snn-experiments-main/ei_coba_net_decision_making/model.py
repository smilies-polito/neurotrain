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

from typing import Callable

import brainpy
import braintrace
import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from general_utils import raster_plot


class GIF(brainpy.state.Neuron):
    def __init__(
        self,
        size,
        V_rest: float = 0.,
        V_th_inf: float = 1.,
        tau: float = 20.,
        tau_th: float = 100.,
        A2: float = 1.,
        tau_I2: float = 50.,
        A1: float = 0.01,
        tau_I1: float = 50.,
        Ath: float = 1.,
        diff_spike: bool = True,
        V_initializer: Callable = braintools.init.Uniform(0., 1.),
        I1_initializer: Callable = braintools.init.ZeroInit(),
        I2_initializer: Callable = braintools.init.ZeroInit(),
        Vth_initializer: Callable = braintools.init.Constant(1.),
        spike_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'soft',
        name: str = None,
    ):
        super().__init__(size, name=name, spk_fun=spike_fun, spk_reset=spk_reset)

        # params
        self.diff_spike = diff_spike
        self.V_rest = V_rest
        self.V_th_inf = V_th_inf
        self.tau_rev = 1 / tau
        self.tau_I2_rev = 1 / tau_I2
        self.A2 = A2
        self.tau_I1_rev = 1 / tau_I1
        self.A1 = A1
        self.tau_th_rev = 1 / tau_th
        self.Ath = Ath

        # initializers
        self.V_initializer = V_initializer
        self.I2_initializer = I2_initializer
        self.I1_initializer = I1_initializer
        self.Vth_initializer = Vth_initializer

    @property
    def num(self):
        return self.varshape[0]

    def init_state(self):
        self.V = brainstate.HiddenState(self.V_initializer(self.varshape))
        self.I2 = brainstate.HiddenState(self.I2_initializer(self.varshape))
        self.Vth = brainstate.HiddenState(self.Vth_initializer(self.varshape))
        self.I1 = brainstate.HiddenState(self.I1_initializer(self.varshape))

    def update(self, x=0.):
        last_spk = self.get_spike()
        if not self.diff_spike:
            last_spk = jax.lax.stop_gradient(last_spk)
        V_th = self.Vth.value
        last_V = self.V.value - V_th * last_spk
        V_th = V_th + self.Ath * last_spk
        last_I2 = self.I2.value - self.A2 * last_spk
        last_I1 = self.I1.value + self.A1 * last_spk

        dI2 = lambda I2: - I2 * self.tau_I2_rev
        I2 = brainstate.nn.exp_euler_step(dI2, last_I2)

        dI1 = lambda I1: - I1 * self.tau_I1_rev
        I1 = brainstate.nn.exp_euler_step(dI1, last_I1)

        dVth = lambda Vth: -(V_th - self.V_th_inf) * self.tau_th_rev
        Vth = brainstate.nn.exp_euler_step(dVth, V_th)

        dV = lambda V, I_ext: (- V + self.V_rest + self.sum_current_inputs(I_ext, V)) * self.tau_rev
        V = brainstate.nn.exp_euler_step(dV, last_V, I_ext=(x + I1 + I2))
        V = self.sum_delta_inputs(V)

        self.I1.value = I1
        self.I2.value = I2
        self.Vth.value = Vth
        self.V.value = V
        # return (V - Vth) / Vth

        # outputs
        Vth = jax.lax.stop_gradient(Vth)
        # mem = (V - Vth) / jax.lax.stop_gradient(Vth)
        mem = (V - Vth) / Vth
        mem = jax.nn.standardize(mem, axis=-1)
        return mem

    def get_spike(self, V=None):
        V = self.V.value if V is None else V
        # spk = self.spk_fun((V - self.V_th_inf) / self.V_th_inf)
        # spk = self.spk_fun((V - V_th) / jax.lax.stop_gradient(V_th))
        V_th = jax.lax.stop_gradient(self.Vth.value * 1.0)
        # V_th = self.Vth.value * 1.0
        spk = self.spk_fun((V - V_th) / V_th)
        return spk


class Linear(braintrace.nn.Linear):
    def __init__(self, n_in, n_out, w_init=None, sparsity: float = 0., sign: float = 1.):
        if sparsity > 0.:
            w_mask = brainstate.random.rand(n_in, n_out, dtype=jnp.float32) < sparsity
        else:
            w_mask = None
        self.sign = sign
        super().__init__(n_in, n_out, w_init=w_init, w_mask=w_mask)

    def update(self, x):
        r = super().update(x)
        return u.math.abs(r) * self.sign


class _SNNEINet(brainstate.nn.Module):
    def __init__(
        self,
        n_in,
        n_rec,
        n_out,
        args,
        filepath,
        E_exc,
        E_inh,
    ):
        super().__init__()
        brainstate.random.seed(args.seed)

        self.filepath = filepath
        self.n_exc = int(n_rec * 0.8)
        self.n_inh = n_rec - self.n_exc
        self.args = args

        # neurons
        self.pop = GIF(
            n_rec,
            tau=args.tau_neu,
            tau_I2=args.tau_I2,
            A2=args.A2,
            tau_I1=args.tau_I1,
            A1=args.A1,
            tau_th=args.tau_th,
            Ath=args.Ath,
            diff_spike=args.diff_spike,
        )

        # feedforward
        self.ff2r = brainpy.state.AlignPostProj(
            comm=Linear(
                n_in,
                n_rec,
                w_init=braintools.init.KaimingNormal(scale=args.ff_scale),
                sparsity=args.sparsity
            ),
            syn=brainpy.state.Expon.desc(
                in_size=n_rec,
                tau=args.tau_syn,
                g_initializer=braintools.init.ZeroInit()
            ),
            out=brainpy.state.COBA.desc(E=E_exc),
            post=self.pop
        )

        self.inh2r = brainpy.state.AlignPostProj(
            comm=Linear(
                self.n_inh,
                n_rec,
                w_init=braintools.init.KaimingNormal(scale=args.rec_scale * args.w_ei_ratio),
                sparsity=args.sparsity
            ),
            syn=brainpy.state.Expon.desc(
                in_size=n_rec,
                tau=args.tau_syn,
                g_initializer=braintools.init.ZeroInit()
            ),
            out=brainpy.state.COBA.desc(E=E_inh),
            post=self.pop
        )
        self.exc2r = brainpy.state.AlignPostProj(
            comm=Linear(
                self.n_exc,
                n_rec,
                w_init=braintools.init.KaimingNormal(scale=args.rec_scale),
                sparsity=args.sparsity
            ),
            syn=brainpy.state.Expon.desc(
                in_size=n_rec,
                tau=args.tau_syn,
                g_initializer=braintools.init.ZeroInit()
            ),
            out=brainpy.state.COBA.desc(E=E_exc),
            post=self.pop
        )

        # output
        self.out = braintrace.nn.LeakyRateReadout(n_rec, n_out, tau=args.tau_out)

    def update(self, spk):
        e_sps, i_sps = jnp.split(self.pop.get_spike(), [self.n_exc], axis=-1)
        self.ff2r(spk)
        self.exc2r(e_sps)
        self.inh2r(i_sps)
        return self.out(self.pop())

    @brainstate.transform.jit(static_argnums=0)
    def predict(self, batched_inputs):
        # batched_inputs: [n_seq, n_in]
        brainstate.nn.vmap_init_all_states(self, axis_size=batched_inputs.shape[1], state_tag='new')

        def step(inp):
            model = brainstate.nn.Vmap(self, vmap_states='new')
            out = model(inp)
            spk = self.pop.get_spike()
            rec_mem = self.pop.V.value
            return spk, rec_mem, out

        res = brainstate.transform.for_loop(step, batched_inputs, pbar=brainstate.transform.ProgressBar(10))
        return res

    def visualize(self, inputs, n2show: int = 5, filename: str = None):
        n_seq = inputs.shape[0]
        n_rec = self.pop.num
        indices = np.arange(0, n_rec, n_rec // 10)
        res = self.predict(inputs)
        res = {'rec_spk': res[0], 'rec_mem': res[1][..., indices], 'out': res[2]}

        indices = np.arange(n_seq)
        fig, gs = braintools.visualize.get_figure(4, n2show, 3., 4.5)
        for i in range(n2show):
            # input spikes
            raster_plot(indices, inputs[:, i], ax=fig.add_subplot(gs[0, i]), xlim=(0, n_seq))
            # recurrent spikes
            raster_plot(indices, res['rec_spk'][:, i], ax=fig.add_subplot(gs[1, i]), xlim=(0, n_seq))
            # recurrent membrane potentials
            ax = fig.add_subplot(gs[2, i])
            ax.plot(indices, res['rec_mem'][:, i])
            # output potentials
            ax = fig.add_subplot(gs[3, i])
            ax.plot(indices, res['out'][:, i])

        if filename is None:
            plt.show()
            plt.close()
        else:
            plt.savefig(filename)
            plt.close()


class SNNCubaNet(_SNNEINet):
    def __init__(
        self,
        n_in,
        n_rec,
        n_out,
        args,
        filepath=None,
    ):
        super().__init__(
            n_in=n_in,
            n_rec=n_rec,
            n_out=n_out,
            E_exc=None,
            E_inh=None,
            args=args,
            filepath=filepath,
        )


class SNNCobaNet(_SNNEINet):
    def __init__(
        self,
        n_in,
        n_rec,
        n_out,
        args,
        filepath=None,
    ):
        super().__init__(
            n_in=n_in,
            n_rec=n_rec,
            n_out=n_out,
            E_exc=5.,
            E_inh=-10.,
            args=args,
            filepath=filepath,
        )
