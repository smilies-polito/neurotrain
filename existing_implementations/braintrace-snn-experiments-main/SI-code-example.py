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

import brainpy
import braintrace
import brainstate
import braintools
import jax
import jax.numpy as jnp


class Linear(brainstate.nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        w_init=braintools.init.KaimingNormal(),
        b_init=braintools.init.ZeroInit()
    ):
        super().__init__()
        self.in_size = (n_in,)
        self.out_size = (n_out,)

        # parameters
        param = dict(weight=w_init([n_in, n_out]), bias=b_init([n_out]))
        # operations
        self.weight_op = braintrace.ETraceParam(param, braintrace.MatMulOp())

    def update(self, x):
        # call the model by ".execute" the weight_op
        return self.weight_op.execute(x)


class LIF(brainpy.state.Neuron):
    def __init__(
        self,
        in_size,
        tau: float = 5.,
        V_th: float = 1.,
        V_reset: float = 0.,
        V_rest: float = 0.,
        spk_fun=braintools.surrogate.ReluGrad(),
        spk_reset='soft'
    ):
        super().__init__(in_size, spk_fun=spk_fun, spk_reset=spk_reset)

        # parameters
        self.tau = tau
        self.V_th = V_th
        self.V_rest = V_rest
        self.V_reset = V_reset

    def dv(self, v, t, x):
        # "sum_current_inputs()" sums up all incoming currents
        x = self.sum_current_inputs(x, v)
        # the differential equation
        return (-v + self.V_rest + x) / self.tau

    def init_state(self, batch_size: int = None, **kwargs):
        # initialize the membrane potential
        bs = () if batch_size is None else (batch_size,)
        V = jax.numpy.full(bs + self.varshape, self.V_rest)
        self.V = brainstate.HiddenState(V)

    def get_spike(self, V=None):
        V = self.V.value if V is None else V
        # scale the membrane potential
        v_scaled = (V - self.V_th) / self.V_th
        # generate spike using the surrogate gradient function
        return self.spk_fun(v_scaled)

    def update(self, x=0.):
        # the last spike and membrane potential
        last_v = self.V.value
        lst_spk = self.get_spike(last_v)
        if self.spk_reset == 'soft':
            V_th = self.V_th
        else:
            V_th = jax.lax.stop_gradient(last_v)
        V = last_v - (V_th - self.V_reset) * lst_spk
        # the current membrane potential
        V = brainstate.nn.exp_euler_step(self.dv, V, None, x)
        V = self.sum_delta_inputs(V)
        self.V.value = V
        # the current spike
        return self.get_spike(V)


class LIF_Delta_Net(brainstate.nn.Module):
    def __init__(
        self,
        n_in: int,
        n_rec: int,
        n_out: int,
        tau_mem: float = 5.,
        V_th: float = 1.,
    ):
        super().__init__()
        self.neu = LIF(n_rec, tau=tau_mem, V_th=V_th)
        self.syn = brainpy.state.DeltaProj(comm=Linear(n_in + n_rec, n_rec), post=self.neu)
        self.out = braintrace.nn.LeakyRateReadout(n_rec, n_out, tau=5.0)

    def update(self, i, spk):
        with brainstate.environ.context(i=i, t=i * brainstate.environ.get_dt()):
            spk = jnp.concat([spk, self.neu.get_spike()], axis=-1)
            self.syn(spk)
            return self.out(self.neu())


brainstate.environ.set(dt=1.0)

# define the one-batch inputs and targets
n_seq = 512
inputs = brainstate.random.rand(n_seq, 10) < 0.1
targets = brainstate.random.randint(0, 2)

# instantiate a spiking network
net = LIF_Delta_Net(10, 100, 2)
brainstate.nn.init_all_states(net)

# online learning algorithm
method = 'es-diag'  # or 'diag', 'hybrid'
if method == 'es-diag':
    model = braintrace.ES_D_RTRL(net, decay_or_rank=0.98)
elif method == 'diag':
    model = braintrace.D_RTRL(net)
elif method == 'hybrid':
    model = braintrace.HybridDimVjpAlgorithm(net, decay_or_rank=0.98)
else:
    raise ValueError(f'Unknown online learning methods: {method}.')

# compile the eligibility trace graph using the one-step input
model.compile_graph(0, inputs[0])

# retrieve parameters that need to compute gradients
weights = net.states(brainstate.ParamState)


# define loss function
def loss_fn(i, x):
    out = model(i, x)
    return jnp.mean(braintools.metric.softmax_cross_entropy_with_integer_labels(out, targets))


# gradient computation using traditional autograd interface
def step_grad(last_grads, ix):
    # gradients computed at the current step
    grads = brainstate.transform.grad(loss_fn, weights)(ix[0], ix[1])
    # accumulate gradients: prev + current
    new_grads = jax.tree.map(jax.numpy.add, last_grads, grads)
    return new_grads, None


# loop over all sequences
indices = jax.numpy.arange(n_seq)
init_grads = jax.tree.map(jax.numpy.zeros_like, weights.to_dict_values())
grads = brainstate.transform.scan(step_grad, init_grads, (indices, inputs))

