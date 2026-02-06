# Copyright 2025 BDP Ecosystem Limited. All Rights Reserved.
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

# -*- coding: utf-8 -*-


import brainstate
import brainunit as u
import jax
import jax.numpy as jnp


class EvidenceAccumulation:
    metadata = {
        'paper_link': 'https://doi.org/10.1038/nn.4403',
        'paper_name': 'History-dependent variability in population dynamics during evidence accumulation in cortex',
    }

    def __init__(
        self,
        t_interval=50. * u.ms,
        t_cue=100. * u.ms,
        t_delay=1000. * u.ms,
        t_recall=150. * u.ms,
        prob: float = 0.3,
        num_cue: int = 7,
        batch_size: int = 128,
        # number of neurons:
        #         left, right, recall, noise
        n_neurons=(25, 25, 25, 25),
        firing_rates=(40., 40., 40., 10.) * u.Hz,
    ):

        # input / output information
        self.batch_size = batch_size

        # time
        self.t_interval = t_interval
        self.t_cue = t_cue
        self.t_delay = t_delay
        self.t_recall = t_recall

        # features
        self.n_neurons = n_neurons
        self.feat_neurons = {
            'left': slice(0, n_neurons[0]),
            'right': slice(n_neurons[0],
                           n_neurons[0] + n_neurons[1]),
            'recall': slice(n_neurons[0] + n_neurons[1],
                            n_neurons[0] + n_neurons[1] + n_neurons[2]),
            'noise': slice(n_neurons[0] + n_neurons[1] + n_neurons[2],
                           n_neurons[0] + n_neurons[1] + n_neurons[2] + n_neurons[3]),
        }
        self.feat_fr = {
            'left': firing_rates[0],
            'right': firing_rates[1],
            'recall': firing_rates[2],
            'noise': firing_rates[3],
        }

        self.firing_rates = firing_rates
        self.prob = prob
        self.num_cue = num_cue

        # input / output information
        dt = brainstate.environ.get_dt()
        t_interval = int(self.t_interval / dt)
        t_cue = int(self.t_cue / dt)
        t_delay = int(self.t_delay / dt)
        t_recall = int(self.t_recall / dt)

        _time_periods = dict()
        for i in range(self.num_cue):
            _time_periods[f'interval {i}'] = t_interval
            _time_periods[f'cue {i}'] = t_cue
        _time_periods['delay'] = t_delay
        _time_periods['recall'] = t_recall
        self.periods = _time_periods
        t_total = sum(_time_periods.values())
        self.n_sim = t_total - t_recall

        def sample_a_trial(key):
            rng = brainstate.random.RandomState(key)

            # assign input spike probability
            ground_truth = rng.rand() < 0.5
            prob = u.math.where(ground_truth, self.prob, 1 - self.prob)

            # for each example in batch, draw which cues are going to be active (left or right)
            cue_assignments = u.math.asarray(rng.random(self.num_cue) > prob, dtype=int)

            X = jnp.zeros((t_total, self.num_inputs))
            # generate input spikes
            for k in range(self.num_cue):
                # input channels only fire when they are selected (left or right)
                i_start = u.math.where(cue_assignments[k],
                                       self.feat_neurons['left'].start,
                                       self.feat_neurons['right'].start)
                fr = u.math.where(cue_assignments[k], self.feat_fr['left'], self.feat_fr['right']) * dt
                update = jnp.ones((t_cue, 25)) * fr

                # reverse order of cues
                i_seq = t_interval + k * (t_interval + t_cue)
                # X[i_seq:i_seq + t_cue, i_start: i_start + 25] = fr
                X = jax.lax.dynamic_update_slice(X, update, (i_seq, i_start))

            X = u.Quantity(X)
            # recall cue
            X[-t_recall:, self.feat_neurons['recall']] = self.feat_fr['recall'] * dt

            # background noise
            X[:, self.feat_neurons['noise']] = self.feat_fr['noise'] * dt

            # generate inputs and targets
            # X = u.math.asarray(rng.rand(*X.shape) < X, dtype=float)
            X = rng.rand(*X.shape) < X
            Y = u.math.asarray(u.math.sum(cue_assignments) > (self.num_cue / 2), dtype=int)
            return X, Y

        self.sampling = jax.jit(jax.vmap(sample_a_trial))

    @property
    def num_inputs(self) -> int:
        return sum(self.n_neurons)

    @property
    def num_outputs(self) -> int:
        return 2

    def __iter__(self):
        while True:
            yield self.sampling(brainstate.random.split_key(self.batch_size))




