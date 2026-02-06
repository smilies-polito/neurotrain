# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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


import brainunit as u
import jax.numpy as jnp
import numpy as np

from brainstate import environ, random
from brainstate.typing import ArrayLike, SeedOrKey, DTypeLike


def _compute_fans(shape, in_axis=-2, out_axis=-1):
    receptive_field_size = np.prod(shape) / shape[in_axis] / shape[out_axis]
    fan_in = shape[in_axis] * receptive_field_size
    fan_out = shape[out_axis] * receptive_field_size
    return fan_in, fan_out


class Orthogonal:
    def __init__(
        self,
        scale: ArrayLike = 1.,
        axis: int = -1,
        seed: SeedOrKey = None,
        unit: u.Unit = u.UNITLESS,
    ):
        super().__init__()
        self.scale = scale
        self.axis = axis
        self.rng = random.default_rng(seed)
        self.unit = unit

    def __call__(self, shape, dtype: DTypeLike = None, ):
        dtype = dtype or environ.dftype()
        n_rows = shape[self.axis]
        n_cols = np.prod(shape) // n_rows
        matrix_shape = (n_rows, n_cols) if n_rows > n_cols else (n_cols, n_rows)
        norm_dst = self.rng.normal(size=matrix_shape, dtype=dtype)

        q_mat, r_mat = jnp.linalg.qr(norm_dst)
        # Enforce Q is uniformly distributed
        q_mat *= jnp.sign(jnp.diag(r_mat))
        if n_rows < n_cols:
            q_mat = q_mat.T
        q_mat = jnp.reshape(q_mat, (n_rows,) + tuple(np.delete(shape, self.axis)))
        q_mat = jnp.moveaxis(q_mat, 0, self.axis)
        r = jnp.asarray(self.scale, dtype=dtype) * q_mat
        return u.maybe_decimal(u.Quantity(r, unit=self.unit))


class VarianceScaling:
    def __init__(
        self,
        scale: ArrayLike,
        mode: str,
        distribution: str,
        in_axis: int = -2,
        out_axis: int = -1,
        seed: SeedOrKey = None,
        unit: u.Unit = u.UNITLESS,
    ):
        assert mode in ['fan_in', 'fan_out', 'fan_avg']
        assert distribution in ['truncated_normal', 'normal', 'uniform']
        self.scale = scale
        self.mode = mode
        self.in_axis = in_axis
        self.out_axis = out_axis
        self.distribution = distribution
        self.rng = random.default_rng(seed)
        self.unit = unit

    def __call__(self, shape, dtype: DTypeLike = None, ):
        dtype = dtype or environ.dftype()
        fan_in, fan_out = _compute_fans(shape, in_axis=self.in_axis, out_axis=self.out_axis)
        if self.mode == "fan_in":
            denominator = fan_in
        elif self.mode == "fan_out":
            denominator = fan_out
        elif self.mode == "fan_avg":
            denominator = (fan_in + fan_out) / 2
        else:
            raise ValueError("invalid mode for variance scaling initializer: {}".format(self.mode))
        variance = (self.scale / denominator).astype(dtype)
        if self.distribution == "truncated_normal":
            stddev = (jnp.sqrt(variance) / .87962566103423978).astype(dtype)
            res = self.rng.truncated_normal(-2, 2, shape, dtype=dtype) * stddev
        elif self.distribution == "normal":
            res = self.rng.randn(*shape, dtype=dtype) * jnp.sqrt(variance).astype(dtype)
        elif self.distribution == "uniform":
            res = (
                self.rng.uniform(low=-1, high=1, size=shape, dtype=dtype) *
                jnp.sqrt(3 * variance).astype(dtype)
            )
        else:
            raise ValueError("invalid distribution for variance scaling initializer")
        return u.maybe_decimal(u.Quantity(res, unit=self.unit))


class KaimingUniform(VarianceScaling):
    def __init__(
        self,
        scale: float = 2.0,
        mode: str = "fan_in",
        distribution: str = "uniform",
        in_axis: int = -2,
        out_axis: int = -1,
        seed: SeedOrKey = None,
        unit: u.Unit = u.UNITLESS,
    ):
        super().__init__(
            scale,
            mode,
            distribution,
            in_axis=in_axis,
            out_axis=out_axis,
            seed=seed,
            unit=unit
        )
