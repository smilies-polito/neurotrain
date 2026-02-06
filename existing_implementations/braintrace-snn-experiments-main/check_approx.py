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

import brainstate
import braintools
import jax
import matplotlib.pyplot as plt
import numpy as np


@jax.jit
def cosine_similarity(l1, l2):
    l1 = l1.flatten()
    l2 = l2.flatten()
    norm_a = jax.numpy.linalg.norm(l1)
    norm_b = jax.numpy.linalg.norm(l2)
    cosine = jax.numpy.inner(l1, l2) / norm_a / norm_b
    return cosine


def check_spike_inputs(n_rank=100, m=1000, n=5000, p=0.01):
    print('Check spike inputs')

    # the presynaptic inputs is spikes,
    # so I sample the presynaptic inputs from a binomial distribution
    hid_pre = np.random.binomial(2, p, [n_rank, m])  # U

    for dist, val in [
        ('N(10., 2.)', np.random.normal(10., 2., [n_rank, n])),
        ('|N(1., 10.)|', np.abs(np.random.normal(1., 10., [n_rank, n]))),
        ('N(1., 100.)', np.random.normal(1., 100., [n_rank, n])),
        ('U(0, 1)', np.random.uniform(0, 1., [n_rank, n])),
    ]:
        l1 = np.einsum('ri,rj->ij', hid_pre, val)
        l2 = np.outer(hid_pre.mean(0), val.sum(0))
        cosine = cosine_similarity(l1, l2)
        print(f'n_rank={n_rank}, n={n}, m={m}, {dist}, Similarity={cosine}')


def check_current_inputs(n_rank=100, m=1000, n=5000):
    print('Check current inputs')

    # the presynaptic inputs is current,
    # so I sample the presynaptic inputs from a uniform distribution
    hid_pre = np.random.uniform(0, 1., [n_rank, m])  # U

    for dist, val in [
        ('N(10., 2.)', np.random.normal(10., 2., [n_rank, n])),
        ('|N(1., 10.)|', np.abs(np.random.normal(1., 10., [n_rank, n]))),
        ('U(0, 1)', np.random.uniform(0, 1., [n_rank, n])),
    ]:
        l1 = np.einsum('ri,rj->ij', hid_pre, val)
        l2 = np.outer(hid_pre.mean(0), val.sum(0))
        cosine = cosine_similarity(l1, l2)
        print(f'n_rank={n_rank}, n={n}, m={m}, {dist}, Similarity={cosine}')


def theory_check_spike_inputs(n_rank=100, m=1000, n=5000, p=0.01):
    print('Check spike inputs')

    # the presynaptic inputs is spikes,
    # so I sample the presynaptic inputs from a binomial distribution
    hid_pre = np.random.binomial(2, p, [n_rank, m])  # U
    mu1, sigma1 = p, p * (1 - p)

    for dist, val, mu, sigma in [
        ['N(10., 2.)', np.random.normal(10., 2., [n_rank, n]), 10., 2.],
        # ('|N(1., 10.)|', np.abs(np.random.normal(1., 10., [n_rank, n]))),
        ['N(1., 100.)', np.random.normal(1., 100., [n_rank, n]), 1., 100.],
        ['U(0, 1)', np.random.uniform(0, 1., [n_rank, n]), 0.5, 1. / 12],
    ]:
        l1 = np.einsum('ri,rj->ij', hid_pre, val)
        l2 = np.outer(hid_pre.mean(0), val.sum(0))
        cosine = cosine_similarity(l1, l2)
        ra = mu1 * mu1 / sigma1
        rb = mu * mu / sigma
        theory = np.sqrt(1 - 1 / (n_rank * ra * rb + ra + rb + 1))

        print(f'n_rank={n_rank}, n={n}, m={m}, {dist}, Similarity={cosine}, Theory={theory}')


def align_pre_visualize_x_truncated_normal_df_uniform(r=1000, m=10, n=10):
    @jax.vmap
    def sample(x_scale, b_f):
        b = 1000.
        left = brainstate.random.truncated_normal(0., b, [r, m], x_mu, x_scale)
        # left = brainstate.random.uniform(0., x_scale, [r, m])
        right = brainstate.random.uniform(0., b_f, [r, n])
        l1 = jax.numpy.einsum('ri,rj->ij', left, right)
        l2 = jax.numpy.outer(left.mean(0), right.sum(0))
        cosine = cosine_similarity(l1, l2)

        alpha = (0 - x_mu) / x_scale
        beta = (b - x_mu) / x_scale
        phi_alpha = 1 / np.sqrt(2 * np.pi) * jax.numpy.exp(-0.5 * alpha ** 2)
        phi_beta = 1 / np.sqrt(2 * np.pi) * jax.numpy.exp(-0.5 * beta ** 2)
        Phi_beta = (1 + jax.lax.erf(beta / 2 ** 0.5)) / 2
        Phi_alpha = (1 + jax.lax.erf(alpha / 2 ** 0.5)) / 2
        Z = Phi_beta - Phi_alpha
        mu_ = x_mu + (phi_alpha - phi_beta) / Z * x_scale
        sigma_ = x_scale ** 2 * (1 - (alpha * phi_alpha - beta * phi_beta) / Z - (phi_alpha - phi_beta) ** 2 / Z ** 2)
        ra = mu_ * mu_ / sigma_
        # ra = (0.5 * x_scale) ** 2 / (x_scale ** 2 / 12)
        rb = (0.5 * b_f) ** 2 / (b_f ** 2 / 12)
        theory = jax.numpy.sqrt(1 - 1 / (r * ra * rb + ra + rb + 1))
        return cosine, theory

    x_mu = 0.
    all_x_sigma = np.arange(0., 100., 1.)
    all_b_f = np.arange(0.01, 1., 0.01)
    all_x_sigma2, all_b_f2 = np.meshgrid(all_x_sigma, all_b_f, indexing='ij')

    exp_results, thy_results = sample(all_x_sigma2.flatten(), all_b_f2.flatten())
    exp_results = exp_results.reshape(all_x_sigma2.shape)
    thy_results = thy_results.reshape(all_x_sigma2.shape)

    fig, gs = braintools.visualize.get_figure(1, 1, 4.5, 6.0)
    ax = fig.add_subplot(gs[0, 0])
    pcm = plt.pcolormesh(all_x_sigma2, all_b_f2, exp_results, cmap='PuBu_r', shading='auto')
    cb = plt.colorbar(pcm, ax=ax, extend='max')
    # cb.formatter.set_powerlimits((0, 8))
    # cb.update_ticks
    plt.xlabel('X sigma')
    plt.ylabel('B f')
    plt.title('Cosine similarity')

    fig, gs = braintools.visualize.get_figure(1, 1, 4.5, 6.0)
    ax = fig.add_subplot(gs[0, 0])
    pcm = plt.pcolormesh(all_x_sigma2, all_b_f2, thy_results, cmap='PuBu_r', shading='auto')
    cb = plt.colorbar(pcm, ax=ax, extend='max')
    plt.xlabel('X sigma')
    plt.ylabel('B f')
    plt.title('Theory similarity')
    plt.show()


def align_pre_visualize_x_normal_df_uniform(r=1000, m=100, n=100):
    @jax.vmap
    def sample(x_scale, b_f):
        left = brainstate.random.normal(0., x_scale, [r, m])
        right = brainstate.random.uniform(0., b_f, [r, n])
        l1 = jax.numpy.einsum('ri,rj->ij', left, right)
        l2 = jax.numpy.outer(left.mean(0), right.sum(0))
        cosine = cosine_similarity(l1, l2)

        ra = x_mu * x_mu / x_scale ** 2
        rb = (0.5 * b_f) ** 2 / (b_f ** 2 / 12)
        theory = jax.numpy.sqrt(1 - 1 / (r * ra * rb + ra + rb + 1))
        return cosine, theory

    x_mu = 10.
    all_x_sigma = np.linspace(1., 100., 100)
    all_b_f = np.linspace(0.01, 1., 100)
    all_x_sigma2, all_b_f2 = np.meshgrid(all_x_sigma, all_b_f, indexing='ij')

    exp_results, thy_results = sample(all_x_sigma2.flatten(), all_b_f2.flatten())
    exp_results = exp_results.reshape(all_x_sigma2.shape)
    thy_results = thy_results.reshape(all_x_sigma2.shape)

    fig, gs = braintools.visualize.get_figure(1, 1, 4.5, 6.0)
    ax = fig.add_subplot(gs[0, 0])
    pcm = plt.pcolormesh(all_x_sigma2, all_b_f2, exp_results, cmap='PuBu_r', shading='auto')
    cb = plt.colorbar(pcm, ax=ax, extend='max')
    # cb.formatter.set_powerlimits((0, 8))
    # cb.update_ticks
    plt.xlabel('X sigma')
    plt.ylabel('B f')
    plt.title('Cosine similarity')

    fig, gs = braintools.visualize.get_figure(1, 1, 4.5, 6.0)
    ax = fig.add_subplot(gs[0, 0])
    pcm = plt.pcolormesh(all_x_sigma2, all_b_f2, thy_results, cmap='PuBu_r', shading='auto')
    cb = plt.colorbar(pcm, ax=ax, extend='max')
    plt.xlabel('X sigma')
    plt.ylabel('B f')
    plt.title('Theory similarity')
    plt.show()


def align_pre_visualize_x_binomial_df_uniform(r=10000, m=10, n=10):
    @jax.vmap
    def sample(p, b_f):
        left = jax.numpy.asarray(brainstate.random.rand(r, m) < p, dtype=float)
        right = brainstate.random.uniform(0., b_f, [r, n])
        l1 = jax.numpy.einsum('ri,rj->ij', left, right)
        l2 = jax.numpy.outer(left.mean(0), right.sum(0))
        cosine = cosine_similarity(l1, l2)

        ra = p * p / p / (1 - p)
        rb = (0.5 * b_f) ** 2 / (b_f ** 2 / 12)
        theory = jax.numpy.sqrt(1 - 1 / (r * ra * rb + ra + rb + 1))
        return cosine, theory

    all_x_sigma = np.linspace(0., 0.05, 100)
    all_b_f = np.linspace(0.1, 100., 200)
    all_x_sigma2, all_b_f2 = np.meshgrid(all_x_sigma, all_b_f, indexing='ij')

    exp_results, thy_results = sample(all_x_sigma2.flatten(), all_b_f2.flatten())
    exp_results = exp_results.reshape(all_x_sigma2.shape)
    thy_results = thy_results.reshape(all_x_sigma2.shape)

    fig, gs = braintools.visualize.get_figure(1, 1, 4.5, 6.0)
    ax = fig.add_subplot(gs[0, 0])
    pcm = plt.pcolormesh(all_x_sigma2, all_b_f2, exp_results, cmap='PuBu_r', shading='auto')
    cb = plt.colorbar(pcm, ax=ax, extend='max')
    plt.xlabel('X sigma')
    plt.ylabel('B f')
    plt.title('Cosine similarity')

    fig, gs = braintools.visualize.get_figure(1, 1, 4.5, 6.0)
    ax = fig.add_subplot(gs[0, 0])
    pcm = plt.pcolormesh(all_x_sigma2, all_b_f2, thy_results, cmap='PuBu_r', shading='auto')
    cb = plt.colorbar(pcm, ax=ax, extend='max')
    plt.xlabel('X sigma')
    plt.ylabel('B f')
    plt.title('Theory similarity')
    plt.show()


def align_pre_visualize_x_binomial_df_normal(r=10000, m=10, n=10):
    def sample(p, b_scale):
        left = jax.numpy.asarray(brainstate.random.rand(r, m) < p, dtype=float)
        right = brainstate.random.normal(b_mu, b_scale, [r, n])
        l1 = jax.numpy.einsum('ri,rj->ij', left, right)
        l2 = jax.numpy.outer(left.mean(0), right.sum(0))
        cosine = cosine_similarity(l1, l2)
        # ra = m * p * p / p / (1 - p)
        # rb = n * b_mu * b_mu / b_scale / b_scale
        ra = p * p / p / (1 - p)
        rb = b_mu * b_mu / b_scale / b_scale
        # theory = jax.numpy.sqrt(1 + (1 / r / r - 1 / r) / (ra * rb + (ra + rb + 1) / r))
        theory = jax.numpy.sqrt(1 - 1 / (r * ra * rb + ra + rb + 1))
        return cosine, theory

    b_mu = 1.
    all_p = np.linspace(0., 0.1, 100)
    all_b_scale = np.linspace(0.1, 100., 200)
    all_p2, all_b_scale2 = np.meshgrid(all_p, all_b_scale, indexing='ij')

    exp_results, thy_results = brainstate.transform.for_loop(sample, all_p2.flatten(), all_b_scale2.flatten())
    exp_results = np.asarray(exp_results.reshape(all_p2.shape))
    thy_results = np.asarray(thy_results.reshape(all_p2.shape))

    print(np.nanmin(exp_results), np.nanmax(exp_results))
    print(np.nanmin(thy_results), np.nanmax(thy_results))

    import seaborn
    seaborn.set(font_scale=1.2, style=None)

    # plt.style.use(['science', 'nature', 'notebook'])

    fig, gs = braintools.visualize.get_figure(1, 1, 4.5, 6.0)
    ax = fig.add_subplot(gs[0, 0])
    # pcm = plt.pcolormesh(all_p2, all_b_scale2, exp_results, cmap='PuBu_r', shading='gouraud', vmin=0., vmax=1.)
    pcm = plt.pcolormesh(all_p2, all_b_scale2, exp_results, cmap='PuBu_r', shading='auto', vmin=0., vmax=1.)
    cb = plt.colorbar(pcm, ax=ax, extend='max')
    plt.xlabel('$p$')
    plt.ylabel('$\sigma_f$')
    plt.title('Cosine similarity')
    # plt.savefig('align-pre-cosine-exp.eps', format='eps')

    fig, gs = braintools.visualize.get_figure(1, 1, 4.5, 6.0)
    ax = fig.add_subplot(gs[0, 0])
    pcm = plt.pcolormesh(all_p2, all_b_scale2, thy_results, cmap='PuBu_r', shading='auto', vmin=0., vmax=1.)
    cb = plt.colorbar(pcm, ax=ax, extend='max')
    plt.xlabel('$p$')
    plt.ylabel('$\sigma_f$')
    plt.title('Theoretical similarity')
    # plt.savefig('align-pre-cosine-theory.eps', format='eps')
    plt.show()


if __name__ == '__main__':
    # theory_check_spike_inputs(n_rank=200, n=100, m=200)
    # check_spike_inputs(n_rank=200, n=100, m=200)
    # check_current_inputs(n_rank=2000, n=100, m=200)
    # align_pre_visualize_x_normal_df_uniform()
    align_pre_visualize_x_binomial_df_normal()
    # align_pre_visualize_x_binomial_df_uniform()
