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

import argparse
import glob
import logging
import os
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np


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


class MyArgumentParser(argparse.ArgumentParser):
    def __init__(
        self,
        *args,
        gpu_pre_allocate=0.99,
        device='0',
        method='bptt',
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.add_argument('--devices', type=str, default=device, help='The GPU device ids.')
        self.add_argument("--method", type=str, default=method, choices=['bptt', 'd-rtrl', 'esd-rtrl'])
        args, _ = self.parse_known_args()

        # device management
        _set_gpu_device(args.devices)
        _set_gpu_preallocation(gpu_pre_allocate)

        # training method
        if args.method != 'bptt':
            self.add_argument("--vjp_method", type=str, default='multi-step', choices=['multi-step', 'single-step'])
            if args.method != 'd-rtrl':
                self.add_argument("--etrace_decay", type=float, default=0.99,
                                  help="The time constant of eligibility trace.")


def copy_files(tar_dir, dest_dir):
    for filename in glob.glob(os.path.join(tar_dir, '*.py')):
        print(filename)
        shutil.copy(filename, dest_dir, follow_symlinks=True)


def save_model_states(
    save_path: str,
    model,
    optimizer=None,
    **kwargs
):
    """
    Save the current state of the model, optimizer, and training progress.

    This function creates a dictionary containing the current epoch, accuracy,
    model state, and optimizer state, then saves it to a file using MessagePack format.

    Parameters:
    -----------
    model : brainstate.nn.Module
        The neural network model whose state is to be saved.
    optimizer : braintools.optim.Optimizer
        The optimizer used for training, whose state is to be saved.
    epoch : int
        The current epoch number.
    accuracy : float
        The current accuracy of the model.
    save_path : str
        The file path where the model state will be saved.

    Returns:
    --------
    None
        This function doesn't return any value, but it saves the state to a file
        and prints a confirmation message.
    """
    import brainstate
    import braintools
    state = {
        'state_dict': model.states(brainstate.LongTermState),
        **kwargs
    }
    if optimizer is not None:
        state['optimizer_state_dict'] = brainstate.graph.states(optimizer)
    braintools.file.msgpack_save(save_path, state)


def load_model_states(
    save_path: str,
    model,
    optimizer=None,
    **kwargs
):
    """
    Save the current state of the model, optimizer, and training progress.

    This function creates a dictionary containing the current epoch, accuracy,
    model state, and optimizer state, then saves it to a file using MessagePack format.

    Parameters:
    -----------
    model : brainstate.nn.Module
        The neural network model whose state is to be saved.
    optimizer : braintools.optim.Optimizer
        The optimizer used for training, whose state is to be saved.
    epoch : int
        The current epoch number.
    accuracy : float
        The current accuracy of the model.
    save_path : str
        The file path where the model state will be saved.
    """
    import brainstate
    import braintools
    state = {
        'state_dict': model.states(brainstate.LongTermState),
        **kwargs
    }
    if optimizer is not None:
        state['optimizer_state_dict'] = brainstate.graph.states(optimizer)
    pytree = braintools.file.msgpack_load(save_path, state)
    return pytree


def setup_logging(log_file: str) -> logging.Logger:
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)  # Set the minimum logging level

    # Create a formatter to customize the log message format
    # formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(message)s')

    # Create a StreamHandler to output to stdout
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.WARNING)  # Set the logging level for stdout
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    # Create a FileHandler to output to a file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.WARNING)  # Set the logging level for the file
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', unit=''):
        self.name = name
        self.fmt = fmt
        self.unit = unit
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '}' + self.unit + ' ({avg' + self.fmt + '}' + self.unit + ')'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        return ', '.join(entries)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


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
