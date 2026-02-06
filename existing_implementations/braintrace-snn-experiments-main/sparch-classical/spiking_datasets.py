#
# SPDX-FileCopyrightText: Copyright © 2022 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Alexandre Bittar <abittar@idiap.ch>
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This file is part of the sparch package
#
"""
This is where the dataloader is defined for the SHD and SSC datasets.
"""

import os
import platform

import brainstate
import h5py
import numpy as np
import tonic
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import (
    RandomPerspective,
    RandomResizedCrop,
    RandomRotation,
    InterpolationMode,
)

__all__ = [
    'load_dataset',
]


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

        # Read data from h5py file
        filename = f"{data_folder}/{dataset_name}_{split}.h5"
        self.h5py_file = h5py.File(filename, "r")
        self.firing_times = self.h5py_file["spikes"]["times"]
        max_time = 0.
        for i in range(self.firing_times.shape[0]):
            mx = np.max(self.firing_times[i])
            if mx > max_time:
                max_time = mx
        self.max_time = max_time
        self.time_bins = np.linspace(0, self.max_time, num=self.nb_steps)
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
        xs = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)
        ys = torch.LongTensor(ys).to(self.device)

        # xlens = torch.tensor([x.shape[0] for x in xs])
        # return xs, xlens, ys
        return xs, ys


class FormattedDataset:
    def __init__(self, filepath: str):
        self.filepath = filepath
        data = np.load(filepath, allow_pickle=True)
        self.xs_row = data['xs_row']
        self.xs_col = data['xs_col']
        self.xs_data = data['xs_data']
        self.ys = data['ys']
        img_size = data['img_size']
        self.img_size = (int(img_size[0]), int(np.prod(img_size[1:])))

    def load(self, idx):
        arr = np.zeros(self.img_size, dtype=np.float32)
        row = self.xs_row[idx]
        col = self.xs_col[idx]
        arr[row, col] = self.xs_data[idx]
        return arr

    def __getitem__(self, idx):
        arr = self.load(idx)
        y = self.ys[idx]
        return arr, y

    def __len__(self):
        return len(self.ys)


# We need to stack the batch elements
def _numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [_numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


def load_nmnist_data(args, first_saccade_only=True):
    from tonic.datasets import NMNIST
    import tonic

    in_shape = NMNIST.sensor_size
    out_shape = 10

    def flatten(x):
        return np.asarray(x.reshape(x.shape[0], -1), dtype=np.float32)

    transform = tonic.transforms.ToFrame(sensor_size=in_shape, n_time_bins=args.data_length)
    transform = tonic.transforms.Compose([transform, flatten])
    train_set = NMNIST(save_to=args.data_folder, train=True, transform=transform, first_saccade_only=first_saccade_only)
    test_set = NMNIST(save_to=args.data_folder, train=False, transform=transform, first_saccade_only=first_saccade_only)
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=_numpy_collate,
        num_workers=args.num_workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_set,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=_numpy_collate,
        num_workers=args.num_workers,
    )

    return brainstate.util.DotDict(
        {
            'train_loader': train_loader,
            'test_loader': test_loader,
            'in_shape': int(np.prod(in_shape)),
            'out_shape': out_shape,
        }
    )


def load_nmnist_data_v2(args, first_saccade_only=True):
    in_shape = int(np.prod((34, 34, 2)))
    out_shape = 10
    n_step = args.data_length
    cache_dir = args.data_folder
    train_path = os.path.join(cache_dir, f"NMNIST/NMNIST-train-step={n_step}.npz")
    train_path = os.path.abspath(train_path)
    test_path = os.path.join(cache_dir, f"NMNIST/NMNIST-test-step={n_step}.npz")
    test_path = os.path.abspath(test_path)
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise ValueError(
            f'Cache files {train_path} and {test_path} do not exist. '
            f'please run "dvs-gesture-preprocessing.py" first.'
        )
    else:
        print(f'Used cache files {train_path} and {test_path}.')
    train_set = FormattedDataset(train_path)
    test_set = FormattedDataset(test_path)

    if args.use_augm:
        pass

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=_numpy_collate,
        num_workers=args.num_workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_set,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=_numpy_collate,
        num_workers=args.num_workers
    )

    return brainstate.util.DotDict(
        {
            'train_loader': train_loader,
            'test_loader': test_loader,
            'in_shape': int(np.prod(in_shape)),
            'out_shape': out_shape,
        }
    )


def load_shd_data(args):
    train_dataset = SpikingDataset('shd', args.data_folder, 'train', args.data_length)
    test_dataset = SpikingDataset('shd', args.data_folder, 'test', args.data_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=train_dataset.generate_batch,
        shuffle=True,
        num_workers=0 if platform.platform() else args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        collate_fn=train_dataset.generate_batch,
        shuffle=False,
        num_workers=0 if platform.platform() else args.num_workers,
    )
    return brainstate.util.DotDict(
        {
            'train_loader': train_loader,
            'test_loader': test_loader,
            'in_shape': 700,
            'out_shape': 20,
            'input_process': lambda x: x,
        }
    )


def load_ssc_data(args):
    train_dataset = SpikingDataset('ssc', args.data_folder, 'train', args.data_length)
    test_dataset = SpikingDataset('ssc', args.data_folder, 'test', args.data_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=train_dataset.generate_batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        collate_fn=test_dataset.generate_batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return brainstate.util.DotDict(
        {
            'train_loader': train_loader,
            'test_loader': test_loader,
            'in_shape': 700,
            'out_shape': 35,
            'input_process': lambda x: x,
        }
    )


def load_gesture_data(args):
    # The Dynamic Vision Sensor (DVS) Gesture (DVSGesture) dataset consists of 11 classes of hand gestures recorded
    # by a DVS sensor. The DVSGesture dataset is a spiking version of the MNIST dataset. The dataset consists of
    # 60k training and 10k test samples.

    in_shape = int(np.prod((128, 128, 2)))
    out_shape = 11
    n_step = args.data_length
    cache_dir = args.data_folder
    train_path = os.path.join(cache_dir, f"DVSGesture/DVSGesture-mlp-train-step={n_step}.npz")
    train_path = os.path.abspath(train_path)
    test_path = os.path.join(cache_dir, f"DVSGesture/DVSGesture-mlp-test-step={n_step}.npz")
    test_path = os.path.abspath(test_path)
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise ValueError(
            f'Cache files {train_path} and {test_path} do not exist. '
            f'please run "dvs-gesture-preprocessing.py" first.'
        )
    else:
        print(f'Used cache files {train_path} and {test_path}.')
    train_set = FormattedDataset(train_path)
    test_set = FormattedDataset(test_path)

    if args.use_augm:
        pass

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=_numpy_collate,
        num_workers=args.num_workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_set,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=_numpy_collate,
        num_workers=args.num_workers
    )

    return brainstate.util.DotDict(
        {
            'train_loader': train_loader,
            'test_loader': test_loader,
            'in_shape': in_shape,
            'out_shape': out_shape,
            'input_process': lambda x: x,
        }
    )


class FormattedDVSGestureV2:
    def __init__(self, filepath: str, aug: bool = False):
        self.filepath = filepath
        data = np.load(filepath, allow_pickle=True)
        self.xs_row = data['xs_row']
        self.xs_col = data['xs_col']
        self.xs_data = data['xs_data']
        self.ys = data['ys']
        img_size = data['img_size']
        self.img_size = (int(img_size[0]), int(np.prod(img_size[1:])))
        self.aug = aug
        self.transform = tonic.transforms.Compose(
            [
                torch.tensor,
                RandomResizedCrop(
                    tonic.datasets.DVSGesture.sensor_size[:-1],
                    scale=(0.6, 1.0),
                    interpolation=InterpolationMode.NEAREST
                ),
                RandomPerspective(),
                RandomRotation(25),
            ]
        )

    def load(self, idx):
        arr = np.zeros(self.img_size, dtype=np.float32)
        row = self.xs_row[idx]
        col = self.xs_col[idx]
        arr[row, col] = self.xs_data[idx]
        if self.aug:
            arr = arr.reshape(arr.shape[0], *tonic.datasets.DVSGesture.sensor_size)
            arr = np.transpose(arr, axes=(0, 3, 1, 2))
            arr = self.transform(arr)
            return np.asarray(arr).reshape([arr.shape[0], -1])
        else:
            return arr

    def __getitem__(self, idx):
        arr = self.load(idx)
        y = self.ys[idx]
        return arr, y

    def __len__(self):
        return len(self.ys)


def load_gesture_data_v2(args):
    in_shape = int(np.prod((128, 128, 2)))
    out_shape = 11
    n_step = args.data_length
    cache_dir = args.data_folder
    train_path = os.path.join(cache_dir, f"DVSGesture/DVSGesture-V2-train-step={n_step}.npz")
    train_path = os.path.abspath(train_path)
    test_path = os.path.join(cache_dir, f"DVSGesture/DVSGesture-V2-test-step={n_step}.npz")
    test_path = os.path.abspath(test_path)
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise ValueError(
            f'Cache files {train_path} and {test_path} do not exist. '
            f'please run "dvs-gesture-preprocessing.py" first.'
        )
    else:
        print(f'Used cache files {train_path} and {test_path}.')
    train_set = FormattedDVSGestureV2(train_path, aug=args.use_augm)
    test_set = FormattedDVSGestureV2(test_path, aug=False)

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=_numpy_collate,
        num_workers=args.num_workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_set,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=_numpy_collate,
        num_workers=args.num_workers,
        drop_last=False,
    )

    return brainstate.util.DotDict(
        {
            'train_loader': train_loader,
            'test_loader': test_loader,
            'in_shape': in_shape,
            'out_shape': out_shape,
            'input_process': lambda x: x,
        }
    )


def load_dataset(args):
    if args.dataset_name == 'nmnist':
        return load_nmnist_data(args)
    elif args.dataset_name == 'nmnistv2':
        return load_nmnist_data_v2(args)
    elif args.dataset_name == 'shd':
        return load_shd_data(args)
    elif args.dataset_name == 'ssc':
        return load_ssc_data(args)
    elif args.dataset_name == 'gesture':
        return load_gesture_data(args)
    elif args.dataset_name == 'gesturev2':
        return load_gesture_data_v2(args)
    else:
        raise ValueError(f'Unknown dataset name: {args.dataset_name}')
