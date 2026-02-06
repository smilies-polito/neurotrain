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

import numpy as np
import tonic
from scipy.sparse import coo_matrix
from tonic import SlicedDataset
from tonic.collation import PadTensors
from tonic.datasets import DVSGesture, NMNIST
from tonic.slicers import SliceByTime
from torch.utils.data import DataLoader
from tqdm import tqdm


def _saving_sparse_dataset(
    test_loader,
    train_loader,
    train_filename,
    test_filename,
):
    for loader, filename in [
        (train_loader, train_filename),
        (test_loader, test_filename)
    ]:
        xs_row, xs_col, xs_data, ys_all = [], [], [], []
        bar = tqdm(total=len(loader), desc='preprocessing')
        for i, (xs, ys) in enumerate(loader):
            sparsity = []
            for x in xs:
                x = np.asarray(x)
                coo = coo_matrix(np.reshape(x, (x.shape[0], -1)))
                sparsity.append(coo.nnz / np.prod(coo.shape))
                xs_row.append(coo.row)
                xs_col.append(coo.col)
                xs_data.append(coo.data)
            ys_all.append(np.asarray(ys))
            bar.update(1)
            bar.set_description('sparsity: %.4f' % np.mean(sparsity), refresh=True)
        bar.close()

        img_size = xs.shape[1:]
        xs_row = np.asarray(xs_row, dtype=object)
        xs_col = np.asarray(xs_col, dtype=object)
        xs_data = np.asarray(xs_data, dtype=object)
        ys_all = np.concatenate(ys_all)
        np.savez(
            filename,
            xs_row=xs_row,
            xs_col=xs_col,
            xs_data=xs_data,
            ys=ys_all,
            img_size=np.asarray(img_size, dtype=np.int32)
        )


def dvs_gesture_preprocessing(
    num_workers: int,
    n_step: int = 100,
    cache_dir=os.path.expanduser("data/")
):
    train_filename = os.path.join(cache_dir, f'DVSGesture/DVSGesture-mlp-train-step={n_step}.npz')
    test_filename = os.path.join(cache_dir, f'DVSGesture/DVSGesture-mlp-test-step={n_step}.npz')
    if os.path.exists(train_filename) and os.path.exists(test_filename):
        return

    batch_size = 128
    in_shape = DVSGesture.sensor_size
    transform = tonic.transforms.ToFrame(sensor_size=in_shape, n_time_bins=n_step)
    train_set = DVSGesture(save_to=cache_dir, train=True, transform=transform)
    test_set = DVSGesture(save_to=cache_dir, train=False, transform=transform)
    train_loader = DataLoader(
        train_set,
        shuffle=False,
        batch_size=batch_size,
        collate_fn=PadTensors(batch_first=True),
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set,
        shuffle=False,
        batch_size=batch_size,
        collate_fn=PadTensors(batch_first=True),
        num_workers=num_workers
    )

    _saving_sparse_dataset(test_loader, train_loader, train_filename, test_filename)


def dvs_gesture_preprocessing_v2(
    num_workers: int,
    n_step: int = 100,
    cache_dir=os.path.expanduser("data/")
):
    train_filename = os.path.join(cache_dir, f'DVSGesture/DVSGesture-V2-train-step={n_step}.npz')
    test_filename = os.path.join(cache_dir, f'DVSGesture/DVSGesture-V2-test-step={n_step}.npz')
    if os.path.exists(train_filename) and os.path.exists(test_filename):
        return

    frame_time = 25  # ms
    time_window = 1.7 * 1e6  # 1.7 s
    overlap = 0
    sensor_size = tonic.datasets.DVSGesture.sensor_size
    frame_time = frame_time * 1000

    train_set = tonic.datasets.DVSGesture(save_to=cache_dir, train=True)
    test_set = tonic.datasets.DVSGesture(save_to=cache_dir, train=False)
    transform = tonic.transforms.ToFrame(
        sensor_size=sensor_size,
        time_window=frame_time,
        include_incomplete=False
    )

    batch_size = 128
    slicer_by_time = SliceByTime(time_window=time_window, overlap=overlap, include_incomplete=False)
    train_data = SlicedDataset(train_set, slicer=slicer_by_time, transform=transform)
    test_data = SlicedDataset(test_set, slicer=slicer_by_time, transform=transform)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=tonic.collation.PadTensors(batch_first=True),
        drop_last=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        collate_fn=tonic.collation.PadTensors(batch_first=True),
        drop_last=False,
    )

    _saving_sparse_dataset(test_loader, train_loader, train_filename, test_filename)


def nmnist_preprocessing(
    num_workers: int,
    n_step: int = 100,
    cache_dir=os.path.expanduser("data/")
):
    train_filename = os.path.join(cache_dir, f'NMNIST/NMNIST-train-step={n_step}.npz')
    test_filename = os.path.join(cache_dir, f'NMNIST/NMNIST-test-step={n_step}.npz')
    if os.path.exists(train_filename) and os.path.exists(test_filename):
        return

    batch_size = 128
    in_shape = NMNIST.sensor_size
    transform = tonic.transforms.ToFrame(sensor_size=in_shape, n_time_bins=n_step)
    train_set = NMNIST(save_to=cache_dir, train=True, transform=transform)
    test_set = NMNIST(save_to=cache_dir, train=False, transform=transform)
    train_loader = DataLoader(
        train_set,
        shuffle=False,
        batch_size=batch_size,
        collate_fn=PadTensors(batch_first=True),
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set,
        shuffle=False,
        batch_size=batch_size,
        collate_fn=PadTensors(batch_first=True),
        num_workers=num_workers
    )
    _saving_sparse_dataset(test_loader, train_loader, train_filename, test_filename)


if __name__ == '__main__':
    # for n_seq in [50, 100, 200, 300, 400, 600, 800, 1000]:
    #     print(f'Processing the data with length of {n_seq}')
    #     _dvs_gesture_preprocessing(0, n_seq)

    # _nmnist_preprocessing(0, 200)
    dvs_gesture_preprocessing_v2(0, 200)
    # nmnist_preprocessing(0, 100)
