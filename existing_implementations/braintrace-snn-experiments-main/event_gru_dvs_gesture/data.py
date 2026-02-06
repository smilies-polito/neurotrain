# Copyright (c) 2023  Khaleelulla Khan Nazeer, Anand Subramoney, Mark Schöne, David Kappel
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
import platform

import tonic
import torch
import torchvision
from tonic import DiskCachedDataset, SlicedDataset
from tonic.slicers import SliceByTime
from torch.utils.data import DataLoader
from torchvision.transforms import RandomPerspective, RandomResizedCrop, RandomRotation

num_worker = 0 if platform.system() == 'Windows' else 10


def get_dvs128_train_val(
    args,
    split: float = 1.0,
    augmentation: bool = False,
):
    """
    Make dataloaders for train and validation sets
    """
    transform, tr_str = get_transforms(args)

    dataset = tonic.datasets.DVSGesture(
        save_to=os.path.join(args.data, 'train'),
        train=True,
        transform=None,
        target_transform=None
    )

    train_size = int(split * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    min_time_window = 1.7 * 1e6  # 1.7 s
    overlap = 0
    metadata_path = f'_{min_time_window}_{overlap}_{args.frame_time}_' + tr_str
    slicer_by_time = SliceByTime(
        time_window=min_time_window,
        overlap=overlap,
        include_incomplete=False
    )
    train_dataset_timesliced = SlicedDataset(
        train_set,
        slicer=slicer_by_time,
        transform=transform,
        metadata_path=None
    )
    val_dataset_timesliced = SlicedDataset(
        val_set,
        slicer=slicer_by_time,
        transform=transform,
        metadata_path=None
    )

    if args.event_agg_method == 'none' or args.event_agg_method == 'mean':
        data_max = 19.0  # commented to save time, re calculate if min_time_window changes
        print(f'Max train value: {data_max}')
        norm_transform = lambda x: x / data_max
    else:
        norm_transform = None

    if augmentation:
        post_cache_transform = tonic.transforms.Compose(
            [
                norm_transform,
                torch.tensor,
                RandomResizedCrop(
                    tonic.datasets.DVSGesture.sensor_size[:-1],
                    scale=(0.6, 1.0),
                    interpolation=torchvision.transforms.InterpolationMode.NEAREST
                ),
                RandomPerspective(),
                RandomRotation(25),
            ]
        )
    else:
        post_cache_transform = norm_transform

    train_cached_dataset = DiskCachedDataset(
        train_dataset_timesliced,
        transform=post_cache_transform,
        cache_path=os.path.join(args.cache, 'diskcache_train' + metadata_path)
    )

    val_cached_dataset = DiskCachedDataset(
        val_dataset_timesliced,
        transform=post_cache_transform,
        cache_path=os.path.join(args.cache, 'diskcache_val' + metadata_path)
    )

    train_dataset = DataLoader(
        train_cached_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=tonic.collation.PadTensors(batch_first=False),
        drop_last=True,
        num_workers=num_worker,
    )
    val_dataset = DataLoader(
        val_cached_dataset,
        batch_size=args.batch_size,
        collate_fn=tonic.collation.PadTensors(batch_first=False),
        drop_last=True,
    )

    print(f"Loaded train dataset with {len(train_dataset.dataset)} samples")
    print(f"Loaded test dataset with {len(val_dataset.dataset)} samples")

    return train_dataset, val_dataset


def get_dvs128_test_dataset(args):
    """ Make dataloaders for test set
    """
    transform, tr_str = get_transforms(args)

    test_dataset = tonic.datasets.DVSGesture(
        save_to=os.path.join(args.data, 'test'),
        train=False,
        transform=None,
        target_transform=None
    )

    min_time_window = 1.7 * 1e6  # 1.7 s
    overlap = 0  #
    slicer_by_time = SliceByTime(
        time_window=min_time_window,
        overlap=overlap,
        include_incomplete=False
    )
    # os.makedirs(os.path.join(opt.cache, 'test'), exist_ok=True)
    metadata_path = f'_{min_time_window}_{overlap}_{args.frame_time}_' + tr_str
    test_dataset_timesliced = SlicedDataset(
        test_dataset,
        slicer=slicer_by_time,
        transform=transform,
        metadata_path=None
    )

    if args.event_agg_method == 'none' or args.event_agg_method == 'mean':
        data_max = 19.5  # commented to save time, re calculate if min_time_window changes
        print(f'Max test value: {data_max}')
        norm_transform = torchvision.transforms.Lambda(lambda x: x / data_max)
    else:
        norm_transform = None

    cached_test_dataset_time = DiskCachedDataset(
        test_dataset_timesliced,
        transform=norm_transform,
        cache_path=os.path.join(args.cache, 'diskcache_test' + metadata_path)
    )
    cached_test_dataloader_time = DataLoader(
        cached_test_dataset_time,
        batch_size=args.batch_size,
        collate_fn=tonic.collation.PadTensors(batch_first=False),
        drop_last=False,
        num_workers=num_worker,
    )

    print(f"Loaded test dataset with {len(test_dataset)} samples")
    print(f"Loaded sliced test dataset with {len(cached_test_dataset_time)} samples")

    return cached_test_dataloader_time


def get_transforms(args):
    sensor_size = tonic.datasets.DVSGesture.sensor_size
    frame_transform_time = tonic.transforms.ToFrame(
        sensor_size=sensor_size,
        time_window=args.frame_time * 1000,
        include_incomplete=False
    )
    return frame_transform_time, 'toframe'

    denoise_transform = tonic.transforms.Denoise(filter_time=10000)
    transform = tonic.transforms.Compose(
        [
            # denoise_transform,
            frame_transform_time,
        ]
    )
    return transform, 'toframe'


if __name__ == '__main__':
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data', type=str, default='../data', help='path to datasets')
    argparser.add_argument('--cache', type=str, default='../cache', help='path to temp cache')
    argparser.add_argument('--frame-time', type=int, default=25, help='Time in ms to collect events into each frame')
    argparser.add_argument('--event-agg-method', type=str, default='bool', choices=['mean', 'diff', 'bool', 'none'])
    argparser.add_argument('--batch-size', type=int, default=256)
    args = argparser.parse_args()

    train_data, val_data = get_dvs128_train_val(args)
    test_data = get_dvs128_test_dataset(args)

    print('Train dataset:')
    for batch in train_data:
        print('input shape = ', batch[0].shape, 'target shape = ', batch[1].shape)
    print()

    print('Test dataset:')
    for batch in test_data:
        print('input shape = ', batch[0].shape, 'target shape = ', batch[1].shape)
