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

import argparse
import json
import os
import platform

# 吴思Lab brainpy-tower1-brainpy
if platform.platform() == 'Linux-6.8.0-48-generic-x86_64-with-glibc2.35':
    data_dir = '/home/brainpy/codes/projects/braintrace_event_rnn/event_gru/dvs_gesture/data'
    cache_dir = '/home/brainpy/codes/projects/braintrace_event_rnn/event_gru/dvs_gesture/cache'

# 吴思Lab A100
elif platform.platform() == 'Linux-6.8.0-52-generic-x86_64-with-glibc2.35':
    data_dir = '/home/chaomingwang/code/githubs/EvNN/data'
    cache_dir = '/home/chaomingwang/code/githubs/EvNN/cache'

# 横琴A100
elif platform.platform() == 'Linux-5.15.0-84-generic-x86_64-with-glibc2.31':
    data_dir = '/home/chaoming/projects/evnn/benchmarks/data'
    cache_dir = '/home/chaoming/projects/evnn/benchmarks/cache'


# 雷神WSL
elif platform.platform() == 'Linux-5.15.167.4-microsoft-standard-WSL2-x86_64-with-glibc2.35':
    data_dir = '/mnt/d/codes/githubs/SNN/EvNN/data'
    cache_dir = '/mnt/d/codes/githubs/SNN/EvNN/cache'

# 雷神windows
elif platform.platform() == 'Windows-10-10.0.26100-SP0':
    data_dir = 'D:/codes/githubs/SNN/EvNN/data/'
    cache_dir = 'D:/codes/githubs/SNN/EvNN/cache/'

else:
    data_dir = './data'
    cache_dir = './cache'


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
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.99'


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--seed', type=int, default=None)
    argparser.add_argument('--devices', type=str, default='0', help='The GPU device ids.')
    argparser.add_argument('--data', type=str, default=data_dir, help='path to datasets')
    argparser.add_argument('--cache', type=str, default=cache_dir, help='path to temp cache')
    argparser.add_argument('--logdir', type=str, default='./logs/', help='scratch directory for jobs')
    argparser.add_argument('--resume-path', type=str, required=False, help='Resume training')
    argparser.add_argument('--method', type=str, required=False, default='bptt', choices=['bptt', 'es-d-rtrl', 'd-rtrl'])
    argparser.add_argument('--vjp-method', type=str, required=False, default='multi-step', choices=['multi-step', 'single-step'])
    argparser.add_argument('--etrace-decay', type=float, default=0.9)
    argparser.add_argument('--log-interval', type=int, default=50)
    argparser.add_argument('--batch-size', type=int, default=256)
    argparser.add_argument('--frame-size', type=int, default=128)
    argparser.add_argument('--warmup-ratio', type=float, default=0.)
    argparser.add_argument('--frame-time', type=int, default=25, help='Time in ms to collect events into each frame')
    argparser.add_argument('--event-agg-method', type=str, default='bool', choices=['mean', 'none'])
    argparser.add_argument('--use-cnn', action='store_true')
    argparser.add_argument('--augment-data', action='store_true')
    argparser.add_argument('--learning-rate', type=float, default=0.001)
    argparser.add_argument('--lr-gamma', type=float, default=0.8)
    argparser.add_argument('--lr-decay-epochs', type=int, default=100)
    argparser.add_argument('--use-rmsprop', action='store_true')
    argparser.add_argument('--use-grad-clipping', action='store_true')
    argparser.add_argument('--grad-clip-norm', type=float, default=2.0)
    argparser.add_argument('--rnn-type', type=str, default='lstm')
    argparser.add_argument('--units', type=int, default=256)
    argparser.add_argument('--num-layers', type=int, default=1)
    argparser.add_argument('--train-epochs', type=int, default=100)
    argparser.add_argument('--dropout', type=float, default=0.0)
    argparser.add_argument('--zoneout', type=float, default=0.0)
    argparser.add_argument('--pseudo-derivative-width', type=float, default=1.0)
    argparser.add_argument('--threshold-mean', type=float, default=0.0)
    args = argparser.parse_args()
    _set_gpu_device(args.devices)
    return args


def store_args(file, args):
    with open(file, 'w') as f:
        json.dump(args.__dict__, f, indent=2)


def load_args(file):
    parser = argparse.ArgumentParser()
    args = parser.parse_args('')
    with open(file, 'r') as f:
        args.__dict__ = json.load(f)

    return args
