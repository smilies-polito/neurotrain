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


import os
import re
from collections import defaultdict
import argparse
import numpy as np

# Dictionary to store results: {model: [acc1, acc2, ...]}
results = dict()

# Root directory containing all model folders
root_dir = "bptt_shd/"
root_dir = "bptt-bn/bptt_shd/"
# root_dir = "bptt-none/bptt_shd/"
# root_dir = "bptt-none/bptt_shd/"
# root_dir = "esd-rtrl_0_9_shd/"
root_dir = "esd-rtrl_0_88_shd/"

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str, default=root_dir, help='Root directory containing model logs')
args = parser.parse_args()

# Pattern to extract validation accuracy
acc_pattern = re.compile(r"Best valid acc at epoch \d+: (\d+\.\d+)")

models = ['LIF', 'adLIF', 'RadLIF', 'RLIF']
learning_rates = ['0.02', '0.01', '0.005', '0.001']

# Walk through the d-rtrl-shd-search directory
for model in models:
    results[model] = defaultdict(list)
    for lr in learning_rates:
        model_dir_name = f'{model}_3lay1024_drop0_5_none_nobias_lr{lr.replace(".", "_")}'
        model_dir_name = f'{model}_3lay1024_drop0_1_none_nobias_lr{lr.replace(".", "_")}'
        # model_dir_name = f'{model}_3lay1024_drop0_1_batchnorm_nobias_lr{lr.replace(".", "_")}'
        model_dir = os.path.join(args.root_dir, model_dir_name)

        if os.path.exists(model_dir):
            for run_dir in os.listdir(model_dir):
                log_file = os.path.join(model_dir, run_dir, "exp.log")
                if os.path.exists(log_file):
                    try:
                        with open(log_file, "r") as f:
                            acc_match = acc_pattern.search(f.read())
                            if acc_match:
                                acc_value = float(acc_match.group(1))
                                results[model][lr].append(acc_value)
                    except Exception as e:
                        print(f"Error reading {log_file}: {e}")
        else:
            print(f"Log file {model_dir} not found")

# Print statistics for each model
print("=" * 80)
for model in models:
    print(f"\nStatistics for {model}:")
    print("-" * 30)
    for lr in learning_rates:
        accuracies = results[model].get(lr, [])
        if accuracies:
            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            max_acc = np.max(accuracies)
            print(f"Model={model}, lr={lr}: mean={mean_acc:.4f}, std={std_acc:.4f}, max={max_acc:.4f},\n\t values={accuracies}")
        else:
            print(f"Model={model}, lr={lr}: No results found")
