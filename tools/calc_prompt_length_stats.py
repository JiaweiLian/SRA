import os
import json
import numpy as np

# 根目录
base_dir = 'results/SRA/transfer_attack'

# 获取所有模型文件夹
model_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

for model in sorted(model_dirs):
    log_path = os.path.join(base_dir, model, 'test_cases', 'logs.json')
    if not os.path.exists(log_path):
        print(f"{model}: logs.json not found.")
        continue
    with open(log_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    prompt_lengths = []
    for attack_list in data.values():
        for attack in attack_list:
            for item in attack:
                if 'prompt_length' in item:
                    prompt_lengths.append(item['prompt_length']+1)
    if prompt_lengths:
        mean = np.mean(prompt_lengths)
        std = np.std(prompt_lengths, ddof=1)
        print(f"{model}: mean={mean:.2f}, std={std:.2f}")
    else:
        print(f"{model}: No prompt_length found.")
