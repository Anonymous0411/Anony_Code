import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))  # tools目录
parent_dir = os.path.dirname(current_dir)  # 项目根目录
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import json
from util.utils import *
import pickle
import numpy as np
from typing import List, Tuple


with open(f'CASE_HUB/patient_condiction_trainset_embedding.pkl', 'rb') as file:
    patient_condiction_embedding = pickle.load(file)

keys = list(patient_condiction_embedding.keys())
matrix = np.stack([patient_condiction_embedding[k] for k in keys])  # [N, 2046]

def case_retrieve(
    query_vec: np.ndarray,
    top_k: int = 5
) -> List[str]:

    # 归一化
    query_vec = query_vec / np.linalg.norm(query_vec)
    matrix_ = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)

    # 余弦相似度
    sims = matrix_ @ query_vec  # [N]

    topk_idx = np.argsort(sims)[-top_k:][::-1]
    return [keys[i] for i in topk_idx]


## static_testing 生成
static_testing = {}

# split_data_error = split_list_into_parts(list(patient_condiction_embedding.items()), 10)
# for key, value in tqdm(split_data_error[args.process-1], desc="Case"):
for key, value in tqdm(patient_condiction_embedding.items()):
    static_testing[key] = case_retrieve(value, 6)

with open(f'patient_condiction_embedding.json', 'w', encoding='utf-8') as file:
    json.dump(static_testing, file, indent=4, ensure_ascii=False)



