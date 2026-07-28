import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))  # tools目录
parent_dir = os.path.dirname(current_dir)  # 项目根目录
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import json
from util.utils import *
import pickle


with open('MedChain_data/patient_condiction.json', 'r', encoding='utf-8') as file:
    patient_condiction = json.load(file)


# ## 构建检索embedding向量库
# embedding_hub = {}

# split_data_error = split_list_into_parts(list(patient_condiction.items()), 20)
# for key, value in tqdm(split_data_error[args.process-1], desc="Case"):
# # for key, value in tqdm(patient_condiction.items()):
#     embedding_hub[key] = llm_embedding(value)

# with open(f'MedChain_data/patient_condiction_embedding_{args.process}.pkl', 'wb') as f:
#     pickle.dump(embedding_hub, f)


# ## 合并
# patient_condiction_embedding_all = {}

# for num in range(1, 21):
#     with open(f'MedChain_data/patient_condiction_embedding_{num}.pkl', 'rb') as file:
#         patient_condiction_embedding_part = pickle.load(file)
#     for key, value in patient_condiction_embedding_part.items():
#         patient_condiction_embedding_all[key] = value

# with open(f'MedChain_data/patient_condiction_embedding.pkl', 'wb') as f:
#     pickle.dump(patient_condiction_embedding_all, f)

# print(len(patient_condiction_embedding_all))




