import sys
sys.path.append("/data1/huangguolin/workplace2/graph_agent/graph_agent_v5/")
import json
import random
import re
from util.utils import *


# # # policy 阶段 ------------------------------------------------------------------------------------------------------------------------------------------------------
# from tools.execute_DAG_batch_task2_test import *   ## test_data
# # # from tools.execute_DAG_batch import *   ## train_data
# # random.seed(2025)


# with open('inference_process_data/task1_level2/policy/train_data/task1_level2_policy_test_prompt.json', 'r', encoding='utf-8') as file:
#     test_data = json.load(file)

# inference_policy = {}
# split_data = split_list_into_parts(list(test_data.items()), 20)
# for key, value in tqdm(split_data[args.process-1], desc="Case"):
# # for key, value in test_data.items():
#     # print(value)
#     policy_str = get_llama_api(max_token=10000, temperature=0, system_role='', user_input=value)

#     policy_ = policy_str.split('</think>')[-1]
#     try:
#         policy = json.loads(policy_)
#     except:
#         inference_policy[key] = 'error'
#         continue

#     inference_policy[key] = policy

# with open(f'inference_process_data/task1_level2/policy/train_data/policy_{args.process}.json', 'w', encoding='utf-8') as file:
#     json.dump(inference_policy, file, indent=4, ensure_ascii=False)

# ## 合并策略 
# policy_all = {}
# for num in range(1, 21):
#     with open(f'nips2026_rebuttal/backbone/llama/task1/result_l2_{num}.json', 'r', encoding='utf-8') as file:
#         policy_part = json.load(file)
#     for key, value in policy_part.items():
#         policy_all[key] = value

# with open(f'nips2026_rebuttal/backbone/llama/task1/result_l2.json', 'w', encoding='utf-8') as file:
#     json.dump(policy_all, file, indent=4, ensure_ascii=False)

# print(len(policy_all))

# ## 执行策略
# from tools.execute_DAG_task1_2_policy import *


# with open(f'unified_model/policy_rebuild_test.json', 'r', encoding='utf-8') as file:
#     policy_all = json.load(file)

# # with open('inference_process_data/task1_level2/policy/test_data/doctor_patient_interaction_result.json', 'r', encoding='utf-8') as file:
# #     doctor_patient_interaction_result = json.load(file)

# # with open('inference_process_data/task1_level2/policy/test_data/doctor_reasoning_result.json', 'r', encoding='utf-8') as file:
# #     doctor_reasoning_result = json.load(file)

# with open('nips2026_rebuttal/CoEAgent_wo_DPA/simpatient/task1/second_room_list.json', 'r', encoding='utf-8') as file:
#     second_room_list_result = json.load(file)

# inference_result = {}

# split_data = split_list_into_parts(list(policy_all.items()), 20)
# for key, value in tqdm(split_data[args.process-1], desc="Case"):

#     try:
#         # history_dialogue = doctor_patient_interaction_result[key]
#         # history_diagnosis = doctor_reasoning_result[key]

#         history_dialogue = '无'
#         history_diagnosis = '无'

#         second_room_list = second_room_list_result[key]
#         inference_prompt = execute_DAG_task1_2_policy(patient_key=key, DAG_dict=value, history_dialogue=history_dialogue, history_diagnosis=history_diagnosis, second_room_list=second_room_list, train_executor=True, inference_con=False, node_weight=False)
#         # print(inference_prompt)
#         inference_result[key] = inference_prompt
#     except:
#         continue
#     # print(inference_prompt)
#     # print(inference_prompt)

# with open(f'nips2026_rebuttal/CoEAgent_wo_DPA/simpatient/task1/inference_prompt_{args.process}.json', 'w', encoding='utf-8') as file:
#     json.dump(inference_result, file, indent=4, ensure_ascii=False)

# ## 合并结果
# for file_name in ['interaction', 'think', 'tool']:
#     result_all = {}
#     for num in range(1, 21):
#         with open(f'nips2026_rebuttal/node_type_ablation/result/task1/result_l2_{file_name}_{num}.json', 'r', encoding='utf-8') as file:
#             result_part = json.load(file)
#         for key, value in result_part.items():
#             result_all[key] = value

#     with open(f'nips2026_rebuttal/node_type_ablation/result/task1/result_l2_{file_name}.json', 'w', encoding='utf-8') as file:
#         json.dump(result_all, file, indent=4, ensure_ascii=False)

#     print(len(result_all))


# ## 推理
# with open(f'unified_model/task1_2/inference_prompt_test.json', 'r', encoding='utf-8') as file:
#     inference_prompt = json.load(file)

# result = {}

# split_data = split_list_into_parts(list(inference_prompt.items()), 20)
# for key, value in tqdm(split_data[args.process-1], desc="Case"):
# # for key, value in tqdm(inference_prompt.items()):
#     response = get_llama_api(max_token=10000, temperature=0, system_role='', user_input=value)
#     # response = gpt_55(max_token=10000, temperature=0, system_role='', user_input=value)
#     result[key] = response

# # with open(f'ablation_exp/sft_policy_data/result_task1_2.json', 'w', encoding='utf-8') as file:
# #     json.dump(result, file, indent=4, ensure_ascii=False)

# with open(f'nips2026_rebuttal/backbone/llama/task1/result_l2_{args.process}.json', 'w', encoding='utf-8') as file:
#     json.dump(result, file, indent=4, ensure_ascii=False)



