import sys
sys.path.append("/data1/huangguolin/workplace2/graph_agent/graph_agent_v5/")
import json
import random
import re
from sklearn.metrics import accuracy_score, f1_score
from util.utils import *
import ast


# # policy 阶段 ------------------------------------------------------------------------------------------------------------------------------------------------------
# with open('prompt/generation_DAG/task1/planing.txt', 'r', encoding='utf-8') as file:
#     planing_ori = file.read()

# # with open('prompt/generation_DAG/planing_all_task.txt', 'r', encoding='utf-8') as file:
# #     planing_ori = file.read()

# with open('data/MedChain/task1/test_set_gt.json', 'r', encoding='utf-8') as file:
#     test_data = json.load(file)

# with open('data/MedChain/patient_condiction.json', 'r', encoding='utf-8') as file:
#     patient_condiction = json.load(file)

# inference_policy = {}
# split_data = split_list_into_parts(list(test_data.items()), 20)
# for key, value in tqdm(split_data[args.process-1], desc="Case"):
# # for key, value in test_data.items():
#     planing_prompt = planing_ori.replace('{description}', patient_condiction[key])
#     # print(planing_prompt)
#     policy_str = get_llama_api(max_token=10000, temperature=0, system_role='', user_input=planing_prompt)
#     policy_ = policy_str.split('</think>')[-1]

#     try:
#         policy = ast.literal_eval(policy_)
#     except:
#         # inference_policy[key] = 'error'
#         continue

#     inference_policy[key] = policy
#     # print(policy)
#     # print('---'*10)

# # with open(f'nips2026_rebuttal/evopatient_eveagent/task1_1/policy_test_{args.process}.json', 'w', encoding='utf-8') as file:
# #     json.dump(inference_policy, file, indent=4, ensure_ascii=False)

# ## 合并策略 
# policy_all = {}
# for num in range(1, 21):
#     with open(f'ablation_exp/grpo_policy_exp/policy_test_{num}.json', 'r', encoding='utf-8') as file:
#         policy_part = json.load(file)
#     for key, value in policy_part.items():
#         policy_all[key] = value

# with open(f'ablation_exp/grpo_policy_exp/task1_policy.json', 'w', encoding='utf-8') as file:
#     json.dump(policy_all, file, indent=4, ensure_ascii=False)

# print(len(policy_all))

# ## 加入 tool 节点 与 节点关系
# with open(f'unified_model/policy.json', 'r', encoding='utf-8') as file:
#     policy = json.load(file)
# policy_rebuild = {}
# for key, value in policy.items():
#     single_rebuild_node = []
#     node_num_id = 1
#     search_node_id = []
#     solve_node_id = []
#     edges = []
#     for single_node_id, single_node in value.items():
#         single_rebuild_node.append(single_node)
#         if single_node['task_type'] == 'search':
#             search_node_id.append(node_num_id)
#         if single_node['task_type'] == 'solve':
#             solve_node_id.append(node_num_id)
#         node_num_id += 1
#     single_rebuild_node.append({
#         "node_id": f"n{node_num_id}",
#         "task_type": "tool",
#         "task_desc": "检索相关病例，查看相似病例的分诊情况",
#         "question": "无",
#         "tools": "病例检索"
#     })

#     if len(solve_node_id) != 0:
#         for single_search_node in search_node_id:
#             edges.append({"from_node": f'n{single_search_node}', "to_node": f'n{solve_node_id[0]}'})

#     if len(solve_node_id) > 1:
#         for single_solve_node_id, single_solve_node in enumerate(solve_node_id[:-1]):
#             edges.append({"from_node": f'n{single_solve_node}', "to_node": f'n{solve_node_id[single_solve_node_id+1]}'})

#     policy_rebuild[key] = {'nodes': single_rebuild_node, 'edges': edges}

# with open(f'unified_model/policy_rebuild_test.json', 'w', encoding='utf-8') as file:
#     json.dump(policy_rebuild, file, indent=4, ensure_ascii=False)

# print(len(policy_rebuild))

# ## 执行策略
# from tools.execute_DAG_task1_1_policy import *


# with open(f'unified_model/policy_rebuild_test.json', 'r', encoding='utf-8') as file:
#     policy_all = json.load(file)

# inference_prompt_all = {}

# split_data = split_list_into_parts(list(policy_all.items()), 20)
# for key, value in tqdm(split_data[args.process-1], desc="Case"):
#     try:
#         inference_prompt = execute_DAG_task1_1_policy(key, value, train_executor=True, inference_con=False, node_weight=False)
#         # print(inference_prompt)
#         # print(inference_prompt)
#     except Exception as e:
#         print(e)
#         continue
#     inference_prompt_all[key] = inference_prompt

# with open(f'nips2026_rebuttal/multi_turn/turn3/task1/inference_prompt_{args.process}.json', 'w', encoding='utf-8') as file:
#     json.dump(inference_prompt_all, file, indent=4, ensure_ascii=False)


# ## 获取病人意见
# with open(f'inference_process_data/task1_level1/policy/result.json', 'r', encoding='utf-8') as file:
#     policy_all = json.load(file)

# with open('data/MedChain/patient_condiction.json', 'r', encoding='utf-8') as file:
#     patient_condiction = json.load(file)

# with open(f'prompt/patient_opinion.txt', 'r', encoding='utf-8') as file:
#     patient_opinion_ori = file.read()

# with open(f'prompt/executor/task1/repair.txt', 'r', encoding='utf-8') as file:
#     repair_ori = file.read()

# split_data = split_list_into_parts(list(policy_all.items()), 20)
# for key, value in tqdm(split_data[args.process-1], desc="Case"):
#     patient_opinion_prompt = patient_opinion_ori.replace('{description}', patient_condiction[key]).replace('{report}', str(value['report']))
#     # print(patient_opinion_prompt)
#     response = get_llama_api(max_token=10000, temperature=0, system_role='', user_input=patient_opinion_prompt)
#     # print(response)
#     repair_prompt = repair_ori.replace("{report}", str(value['report'])).replace("{opinion}", response)
#     response = get_llama_api(max_token=10000, temperature=0, system_role='', user_input=repair_prompt)
#     # print(response)
#     # print('------------------------------')

# ## 合并结果
# result_all = {}
# for num in range(1, 21):
#     with open(f'nips2026_rebuttal/backbone/llama/task1/result_l1_{num}.json', 'r', encoding='utf-8') as file:
#         result_part = json.load(file)
#     for key, value in result_part.items():
#         result_all[key] = value

# with open(f'nips2026_rebuttal/backbone/llama/task1/result_l1.json', 'w', encoding='utf-8') as file:
#     json.dump(result_all, file, indent=4, ensure_ascii=False)

# print(len(result_all))

# ## 推理
# with open(f'unified_model/task1_1/inference_prompt_test.json', 'r', encoding='utf-8') as file:
#     inference_prompt = json.load(file)

# result = {}

# split_data = split_list_into_parts(list(inference_prompt.items()), 20)
# for key, value in tqdm(split_data[args.process-1], desc="Case"):
# # for key, value in tqdm(inference_prompt.items()):
#     response = get_llama_api(max_token=10000, temperature=0, system_role='', user_input=value)
#     # response = gpt_55(max_token=10000, temperature=0, system_role='', user_input=value)
#     result[key] = response

# with open(f'nips2026_rebuttal/backbone/llama/task1/result_l1_{args.process}.json', 'w', encoding='utf-8') as file:
#     json.dump(result, file, indent=4, ensure_ascii=False)

# with open(f'nips2026_rebuttal/evopatient_eveagent/task1_1/result.json', 'w', encoding='utf-8') as file:
#     json.dump(result, file, indent=4, ensure_ascii=False)













