import sys
sys.path.append("/data1/huangguolin/workplace2/graph_agent/graph_agent_v5/")
import json
import random
import re
from sklearn.metrics import accuracy_score, f1_score
from util.utils import *


def intersection_over_union(test, gt):
    if gt == []:
        if test == []:
            return 1
        else:
            return 0.8

    set1 = set(test)
    set2 = set(gt)

    intersection = set1.intersection(set2)
    union = set1.union(set2)

    iou = len(intersection) / len(union)
    return iou


# # policy 阶段 ------------------------------------------------------------------------------------------------------------------------------------------------------
# with open('ablation_exp/policy_sft/task2/policy_prompt_test.json', 'r', encoding='utf-8') as file:
#     test_data_prompt = json.load(file)

# policy_result = {}

# split_data = split_list_into_parts(list(test_data_prompt.items()), 5)
# for key, value in tqdm(split_data[args.process-1], desc="Case"):
# # for key, value in test_data_prompt.items():
#     response = get_llama_api(max_token=10000, temperature=0, system_role='', user_input=value).split('</think>')[-1]
    
#     try:
#         DAG = json.loads(response)
#     except Exception as e:
#         print("转化为图模型 - 异常")
#         continue

#     policy_result[key] = DAG

# with open(f'ablation_exp/grpo_policy_exp/policy_test_task2_{args.process}.json', 'w', encoding='utf-8') as file:
#     json.dump(policy_result, file, indent=4, ensure_ascii=False)

# # ------------------------------------------------------------------------------------------------------------------
# test_policy_all = {}
# for num in range(1, 6):
#     with open(f'ablation_exp/grpo_policy_exp/policy_test_task5_{num}.json', 'r', encoding='utf-8') as file:
#         test_policy_part = json.load(file)
#     for key, value in test_policy_part.items():
#         test_policy_all[key] = value

# with open(f'ablation_exp/grpo_policy_exp/policy_test_task5.json', 'w', encoding='utf-8') as file:
#     json.dump(test_policy_all, file, indent=4, ensure_ascii=False)

# print(len(test_policy_all))

# ## 执行 policy
# from tools.execute_DAG_task2 import *


# with open('unified_model/policy_rebuild_test.json', 'r', encoding='utf-8') as file:
#     policy_rebuild = json.load(file)

# inference_phy_prompt = {}
# inference_ass_prompt = {}

# split_data = split_list_into_parts(list(policy_rebuild.items()), 20)
# for key, value in tqdm(split_data[args.process-1], desc="Case"):

#     try:
#         inference_prompt_phy, inference_prompt_ass = execute_DAG_task2(patient_key=key, DAG_dict=value, train_executor=True, inference_con=False, node_weight=False)
#         inference_phy_prompt[key] = inference_prompt_phy
#         inference_ass_prompt[key] = inference_prompt_ass
#     except:
#         continue

#     # print(inference_prompt_phy)
#     # print('--------'*20)
#     # print(inference_prompt_ass)
#     # print(inference_prompt_ass)

# with open(f'nips2026_rebuttal/multi_turn/turn3/task2/inference_phy_prompt_test_{args.process}.json', 'w', encoding='utf-8') as file:
#     json.dump(inference_phy_prompt, file, indent=4, ensure_ascii=False)

# with open(f'nips2026_rebuttal/multi_turn/turn3/task2/inference_ass_prompt_test_{args.process}.json', 'w', encoding='utf-8') as file:
#     json.dump(inference_ass_prompt, file, indent=4, ensure_ascii=False)

# ## 合并
# for file_name in ['interaction', 'think', 'tool']:
#     inference_ass_prompt_all = {}
#     for num in range(1, 21):
#         with open(f'nips2026_rebuttal/node_type_ablation/result/task2/result_phy_wo_{file_name}_{num}.json', 'r', encoding='utf-8') as file:
#             inference_ass_prompt_part = json.load(file)
#         for key, value in inference_ass_prompt_part.items():
#             inference_ass_prompt_all[key] = value

#     with open(f'nips2026_rebuttal/node_type_ablation/result/task2/result_phy_wo_{file_name}.json', 'w', encoding='utf-8') as file:
#         json.dump(inference_ass_prompt_all, file, indent=4, ensure_ascii=False)

#     print(len(inference_ass_prompt_all))

# ## 推理
# for file_name in ['interaction', 'think', 'tool']:
#     with open(F'nips2026_rebuttal/node_type_ablation/task2_physical/inference_phy_prompt_test_wo_{file_name}.json', 'r',  encoding='utf-8') as file:
#         inference_prompt = json.load(file)

#     result = {}

#     split_data = split_list_into_parts(list(inference_prompt.items()), 20)
#     for key, value in tqdm(split_data[args.process-1], desc="Case"):
#     # for key, value in tqdm(inference_prompt.items()):
#         response = get_llama_api(max_token=10000, temperature=0, system_role='', user_input=value).split('</think>')[-1]
#         # response = gpt_55(max_token=10000, temperature=0, system_role='', user_input=value)
#         result[key] = response

#     # with open(F'ablation_exp/sft_policy_data/result_task_2_phy.json', 'w', encoding='utf-8') as file:
#     #     json.dump(result, file, indent=4, ensure_ascii=False)

#     with open(F'nips2026_rebuttal/node_type_ablation/result/task2/result_phy_wo_{file_name}_{args.process}.json', 'w', encoding='utf-8') as file:
#         json.dump(result, file, indent=4, ensure_ascii=False)


