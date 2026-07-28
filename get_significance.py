import sys
import json
import random
import re
from sklearn.metrics import accuracy_score, f1_score
from util.utils import *
import numpy as np


def intersection_over_union(test, gt):
    if gt == []:
        if test == []:
            return 1
        else:
            return 0.5

    set1 = set(test)
    set2 = set(gt)

    intersection = set1.intersection(set2)
    union = set1.union(set2)

    iou = len(intersection) / len(union)
    return iou


def random_sample_values(dictionary, n):
    """
    从字典中随机抽取n个值
    
    Args:
        dictionary: 输入字典
        n: 想要抽取的数量
    
    Returns:
        list: 随机抽取的值列表
    """
    # 确保不超出字典长度
    sample_size = min(n, len(dictionary))
    
    if sample_size == 0:
        return []
    
    # 随机抽取
    sampled_items = random.sample(list(dictionary.items()), sample_size)
    
    # 只返回值
    return [item[1] for item in sampled_items]


# ## 执行策略
# from tools.execute_DAG_task5 import *

# inference_result = {}

# with open(f'random_policy/task5/policy.json', 'r', encoding='utf-8') as file:
#     policy_all = json.load(file)

# with open(f'inference_process_data/task4/policy/pre_data.json', 'r', encoding='utf-8') as file:
#     pre_data = json.load(file)

# policy_inference_prompt = {}

# split_data = split_list_into_parts(list(policy_all.items()), 20)
# for key, value in tqdm(split_data[args.process-1], desc="Case"):
#     policy_inference_prompt_single = {}
#     for single_node_id, single_node in value.items():

#         try:
#             inference_prompt = execute_DAG_task5(key, single_node, pre_data[key], train_executor=True, inference_con=False, node_weight=True)
#         except:
#             inference_prompt = 'policy_wrong'

#         policy_inference_prompt_single[single_node_id] = inference_prompt

#     policy_inference_prompt[key] = policy_inference_prompt_single

# with open(f'random_policy/task5/policy_inference_prompt_{args.process}.json', 'w', encoding='utf-8') as file:
#     json.dump(policy_inference_prompt, file, indent=4, ensure_ascii=False)


# ## 合并结果
# result_all = {}
# for num in range(1, 21):
#     with open(f'random_policy/task5/policy_inference_prompt_{num}.json', 'r', encoding='utf-8') as file:
#         result_part = json.load(file)
#     for key, value in result_part.items():
#         result_all[key] = value

# with open(f'random_policy/task5/policy_inference_prompt.json', 'w', encoding='utf-8') as file:
#     json.dump(result_all, file, indent=4, ensure_ascii=False)

# print(len(result_all))


# ## 执行推理
# with open('random_policy/task5/policy_inference_prompt.json', 'r', encoding='utf-8') as file:
#     policy_inference_prompt = json.load(file)

# new_result = {}

# split_data = split_list_into_parts(list(policy_inference_prompt.items()), 20)
# for key, value in tqdm(split_data[args.process-1], desc="Case"):
# # for key, value in tqdm(policy_inference_prompt.items()):
#     single_result = {}
#     for single_node_id, single_prompt in value.items():
#         response = get_llama_api(max_token=10000, temperature=0, system_role='', user_input=single_prompt).split('</think>')[-1]
#         single_result[single_node_id] = response
#     new_result[key] = single_result

# with open(f'random_policy/task5/result_{args.process}.json', 'w', encoding='utf-8') as file:
#     json.dump(new_result, file, indent=4, ensure_ascii=False)


# ## 合并
# result_all = {}
# for num in range(1, 21):
#     with open(f'random_policy/task4/node_weight_score_{num}.json', 'r', encoding='utf-8') as file:
#         result_part = json.load(file)
#     for key, value in result_part.items():
#         result_all[key] = value

# with open(f'random_policy/task4/node_weight_score.json', 'w', encoding='utf-8') as file:
#     json.dump(result_all, file, indent=4, ensure_ascii=False)

# print(len(result_all))

# ## 确定重要节点 task1
# from collections import Counter


# with open('data/MedChain/task1/train_set_gt.json', 'r', encoding='utf-8') as file:
#     MedChain_data_gt = json.load(file)

# with open(f'random_policy/task1_2/result.json', 'r', encoding='utf-8') as file:
#     result = json.load(file)

# node_weight_result = {}

# for key, value in result.items():
#     node_weight_result[key] = []
#     gt = MedChain_data_gt[key]['second']
#     pre_list = list(value.values())
#     counter_num = dict(Counter(value.values()))
#     if len(counter_num) == 1:
#         continue
    
#     if gt not in pre_list:
#         continue

#     ## 不删除是对的,删除了是错的
#     if gt in pre_list:
#         for single_node_id, single_node_value in value.items():
#             if single_node_value != gt:
#                 node_weight_result[key].append(single_node_id)

# with open(f'random_policy/task1_2/node_weight.json', 'w', encoding='utf-8') as file:
#     json.dump(node_weight_result, file, indent=4, ensure_ascii=False)


# ## 确定重要节点 task2
# from collections import Counter


# with open('data/MedChain/task2/train_set_gt.json', 'r', encoding='utf-8') as file:
#     MedChain_data_gt = json.load(file)

# with open(f'random_policy/task2/result_phy.json', 'r', encoding='utf-8') as file:
#     result_phy = json.load(file)

# with open(f'random_policy/task2/result_ass.json', 'r', encoding='utf-8') as file:
#     result_ass = json.load(file)

# node_weight_result_phy = {}
# node_weight_result_ass = {}

# node_weight_result_phy_score = {}
# node_weight_result_ass_score = {}

# for key, value in result_phy.items():

#     node_weight_result_phy[key] = []
#     node_weight_result_ass[key] = []

#     if MedChain_data_gt[key]['physical'] == '无':
#         physical_gt = []
#     else:
#         physical_gt = list(MedChain_data_gt[key]['physical'].keys())

#     if MedChain_data_gt[key]['auxiliary'] == '无':
#         auxiliary_gt = []
#     else:
#         auxiliary_gt = list(MedChain_data_gt[key]['auxiliary'].keys())

#     pre_phy_score = {}
#     pre_ass_score = {}

#     for single_node_id, single_node_value in result_phy[key].items():

#         physical_pre = []
#         physical_gt_list = ['皮肤检查', '泌尿生殖系统检查', '腹部检查', '头颅眼耳鼻喉检查', '胸部检查', '一般检查', '神经系统检查', '颈部检查', '脊柱和四肢检查']

#         for single_physical_item in physical_gt_list:
#             if single_physical_item in single_node_value:
#                 physical_pre.append(single_physical_item)

#         pre_phy_score[single_node_id] = intersection_over_union(physical_pre, physical_gt)


#     for single_node_id, single_node_value in result_ass[key].items():

#         auxiliary_pre = []
#         auxiliary_gt_list = ['CT', '超声', '内镜检查', '病理检查', '粪便检查', '血液学检查', 'X-ray', '尿液检查', '核医学成像', 'MRI']

#         for single_auxiliary_item in auxiliary_gt_list:
#             if single_auxiliary_item in single_node_value:
#                 auxiliary_pre.append(single_auxiliary_item)
                
#         pre_ass_score[single_node_id] = intersection_over_union(auxiliary_pre, auxiliary_gt)


#     node_weight_result_phy_score[key] = pre_phy_score
#     node_weight_result_ass_score[key] = pre_ass_score

#     ## 不删除是对的,删除了是错的
#     counter_num_phy = dict(Counter(pre_phy_score.values()))
#     counter_num_ass = dict(Counter(pre_ass_score.values()))


#     if len(counter_num_phy) != 1:
#         score_list = list(pre_phy_score.values())
#         max_score = max(score_list)
#         for single_node_id_, single_node_score in pre_phy_score.items():
#             if single_node_score < max_score:
#                 node_weight_result_phy[key].append(single_node_id_)

#     if len(counter_num_ass) != 1:
#         score_list = list(pre_ass_score.values())
#         max_score = max(score_list)
#         for single_node_id_, single_node_score in pre_ass_score.items():
#             if single_node_score < max_score:
#                 node_weight_result_ass[key].append(single_node_id_)


# with open(f'random_policy/task2/node_weight_phy.json', 'w', encoding='utf-8') as file:
#     json.dump(node_weight_result_phy, file, indent=4, ensure_ascii=False)

# with open(f'random_policy/task2/node_weight_ass.json', 'w', encoding='utf-8') as file:
#     json.dump(node_weight_result_ass, file, indent=4, ensure_ascii=False)

# with open(f'random_policy/task2/node_weight_phy_score.json', 'w', encoding='utf-8') as file:
#     json.dump(node_weight_result_phy_score, file, indent=4, ensure_ascii=False)

# with open(f'random_policy/task2/node_weight_ass_score.json', 'w', encoding='utf-8') as file:
#     json.dump(node_weight_result_ass_score, file, indent=4, ensure_ascii=False)


# ## 确定重要节点 task4
# from collections import Counter


# with open('data/MedChain/task4/train_set_gt.json', 'r', encoding='utf-8') as file:
#     MedChain_data_gt = json.load(file)

# with open(f'random_policy/task4/result.json', 'r', encoding='utf-8') as file:
#     result = json.load(file)

# with open('prompt/generation_DAG/task4/get_score.txt', 'r', encoding='utf-8') as file:
#     get_score = file.read()

# with open('train_set.json', 'r', encoding='utf-8') as file:
#     train_set = json.load(file)

# node_weight_result = {}

# node_weight_result_score = {}

# useful_data = {}

# ## 筛选无效数据
# for key, value in tqdm(result.items()):
#     counter_num = dict(Counter(value.values()))
#     if len(counter_num) == 1:
#         continue
#     useful_data[key] = value

# split_data = split_list_into_parts(list(useful_data.items()), 20)
# for key, value in tqdm(split_data[args.process-1], desc="Case"):
# # for key, value in tqdm(useful_data.items()):
#     node_weight_result[key] = []
#     node_score =  {}
#     for single_node_id, single_node_value in value.items():
#         diagnosis_gt = MedChain_data_gt[key]
#         get_score_prompt = get_score.replace("{report}", str(train_set[key])).replace("{gt_result}", str(diagnosis_gt)).replace("{model_result}", single_node_value)
#         # print(get_score_prompt)
#         final_score = gpt_4o_mini(max_token=10000, temperature=0, system_role='', user_input=get_score_prompt)
#         try:
#             node_score[single_node_id] = float(final_score)
#         except:
#             node_score[single_node_id] = 'null'

#     node_weight_result_score[key] = node_score

#     ## 不删除是对的,删除了是错的
#     counter_num_score = dict(Counter(node_score.values()))
#     if len(counter_num_score) == 1:
#         continue

#     score_list = list(node_score.values())
#     max_score = max(score_list)
#     for single_node_id_, single_node_score in node_score.items():
#         if single_node_score < max_score:
#             node_weight_result[key].append(single_node_id_)

# with open(f'random_policy/task4/node_weight_score_{args.process}.json', 'w', encoding='utf-8') as file:
#     json.dump(node_weight_result_score, file, indent=4, ensure_ascii=False)

# with open(f'random_policy/task4/node_weight_{args.process}.json', 'w', encoding='utf-8') as file:
#     json.dump(node_weight_result, file, indent=4, ensure_ascii=False)


# ## 确定重要节点 task5
# from collections import Counter


# with open('data/MedChain/task5/train_set_gt.json', 'r', encoding='utf-8') as file:
#     MedChain_data_gt = json.load(file)

# with open(f'random_policy/task5/result.json', 'r', encoding='utf-8') as file:
#     result = json.load(file)

# node_weight_result = {}

# node_weight_result_score = {}

# for key, value in tqdm(result.items()):
    
#     if len(value) > 4:
#         value = dict(list(value.items())[:4])

#     counter_num = dict(Counter(value.values()))
#     if len(counter_num) == 1:
#         continue

#     node_weight_result[key] = []
#     node_score =  {}

#     for single_node_id, single_node_value in value.items():

#         diagnosis_gt = MedChain_data_gt[key]
#         pre_list = []
#         GT_list = ['物理疗法', '免疫疗法', '手术', '基因治疗', '药物治疗', '介入治疗', '放射治疗', '抗生素治疗', '化学治疗', '心理治疗', '中医治疗']
#         for single_gt in GT_list:
#             if single_gt in single_node_value:
#                 pre_list.append(single_gt)

#         node_score[single_node_id] = intersection_over_union(pre_list, diagnosis_gt)

#     node_weight_result_score[key] = node_score

#     ## 不删除是对的,删除了是错的
#     counter_num_score = dict(Counter(node_score.values()))
#     if len(counter_num_score) == 1:
#         continue

#     score_list = list(node_score.values())
#     max_score = max(score_list)
#     for single_node_id_, single_node_score in node_score.items():
#         if single_node_score < max_score:
#             node_weight_result[key].append(single_node_id_)


# with open(f'random_policy/task5/node_weight_score.json', 'w', encoding='utf-8') as file:
#     json.dump(node_weight_result_score, file, indent=4, ensure_ascii=False)

# with open(f'random_policy/task5/node_weight.json', 'w', encoding='utf-8') as file:
#     json.dump(node_weight_result, file, indent=4, ensure_ascii=False)


# ## 节点重组
# ## task 1
# with open('data/MedChain/task1/train_set_gt.json', 'r', encoding='utf-8') as file:
#     train_set_gt = json.load(file)

# with open('random_policy/task1_1/node_weight.json', 'r', encoding='utf-8') as file:
#     task1_1_weight = json.load(file)

# with open('random_policy/task1_2/node_weight.json', 'r', encoding='utf-8') as file:
#     task1_2_weight = json.load(file)

# with open('random_policy/task2/node_weight_ass.json', 'r', encoding='utf-8') as file:
#     task2_ass = json.load(file)

# with open('random_policy/task2/node_weight_phy.json', 'r', encoding='utf-8') as file:
#     task2_phy = json.load(file)

# with open('random_policy/task4/node_weight.json', 'r', encoding='utf-8') as file:
#     task4_weight = json.load(file)

# with open('random_policy/task5/node_weight.json', 'r', encoding='utf-8') as file:
#     task5_weight = json.load(file)

# ## 载入 policy
# with open('inference_process_data/task1_level1/policy/train_data/policy.json', 'r', encoding='utf-8') as file:
#     task1_1_policy = json.load(file)

# with open('inference_process_data/task1_level2/policy/train_data/policy.json', 'r', encoding='utf-8') as file:
#     task1_2_policy = json.load(file)

# with open('inference_process_data/task2/policy/train_policy.json', 'r', encoding='utf-8') as file:
#     task2_policy = json.load(file)

# with open('inference_process_data/task4/policy/policy_train_data_dict.json', 'r', encoding='utf-8') as file:
#     task4_policy = json.load(file)

# with open('inference_process_data/task5/policy/policy_train_data_dict.json', 'r', encoding='utf-8') as file:
#     task5_policy = json.load(file)


# def extract_node_search(node_list):
#     new_policy = {}
#     # print(key)
#     for single_node_id, single_node in enumerate(node_list):
#         if 'tools' not in list(single_node.keys()):
#             single_node['tools'] = '无'

#         if single_node['task_type'] == 'search' and single_node['tools'] == '无':
#             node_id = single_node['node_id']
#             new_policy[node_id] = single_node
#     return new_policy


# def extract_node_solve(node_list):
#     new_policy = {}
#     # print(key)
#     for single_node_id, single_node in enumerate(node_list):
#         if 'tools' not in list(single_node.keys()):
#             single_node['tools'] = '无'

#         if single_node['task_type'] == 'solve' and single_node['tools'] == '无':
#             node_id = single_node['node_id']
#             new_policy[node_id] = single_node
#     return new_policy


# rebuild_node_data = {}
# policy_length = []

# for key, value in train_set_gt.items():

#     high_node = {}

#     ## 构建 node pooling
#     node_pool = {}

#     try:
#         node_pool['task1_1'] = extract_node_search(task1_1_policy[key]['nodes'])
#     except:
#         node_pool['task1_1'] = []

#     try:
#         node_pool['task1_2'] = extract_node_search(task1_2_policy[key]['nodes'])
#     except:
#         node_pool['task1_2'] = []

#     try:
#         node_pool['task2'] = extract_node_search(task2_policy[key]['nodes'])
#     except:
#         node_pool['task2'] = []

#     try:
#         node_pool['task4'] = extract_node_search(task4_policy[key]['nodes'])
#     except:
#         node_pool['task4'] = []

#     try:
#         node_pool['task5'] = extract_node_search(task5_policy[key]['nodes'])
#     except:
#         node_pool['task5'] = []

#     ## 构建 node pooling solve
#     node_pool_solve = {}

#     try:
#         node_pool_solve['task1_1'] = extract_node_solve(task1_1_policy[key]['nodes'][:-1])
#     except:
#         node_pool_solve['task1_1'] = []

#     try:
#         node_pool_solve['task1_2'] = extract_node_solve(task1_2_policy[key]['nodes'][:-1])
#     except:
#         node_pool_solve['task1_2'] = []

#     try:
#         node_pool_solve['task2'] = extract_node_solve(task2_policy[key]['nodes'][:-1])
#     except:
#         node_pool_solve['task2'] = []

#     try:
#         node_pool_solve['task4'] = extract_node_solve(task4_policy[key]['nodes'][:-1])
#     except:
#         node_pool_solve['task4'] = []

#     try:
#         node_pool_solve['task5'] = extract_node_solve(task5_policy[key]['nodes'][:-1])
#     except:
#         node_pool_solve['task5'] = []

#     ## task1-1
#     try:
#         task1_1_nodel = task1_1_weight[key]
#     except:
#         task1_1_nodel = []

#     if len(task1_1_nodel) != 0:
#         high_node['task1_1'] = task1_1_nodel

#     ## task1-2
#     try:
#         task1_2_nodel = task1_2_weight[key]
#     except:
#         task1_2_nodel = []

#     if len(task1_2_nodel) != 0:
#         high_node['task1_2'] = task1_2_nodel

#     ## task2-ass
#     try:
#         task2_ass_nodel = task2_ass[key]
#     except:
#         task2_ass_nodel = []

#     if len(task2_ass_nodel) != 0:
#         high_node['task2_ass'] = task2_ass_nodel

#     ## task2-phy
#     try:
#         task2_phy_nodel = task2_phy[key]
#     except:
#         task2_phy_nodel = []

#     if len(task2_phy_nodel) != 0:
#         high_node['task2_phy'] = task2_phy_nodel

#     ## task4
#     try:
#         task4_nodel = task4_weight[key]
#     except:
#         task4_nodel = []

#     if len(task4_nodel) != 0:
#         high_node['task4'] = task4_nodel

#     ## task5
#     try:
#         task5_nodel = task5_weight[key]
#     except:
#         task5_nodel = []

#     if len(task5_nodel) != 0:
#         high_node['task5'] = task5_nodel

#     if len(high_node) == 0:
#         continue

#     chosen_question = []
#     for task_id, node_id in high_node.items():
#         for single_node_id in node_id:
#             try:
#                 chosen_question.append(node_pool[task_id][single_node_id])
#                 node_pool[task_id].pop(single_node_id, None)
#             except:
#                 pass
    
#     ## 如果找不到 search 节点
#     if len(chosen_question) < 3:
#         chosen_question += random_sample_values(node_pool['task5'], 3 - len(chosen_question))

#     ## 加入 solve 节点
#     for task_id, node_dict in node_pool_solve.items():
#         if len(node_dict) != 0:
#             for _, node_contenct in node_dict.items():
#                 chosen_question.append(node_contenct)

#     single_rebuild_node_data = {}
#     for single_node_num, single_node in enumerate(chosen_question, 1):
#         single_node['node_id'] = f'n{single_node_num}'
#         single_rebuild_node_data[f'n{single_node_num}'] = single_node

#     policy_length.append(len(chosen_question))
#     rebuild_node_data[key] = single_rebuild_node_data

# print(len(rebuild_node_data))

# mean_value = np.mean(policy_length)
# print(f"均值：{mean_value}")

# median_value = np.median(policy_length)
# print(f"中位数：{median_value}")

# with open('random_policy/policy_all_task.json', 'w', encoding='utf-8') as file:
#     json.dump(rebuild_node_data, file, indent=4, ensure_ascii=False)


# ## 生成 SFT 数据
# with open('random_policy/policy_all_task.json', 'r', encoding='utf-8') as file:
#     policy_all_task = json.load(file)

# with open('prompt/generation_DAG/planing_all_task.txt', 'r', encoding='utf-8') as file:
#     planing_all_task = file.read()

# with open('data/MedChain/patient_condiction.json', 'r', encoding='utf-8') as file:
#     patient_description = json.load(file)

# sft_data = []

# for key, value in policy_all_task.items():
#     single_patient_description = patient_description[key]
#     planing_all_task_prompt = planing_all_task.replace("{description}", single_patient_description)
#     sft_data.append(
#         {
#             "instruction": planing_all_task_prompt,
#             "input": "",
#             "output": str(value)
#         }
#     )

# with open(F'random_policy/policy_all_task_train_data.json', 'w', encoding='utf-8') as file:
#     json.dump(sft_data, file, indent=4, ensure_ascii=False)
    
# print(len(sft_data))

