import sys
sys.path.append("/data1/huangguolin/workplace2/graph_agent/graph_agent_v5/")
import json
import os
from util.utils import *
import ast


# ## 生成策略
# with open('ablation_exp/policy_sft/task4/policy_prompt_test.json', 'r', encoding='utf-8') as file:
#     train_prompt = json.load(file)

# policy_all = {}

# # for key, value in tqdm(train_prompt.items()):

# split_data = split_list_into_parts(list(train_prompt.items()), 20)
# for key, value in tqdm(split_data[args.process-1], desc="Case"):
#     response = get_llama_api(max_token=10000, temperature=0, system_role='', user_input=value).split('</think>')[-1]
#     policy_all[key] = response

# with open(F'ablation_exp/grpo_policy_exp/policy_test_task4_{args.process}.json', 'w', encoding='utf-8') as file:
#     json.dump(policy_all, file, indent=4, ensure_ascii=False)

# ## 合并
# policy_train_data_all = {}
# for num in range(1, 21):
#     with open(f'nips2026_rebuttal/node_type_ablation/result/task4/result_inference_prompt_wo_tool_{num}.json', 'r', encoding='utf-8') as file:
#         policy_train_data_part = json.load(file)
#     for key, value in policy_train_data_part.items():
#         policy_train_data_all[key] = value

# with open(F'nips2026_rebuttal/node_type_ablation/result/task4/result_inference_prompt_wo_tool.json', 'w', encoding='utf-8') as file:
#     json.dump(policy_train_data_all, file, indent=4, ensure_ascii=False)

# print(len(policy_train_data_all))

# ## 执行策略
# from tools.execute_DAG_task4 import *

# with open(F'unified_model/policy_rebuild_test.json', 'r', encoding='utf-8') as file:
#     policy_train_data = json.load(file)

# with open(f'inference_process_data/task4/policy/pre_data_test_data.json', 'r', encoding='utf-8') as file:
#     pre_data = json.load(file)

# inference_prompt = {}

# split_data = split_list_into_parts(list(policy_train_data.items()), 20)
# for key, value in tqdm(split_data[args.process-1], desc="Case"):
# # for key, value in policy_train_data.items():

#     try:
#         # plan_DAG = ast.literal_eval(value)
#         plan_DAG = value
#         sft_input = execute_DAG_task4(key, plan_DAG, pre_data[key], train_executor=True, inference_con=False, node_weight=False)
#         # print(sft_input)
#         inference_prompt[key] = sft_input
#     except:
#         continue

# with open(F'nips2026_rebuttal/multi_turn/turn3/task4/inference_prompt_{args.process}.json', 'w', encoding='utf-8') as file:
#     json.dump(inference_prompt, file, indent=4, ensure_ascii=False)

# ##  合并
# for file_name in ['interaction', 'think', 'tool']:
#     inference_test_all = {}
#     for num  in range(1, 21):
#         with open(f'nips2026_rebuttal/node_type_ablation/result/task4/result_wo_{file_name}_{num}.json', 'r', encoding='utf-8') as file:
#             inference_test_part = json.load(file)
#         for key, value in inference_test_part.items():
#             inference_test_all[key] = value

#     with open(F'nips2026_rebuttal/node_type_ablation/result/task4/result_wo_{file_name}.json', 'w', encoding='utf-8') as file:
#         json.dump(inference_test_all, file, indent=4, ensure_ascii=False)

#     print(len(inference_test_all))

# # 直接推理 
# for file_name in ['interaction', 'think', 'tool']:

#     with open(f'nips2026_rebuttal/node_type_ablation/task4/inference_prompt_wo_{file_name}.json', 'r',  encoding='utf-8') as file:
#         inference_test = json.load(file)

#     result = {}

#     split_data = split_list_into_parts(list(inference_test.items()), 20)
#     for key, value in tqdm(split_data[args.process-1], desc="Case"):
#     # for key, value in tqdm(inference_test.items()):
#         response = get_llama_api(max_token=10000, temperature=0, system_role='', user_input=value).split('</think>')[-1]
#         # response = gpt_55(max_token=10000, temperature=0, system_role='', user_input=value)
#         result[key] = response

#     # with open(F'ablation_exp/sft_policy_data/result_task4.json', 'w', encoding='utf-8') as file:
#     #     json.dump(result, file, indent=4, ensure_ascii=False)

#     with open(F'nips2026_rebuttal/node_type_ablation/result/task4/result_wo_{file_name}_{args.process}.json', 'w', encoding='utf-8') as file:
#         json.dump(result, file, indent=4, ensure_ascii=False)

# # 直接推理 (断点)
# import json
# import os
# from pathlib import Path
# from tqdm import tqdm


# # 读取推理数据
# inference_path = Path(
#     "nips2026_rebuttal/CoEAgent_wo_DPA/evopatient/task4/inference_prompt.json"
# )

# with inference_path.open("r", encoding="utf-8") as file:
#     inference_test = json.load(file)


# # 每个 process 使用独立的结果文件
# output_path = Path(
#     f"nips2026_rebuttal/CoEAgent_wo_DPA/evopatient/task4/"
#     f"result_gpt55_{args.process}.json"
# )
# output_path.parent.mkdir(parents=True, exist_ok=True)


# # 读取已经生成的结果
# if output_path.exists():
#     with output_path.open("r", encoding="utf-8") as file:
#         result = json.load(file)

#     if not isinstance(result, dict):
#         raise ValueError(f"结果文件格式错误，应为字典：{output_path}")

#     print(f"已读取 {len(result)} 条历史结果，将继续生成未完成病例。")
# else:
#     result = {}


# # 将数据划分成20份，选择当前进程负责的部分
# split_data = split_list_into_parts(
#     list(inference_test.items()),
#     20,
# )
# current_process_data = split_data[args.process - 1]


# # 找出当前进程尚未完成的病例
# pending_data = [
#     (key, value)
#     for key, value in current_process_data
#     if key not in result
# ]

# print(
#     f"Process {args.process}: "
#     f"总计 {len(current_process_data)} 条，"
#     f"已完成 {len(current_process_data) - len(pending_data)} 条，"
#     f"剩余 {len(pending_data)} 条。"
# )


# for key, value in tqdm(pending_data, desc=f"Process {args.process}"):

#     response = gpt_55(
#         max_token=10000,
#         temperature=0,
#         system_role="",
#         user_input=value,
#     )

#     result[key] = response

#     # 先写入临时文件，再原子替换正式结果文件。
#     # 避免保存过程中程序中断导致正式 JSON 文件损坏。
#     temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")

#     with temporary_path.open("w", encoding="utf-8") as file:
#         json.dump(
#             result,
#             file,
#             indent=4,
#             ensure_ascii=False,
#         )

#     os.replace(temporary_path, output_path)


# print(
#     f"Process {args.process} 推理完成，"
#     f"共保存 {len(result)} 条结果至：{output_path}"
# )


# ## 评估
# for file_name in ['interaction', 'think', 'tool']:
#     with open('data/MedChain/task4/test_set_gt.json', 'r', encoding='utf-8') as file:
#         MedChain_data_gt = json.load(file)

#     with open('prompt/generation_DAG/task4/get_score.txt', 'r', encoding='utf-8') as file:
#         get_score = file.read()

#     with open('test_set.json', 'r', encoding='utf-8') as file:
#         train_set = json.load(file)

#     with open(F'nips2026_rebuttal/node_type_ablation/result/task4/result_wo_{file_name}.json', 'r', encoding='utf-8') as file:
#         result_test = json.load(file)

#     score_total = 0
#     final_result = {}
#     error_num = 0

#     split_data = split_list_into_parts(list(result_test.items()), 20)
#     for key, value in tqdm(split_data[args.process-1], desc="Case"):
#     # for key, value in tqdm(result_test.items()):

#         diagnosis_gt = MedChain_data_gt[key]
#         get_score_prompt = get_score.replace("{report}", str(train_set[key])).replace("{gt_result}", str(diagnosis_gt)).replace("{model_result}", str(value))

#         # print(get_score_prompt)

#         # print(response)

#         # print(response)

#         response = gpt_4o_mini(max_token=10000, temperature=0, system_role='', user_input=get_score_prompt).split('</think>')[-1]
#         final_result[key] = float(response)
        
#         # try:
#         #     response = gpt_4o_mini(max_token=10000, temperature=0, system_role='', user_input=get_score_prompt).split('</think>')[-1]
#         #     final_result[key] = float(response)
#         # except:
#         #     response = get_llama_api(max_token=10000, temperature=0, system_role='', user_input=get_score_prompt).split('</think>')[-1]
#         #     final_result[key] = float(response)
#         #     error_num += 1
#         # # score_total += float(response)

#     with open(F'nips2026_rebuttal/node_type_ablation/result/task4/task4_wo_{file_name}_score_{args.process}.json', 'w', encoding='utf-8') as file:
#         json.dump(final_result, file, indent=4, ensure_ascii=False)

#     # print(error_num)

# ## 合并 
# result_test_all = {}
# score_total = 0
# for num in range(1, 21):
#     with open(f'nips2026_rebuttal/multi_turn/turn3/task4/task4_score_{num}.json', 'r', encoding='utf-8') as file:
#         result_test_part = json.load(file)
#     for key, value in result_test_part.items():
#         result_test_all[key] = value

# with open(F'nips2026_rebuttal/multi_turn/turn3/task4/task4_score.json', 'w', encoding='utf-8') as file:
#     json.dump(result_test_all, file, indent=4, ensure_ascii=False)

# print(len(result_test_all))
