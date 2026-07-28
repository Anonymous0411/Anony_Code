import json
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))  # tools目录
parent_dir = os.path.dirname(current_dir)  # 项目根目录
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from util.utils import *
from tools.patient_agent import *
from tools.search_dependency import *


with open('MedChain_data/patient_memory.json', 'r', encoding='utf-8') as file:
  patient_memory_hub = json.load(file)

with open('prompt/excute/excute_solve_node.txt', 'r', encoding='utf-8') as file:
  excute_solve_node_ori = file.read()

with open('prompt/excute/task1_first_level.txt', 'r', encoding='utf-8') as file:
  final_answer_ori = file.read()

with open('CASE_HUB/patient_condiction_static_testing.json', 'r', encoding='utf-8') as file:
  patient_condiction_embedding = json.load(file)

with open('MedChain_data/patient_condiction.json', 'r', encoding='utf-8') as file:
    patient_condiction = json.load(file)

with open('prompt/generation_DAG/build_DAG.txt', 'r', encoding='utf-8') as file:
    plan_to_DAG_ori = file.read()

with open('MedChain_data/filter_final.json', 'r', encoding='utf-8') as file:
    MedChain_data_gt = json.load(file)

with open('MedChain_data/task1_plan.json', 'r', encoding='utf-8') as file:
    task1_plan = json.load(file)


def build_DAG(plan_str):
  plan_to_DAG = plan_to_DAG_ori.replace("{plan}", plan_str).replace("{task_type}", '分诊')
  response = get_llama_api(max_token=10000, temperature=0, system_role='', user_input=plan_to_DAG)
  return response


def execute_DAG_task1(patient_key, plan_str):

  DAG_dict = json.loads(build_DAG(plan_str))

  ## 初始化
  for idx, single_node in enumerate(DAG_dict['nodes']):
    DAG_dict['nodes'][idx]['status_tools'] = 'pending'
    DAG_dict['nodes'][idx]['result_tools'] = 'pending'
    DAG_dict['nodes'][idx]['result_solve'] = 'pending'

  ## 增加最终执行节点
  DAG_dict['nodes'].append(
    {
      "node_id": "final_answer",
      "task_type": "final_answer",
      "task_desc": "final_answer",
      "question": "无",
      "tools": "无"
    }
  )

  history = patient_memory_hub[patient_key]
  dialogue_record = ""
  tools_result = ""
  solve_result = ""
  tool_call_count = 0

  node_dependencies = get_node_dependencies(DAG_dict['edges'])

  ## 执行
  for idx, single_node in enumerate(DAG_dict['nodes']):

    ## 医患交互
    if single_node['question'] != '无':
      question = single_node['question']
      try:
        response, dialogue_record_new = patient_response(history, dialogue_record, question)
        dialogue_record = dialogue_record_new
        DAG_dict['nodes'][idx]['result_communication'] = response
        DAG_dict['nodes'][idx]['status_communication'] = 'success'
      except:
        DAG_dict['nodes'][idx]['result_communication'] = "communication failure!!!"
        DAG_dict['nodes'][idx]['status_communication'] = 'fail'
    

    ## 工具调用
    if single_node['tools'] != '无':
      if "病例检索" in single_node['tools']:
        tool_call_count += 1
        top3_case_key = patient_condiction_embedding[patient_key][1:4]
        re_case_1_desc = patient_condiction[top3_case_key[0]]
        re_case_1_first = MedChain_data_gt[top3_case_key[0]]['task1']['first']
        re_case_2_desc = patient_condiction[top3_case_key[1]]
        re_case_2_first = MedChain_data_gt[top3_case_key[1]]['task1']['first']
        re_case_3_desc = patient_condiction[top3_case_key[2]]
        re_case_3_first = MedChain_data_gt[top3_case_key[2]]['task1']['first']

        tools_result += f"使用工具：病例检索\n当前病例：{patient_condiction[patient_key]}\n过往相似病例：\n1、病例描述：{re_case_1_desc}\n所属一级科室：{re_case_1_first}\n2、病例描述：{re_case_2_desc}\n所属一级科室：{re_case_2_first}\n3、病例描述：{re_case_3_desc}\n所属一级科室：{re_case_3_first}\n\n\n"

    ## 推理节点执行
    if single_node['task_type'] == 'solve':
      ## 节点依赖
      single_node_dependencies = node_dependencies[single_node['node_id']]
      excute_solve_node_prompt = excute_solve_node_ori.replace("{dialogue_record}", dialogue_record).replace("{tools_result}", tools_result).replace("{question}", single_node['task_desc']).replace("{diagnosis}", solve_result)
      # print(excute_solve_node_prompt)
      excute_result = get_llama_api(max_token=10000, temperature=0, system_role='', user_input=excute_solve_node_prompt)
      DAG_dict['nodes'][idx]['result_solve'] = excute_result
      solve_result += f"Q: {single_node['task_desc']}\nA: {excute_result}\n\n"

    if single_node['node_id'] == 'final_answer':
      final_answer_prompt = final_answer_ori.replace("{dialogue_record}", dialogue_record).replace("{tools_result}", tools_result).replace("{diagnosis}", solve_result)
      # print(final_answer_prompt)
      final_result = get_llama_api(max_token=10000, temperature=0, system_role='', user_input=final_answer_prompt)

  return final_result, tool_call_count


def compute_reward(final_result, tool_call_count, patient_key):
    gt = MedChain_data_gt[patient_key]['task1']['first']
    pred = final_result

    # 1. 正确性
    R_correct = 1.0 if pred == gt else 0.0

    # 2. 工具惩罚
    T_free = 2
    lambda_tool = 0.2
    R_tool = max(0, tool_call_count - T_free)

    # 3. 总 reward
    R = R_correct - lambda_tool * R_tool
    return R


# DAG_result = {}
# for key, value in tqdm(task1_plan.items()):

#   result, tool_call_count = execute_DAG_task1(patient_key=key, plan_str=value)
#   DAG_result[key] = result

#   R = compute_reward(result, tool_call_count, key)

#   print(f"{result}, {tool_call_count}, {R}")

#   # with open(f'output/result_test.json', 'w', encoding='utf-8') as file:
#   #     json.dump(DAG_result, file, indent=4, ensure_ascii=False)










