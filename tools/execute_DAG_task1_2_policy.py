import sys
sys.path.append("/data1/huangguolin/workplace2/graph_agent/graph_agent_v5/")
import json
from util.utils import *
from tools.patient_agent_batch import *
from tools.search_dependency import *
from concurrent.futures import ThreadPoolExecutor, as_completed


with open('data/MedChain/patient_memory.json', 'r', encoding='utf-8') as file:
  patient_memory_hub = json.load(file)

with open('prompt/excute/excute_solve_node_task1_level2.txt', 'r', encoding='utf-8') as file:
  excute_solve_node_ori = file.read()

with open('prompt/excute/task1_first_level2.txt', 'r', encoding='utf-8') as file:
  final_answer_ori = file.read()

with open('CASE_HUB/patient_condiction_testset_embedding.json', 'r', encoding='utf-8') as file:
  patient_condiction_embedding = json.load(file)

with open('data/MedChain/patient_condiction.json', 'r', encoding='utf-8') as file:
    patient_condiction = json.load(file)

with open('data/MedChain/task1/train_set_gt.json', 'r', encoding='utf-8') as file:
    MedChain_data_gt = json.load(file)


def process_single_node(idx, single_node, history, single_patient_condiction, patient_key):
    if single_node.get('question') != '无':
        question = single_node['question']
        response = patient_response(history, single_patient_condiction, question, patient_key=patient_key, stage="task1_level2")
        return idx, response
    return idx, None


def process_dag_nodes_parallel(DAG_dict, history, single_patient_condiction, patient_key, max_workers=5):
    futures = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, single_node in enumerate(DAG_dict['nodes']):
            futures.append(
                executor.submit(process_single_node, idx, single_node, history, single_patient_condiction, patient_key)
            )

        # 汇总结果（按完成顺序返回，但 idx 不乱）
        for future in as_completed(futures):
            idx, response = future.result()
            if response is not None:
                DAG_dict['nodes'][idx]['result_communication'] = response


def execute_DAG_task1_2_policy(patient_key, DAG_dict, history_dialogue, history_diagnosis, second_room_list, train_executor=False, inference_con=False, node_weight=False):

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

  ## 初始化
  for idx, single_node in enumerate(DAG_dict['nodes']):
    DAG_dict['nodes'][idx]['result_communication'] = 'pending'
    DAG_dict['nodes'][idx]['result_tools'] = 'pending'
    DAG_dict['nodes'][idx]['result_solve'] = 'pending'

  history = patient_memory_hub[patient_key]
  tool_call_count = 0

  node_dependencies = get_node_dependencies(DAG_dict['edges'])
  
  single_patient_condiction = patient_condiction[patient_key]

  process_dag_nodes_parallel(DAG_dict, history, single_patient_condiction, patient_key, max_workers=5)

  ## 执行
  for idx, single_node in enumerate(DAG_dict['nodes']):
    ## 工具调用
    if single_node['tools'] != '无':
      if "病例检索" in single_node['tools']:
        tool_call_count += 1
        top3_case_key = patient_condiction_embedding[patient_key][1:4]
        re_case_1_desc = patient_condiction[top3_case_key[0]].replace('\n', '')
        re_case_1_first = MedChain_data_gt[top3_case_key[0]]['second']
        re_case_2_desc = patient_condiction[top3_case_key[1]].replace('\n', '')
        re_case_2_first = MedChain_data_gt[top3_case_key[1]]['second']
        re_case_3_desc = patient_condiction[top3_case_key[2]].replace('\n', '')
        re_case_3_first = MedChain_data_gt[top3_case_key[2]]['second']
        DAG_dict['nodes'][idx]['result_tools'] = f'使用工具：病例检索\n过往相似病例：\n1、病例描述："{re_case_1_desc}"\n所属二级科室：{re_case_1_first}\n2、病例描述："{re_case_2_desc}"\n所属二级科室：{re_case_2_first}\n3、病例描述："{re_case_3_desc}"\n所属二级科室：{re_case_3_first}\n'


  ## 执行
  for idx, single_node in enumerate(DAG_dict['nodes']):
    dialogue_record = ""
    tools_result = ""
    solve_result = ""

    ## 推理节点执行
    if single_node['task_type'] == 'solve':
      ## 节点依赖
      single_node_dependencies_list = node_dependencies[single_node['node_id']]
      ## 汇总聊天记录
      for single_node_dependencies in single_node_dependencies_list:
        for single_node_child in DAG_dict['nodes']:
          if single_node_child['node_id'] == single_node_dependencies and single_node_child['result_communication'] != 'pending':
            if single_node_child['question'] == 'random':
              dialogue_record += f"对话 —— 对话目的：......\n医生：......\n患者：......\n"
            else:
              dialogue_record += f"对话 —— 对话目的：{single_node_child['task_desc']}\n医生：{single_node_child['question']}\n患者：{single_node_child['result_communication']}\n"

      ## 汇总工具记录
      for single_node_dependencies in single_node_dependencies_list:
        for single_node_child in DAG_dict['nodes']:
          if single_node_child['node_id'] == single_node_dependencies and single_node_child['result_tools'] != 'pending':
            tools_result += single_node_child['result_tools']

      ## 汇总 solve 记录
      for single_node_dependencies in single_node_dependencies_list:
        for single_node_child in DAG_dict['nodes']:
          if single_node_child['node_id'] == single_node_dependencies and single_node_child['result_solve'] != 'pending':
            solve_result += single_node_child['result_solve']

      excute_solve_node_prompt = excute_solve_node_ori.replace("{dialogue_record}", dialogue_record).replace("{tools_result}", tools_result).replace("{question}", single_node['task_desc']).replace("{diagnosis}", solve_result).replace("{history_dialogue}", history_dialogue).replace("{history_diagnosis}", history_diagnosis).replace("{patient_des}", patient_condiction[patient_key])
      # print(excute_solve_node_prompt)
      excute_result = get_llama_api(max_token=10000, temperature=0, system_role='', user_input=excute_solve_node_prompt)
      DAG_dict['nodes'][idx]['result_solve'] = f"Q: {single_node['task_desc']}\nA: {excute_result}\n"

    if single_node['node_id'] == 'final_answer':
      for single_node_child in DAG_dict['nodes']:

        if single_node_child['result_communication'] != 'pending':
          if single_node_child['question'] == 'random':
            dialogue_record += f"对话 —— 对话目的：......\n医生：......\n患者：......\n"
          else:
            dialogue_record += f"对话 —— 对话目的：{single_node_child['task_desc']}\n医生：{single_node_child['question']}\n患者：{single_node_child['result_communication']}\n"

        if single_node_child['result_tools'] != 'pending':
          tools_result += single_node_child['result_tools']

        if single_node_child['result_solve'] != 'pending':
          solve_result += single_node_child['result_solve']

      if node_weight == True:
        final_answer_prompt = final_answer_ori.replace("{dialogue_record}", dialogue_record).replace("{tools_result}", '无').replace("{diagnosis}", '无').replace("{history_dialogue}", '无').replace("{history_diagnosis}", '无').replace("{history_diagnosis}", '无').replace("{second_room_list}", second_room_list).replace("{patient_des}", patient_condiction[patient_key])
      else:
        final_answer_prompt = final_answer_ori.replace("{dialogue_record}", dialogue_record).replace("{tools_result}", tools_result).replace("{diagnosis}", solve_result).replace("{history_dialogue}", history_dialogue).replace("{history_diagnosis}", history_diagnosis).replace("{history_diagnosis}", history_diagnosis).replace("{second_room_list}", second_room_list).replace("{patient_des}", patient_condiction[patient_key])
      # print(final_answer_prompt)
      if train_executor:
        return final_answer_prompt
      final_result = get_llama_api(max_token=10000, temperature=0, system_role='', user_input=final_answer_prompt)

  return final_result, tool_call_count, DAG_dict









