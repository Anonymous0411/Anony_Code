import sys
sys.path.append("/data1/huangguolin/workplace2/graph_agent/graph_agent_v5/")
import json
from util.utils import *
from tools.patient_agent_batch import *
from tools.search_dependency import *
from concurrent.futures import ThreadPoolExecutor, as_completed
from ast import literal_eval
import re


with open('data/MedChain/patient_memory.json', 'r', encoding='utf-8') as file:
  patient_memory_hub = json.load(file)

with open('prompt/excute/excute_solve_node.txt', 'r', encoding='utf-8') as file:
  excute_solve_node_ori = file.read()

with open('prompt/excute/task1_first_level.txt', 'r', encoding='utf-8') as file:
  final_answer_ori = file.read()

with open('data/MedChain/patient_condiction.json', 'r', encoding='utf-8') as file:
    patient_condiction = json.load(file)

with open('prompt/generation_DAG/build_DAG.txt', 'r', encoding='utf-8') as file:
    plan_to_DAG_ori = file.read()

with open('data/MedChain/task1/train_set_gt.json', 'r', encoding='utf-8') as file:
    MedChain_data_gt = json.load(file)
    

def extract_nodes_and_edges(text):
    new_nodes = []
    new_nodes_edge = []
    
    nodes_match = re.search(r'新增节点：\s*(.+?)(?=新增关系：|\Z)', text, re.DOTALL)
    if nodes_match:
        nodes_text = nodes_match.group(1)
        node_pattern = r"\{[^}]*\}"
        node_strings = re.findall(node_pattern, nodes_text)
        
        for node_str in node_strings:
            try:
                node_dict = literal_eval(node_str)
                new_nodes.append(node_dict)
            except (SyntaxError, ValueError) as e:
                print(f"解析节点时出错: {node_str}")
                print(f"错误信息: {e}")
    
    edges_match = re.search(r'新增关系：\s*(.+?)(?=\Z)', text, re.DOTALL)
    if edges_match:
        edges_text = edges_match.group(1)
        edge_pattern = r"\{[^}]*\}"
        edge_strings = re.findall(edge_pattern, edges_text)
        
        for edge_str in edge_strings:
            try:
                edge_dict = literal_eval(edge_str)
                new_nodes_edge.append(edge_dict)
            except (SyntaxError, ValueError) as e:
                print(f"解析关系时出错: {edge_str}")
                print(f"错误信息: {e}")
    return new_nodes, new_nodes_edge


def extract_policy_report_task1_1(DAG_dict):
    DAG_dict['nodes'] = DAG_dict['nodes'][:-1]
    for node_idx, _ in enumerate(DAG_dict['nodes']):
        DAG_dict['nodes'][node_idx].pop("tools", "N/A")
        DAG_dict['nodes'][node_idx].pop("result_tools", "N/A")

    return DAG_dict


def process_single_node(idx, single_node, history, single_patient_condiction):
  if single_node.get('question') != '无' and single_node['carry_out'] == False:
      question = single_node['question']
      response = patient_response(history, single_patient_condiction, question)
      return idx, response
  return idx, None


def process_dag_nodes_parallel(DAG_dict, history, single_patient_condiction,  max_workers=5):
  futures = []

  with ThreadPoolExecutor(max_workers=max_workers) as executor:
      for idx, single_node in enumerate(DAG_dict['nodes']):
          futures.append(
              executor.submit(process_single_node, idx, single_node, history, single_patient_condiction)
          )

      # 汇总结果（按完成顺序返回，但 idx 不乱）
      for future in as_completed(futures):
          idx, response = future.result()
          if response is not None:
              DAG_dict['nodes'][idx]['result_communication'] = response
              DAG_dict['nodes'][idx]['carry_out'] = True


def execute_DAG_task1_1_critic(patient_key, DAG_dict, train_executor=False, inference_con=False):

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
  tool_call_count = 0

  node_dependencies = get_node_dependencies(DAG_dict['edges'])
  
  single_patient_condiction = patient_condiction[patient_key]

  process_dag_nodes_parallel(DAG_dict, history, single_patient_condiction, max_workers=5)

  ## 执行
  for idx, single_node in enumerate(DAG_dict['nodes']):
    ## 工具调用
    if single_node['tools'] != '无' and single_node['carry_out'] == False:
      if "病例检索" in single_node['tools']:
        tool_call_count += 1
        top3_case_key = patient_condiction_embedding[patient_key][1:4]
        re_case_1_desc = patient_condiction[top3_case_key[0]].replace('\n', '')
        re_case_1_first = MedChain_data_gt[top3_case_key[0]]['first']
        re_case_2_desc = patient_condiction[top3_case_key[1]].replace('\n', '')
        re_case_2_first = MedChain_data_gt[top3_case_key[1]]['first']
        re_case_3_desc = patient_condiction[top3_case_key[2]].replace('\n', '')
        re_case_3_first = MedChain_data_gt[top3_case_key[2]]['first']
        DAG_dict['nodes'][idx]['result_tools'] = f'使用工具：病例检索\n当前病例："{patient_condiction[patient_key]}"\n\n过往相似病例：\n1、病例描述："{re_case_1_desc}"\n所属一级科室：{re_case_1_first}\n2、病例描述："{re_case_2_desc}"\n所属一级科室：{re_case_2_first}\n3、病例描述："{re_case_3_desc}"\n所属一级科室：{re_case_3_first}\n'


  ## 执行
  for idx, single_node in enumerate(DAG_dict['nodes']):
    dialogue_record = ""
    tools_result = ""
    solve_result = ""

    ## 推理节点执行
    if single_node['task_type'] == 'solve' and single_node['carry_out'] == False:
      ## 节点依赖
      single_node_dependencies_list = list(node_dependencies.items())[-1][1]
      ## 汇总聊天记录
      for single_node_dependencies in single_node_dependencies_list:
        for single_node_child in DAG_dict['nodes']:
          if single_node_child['node_id'] == single_node_dependencies and single_node_child['result_communication'] != 'pending':
            dialogue_record += f"医生：{single_node_child['question']}\n病人：{single_node_child['result_communication']}\n"
            
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

      excute_solve_node_prompt = excute_solve_node_ori.replace("{dialogue_record}", dialogue_record).replace("{tools_result}", tools_result).replace("{question}", single_node['task_desc']).replace("{diagnosis}", solve_result)
      # print(excute_solve_node_prompt)
      excute_result = get_llama_api(max_token=10000, temperature=0, system_role='', user_input=excute_solve_node_prompt)
      DAG_dict['nodes'][idx]['result_solve'] = f"Q: {single_node['task_desc']}\nA: {excute_result}\n"
      DAG_dict['nodes'][idx]['carry_out'] = True

    if single_node['node_id'] == 'final_answer':
      for single_node_child in DAG_dict['nodes'][:-1]:

        if single_node_child['result_communication'] != 'pending':
          dialogue_record += f"医生：{single_node_child['question']}\n病人：{single_node_child['result_communication']}\n"

        if single_node_child['result_tools'] != 'pending':
          tools_result += single_node_child['result_tools']

        if single_node_child['result_solve'] != 'pending':
          solve_result += single_node_child['result_solve']

      final_answer_prompt = final_answer_ori.replace("{dialogue_record}", dialogue_record).replace("{tools_result}", tools_result).replace("{diagnosis}", solve_result)
      if train_executor:
        return final_answer_prompt
      # print(final_answer_prompt)
      if inference_con:
        final_result = inference_agent(max_token=10000, temperature=0, system_role='', user_input=final_answer_prompt)
      else:
        final_result = get_llama_api(max_token=10000, temperature=0, system_role='', user_input=final_answer_prompt)

  return final_result, tool_call_count, DAG_dict









