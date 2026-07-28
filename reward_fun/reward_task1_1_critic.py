import sys
sys.path.append("/data1/huangguolin/workplace2/graph_agent/graph_agent_v5/")
from tools.execute_DAG_task1_1_critic import *


# reward
T_FREE_TOOL = 2
LAMBDA_TOOL = 0.2

with open('inference_process_data/task1_level1/policy/train_data/result.json', 'r', encoding='utf-8') as file:
    ori_pre = json.load(file)


def critic_reward_fn(prompts, completions, completion_ids=None, patient_key=None, correct_flag=None, **kwargs):

    rewards = [0] * len(completions)

    for plan_id, plan_str in enumerate(tqdm(completions)):

        patient_id = patient_key[plan_id]

        addition = plan_str[0]['content'].split('</think>')[-1]

        pred = ori_pre[patient_id]['pre']
        gt = MedChain_data_gt[patient_id]["first"]

        if "正确" in addition:
            if pred == gt:
                rewards[plan_id] = 1
            else:
                rewards[plan_id] = -1
            continue
        else:
            if pred != gt:
                rewards[plan_id] = 1
                new_nodes, new_nodes_edge = extract_nodes_and_edges(addition)
                new_report = ori_pre[patient_id]['report']
                new_report['nodes'] = new_report['nodes'][:-1]
                for node_index, _ in enumerate(new_report['nodes']):
                    new_report['nodes'][node_index]['carry_out'] = True

                for single_node in new_nodes:
                    single_node['node_id'] = 'critic'
                    single_node['carry_out'] = False
                    single_node['tools'] = '无'
                    single_node['result_communication'] = 'pending'
                    single_node['result_tools'] = 'pending'
                    single_node['result_solve'] = 'pending'
                    new_report['nodes'].append(single_node)

                try:
                    final_result, tool_calls, DAG_dict = execute_DAG_task1(patient_id, new_report)
                    gt = MedChain_data_gt[patient_id]["first"]
                    # print(gt)
                    pred = final_result
                    pred_ori = ori_pre[patient_id]['pre']

                    if pred == gt and pred_ori == gt:
                        R_revise = 0.8
                    elif pred == gt and pred_ori != gt:
                        R_revise = 1
                    elif pred != gt and pred_ori == gt:
                        R_revise = -1
                    elif pred != gt and pred_ori != gt:
                        R_revise = 0.5

                    rewards[plan_id] = R_revise

                except Exception as e:
                    print(e)
                    rewards[plan_id] = -1

            else:
                rewards[plan_id] = -1
                continue


    print(f"reward: {rewards}\n\n")
    return rewards















