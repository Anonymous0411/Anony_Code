from tools.execute_DAG_batch_task2_critic import *


# reward
T_FREE_TOOL = 2
LAMBDA_TOOL = 0.2

with open('inference_process_data/task1_level2/policy/train_data/policy_answer.json', 'r', encoding='utf-8') as file:
    ori_pre = json.load(file)

with open('data/MedChain/task1/train_set_gt.json', 'r', encoding='utf-8') as file:
    MedChain_data_gt = json.load(file)


def critic_reward_fn(prompts, completions, completion_ids=None, patient_key=None, history_dialogue=None, history_diagnosis=None, second_room_list=None, **kwargs):

    rewards = [0] * len(completions)

    # print(prompts[0][1]['content'])

    for plan_id, plan_str in enumerate(tqdm(completions)):

        patient_id = patient_key[plan_id]

        addition = plan_str[0]['content'].split('</think>')[-1]

        if "无需修改" in addition:
            pred = ori_pre[patient_id]['pre']
            gt = MedChain_data_gt[patient_id]["second"]
            if pred == gt:
                rewards[plan_id] = 0.3
            else:
                rewards[plan_id] = -1
            continue
        else:
            new_nodes, new_nodes_edge = extract_nodes_and_edges(addition)

            new_report = ori_pre[patient_id]['dict_report']
            new_report['nodes'] = new_report['nodes'][:-1]
            for node_index, _ in enumerate(new_report['nodes']):
                new_report['nodes'][node_index]['carry_out'] = True

            for single_node in new_nodes:
                single_node['carry_out'] = False
                single_node['tools'] = '无'
                single_node['result_communication'] = 'pending'
                single_node['result_tools'] = 'pending'
                single_node['result_solve'] = 'pending'
                new_report['nodes'].append(single_node)

            for single_edge in new_nodes_edge:
                new_report['edges'].append(single_edge)
            
            try:
                final_result, tool_calls, DAG_dict = execute_DAG_task1(patient_id, new_report, history_dialogue[plan_id], history_diagnosis[plan_id], second_room_list[plan_id])
                gt = MedChain_data_gt[patient_id]["first"]
                # print(gt)
                pred = final_result
                pred_ori = ori_pre[patient_id]['pre']

                if pred == gt and pred_ori == gt:
                    R_revise = 0.5
                elif pred == gt and pred_ori != gt:
                    R_revise = 1
                elif pred != gt and pred_ori == gt:
                    R_revise = -0.8
                elif pred != gt and pred_ori != gt:
                    R_revise = 0.5

                # R_tool = max(0, tool_calls - T_FREE_TOOL)
                # R = R_revise - LAMBDA_TOOL * R_tool
                # rewards[idx] = R

                rewards[plan_id] = R_revise

            except Exception as e:
                print(e)
                rewards[plan_id] = -1

    print(f"reward: {rewards}\n\n")
    return rewards















