from tools.execute_DAG_task1_1_policy import *

# reward
T_FREE_TOOL = 2
LAMBDA_TOOL = 0.2


def extract_policy_report_task1_1(DAG_dict):
    DAG_dict['nodes'] = DAG_dict['nodes'][:-1]
    for node_idx, _ in enumerate(DAG_dict['nodes']):
        DAG_dict['nodes'][node_idx].pop("tools", "N/A")
        DAG_dict['nodes'][node_idx].pop("result_tools", "N/A")

    return DAG_dict


# def reward_fn(samples, **kwargs):
def reward_fn(prompts, completions, completion_ids=None, patient_key=None, **kwargs):
    
    rewards = [0] * len(completions)
    plan_DAG = {}
    
    for plan_id, plan_str in enumerate(completions):
        plan_str_input = plan_str[0]['content'].split('</think>')[-1]
        try:
            plan_DAG[plan_id] = json.loads(plan_str_input)
            # print(plan_DAG[plan_id])
        except Exception as e:
            # print("转化为图模型 - 异常")
            rewards[plan_id] = -1
            plan_DAG[plan_id] = 'error'

    for idx, (key, value) in enumerate(tqdm(plan_DAG.items(), desc="reward")):
        if value == 'error':
            continue
        try:
            final_result, tool_calls, DAG_dict = execute_DAG_task1_1_policy(patient_key[idx], value)
            gt = MedChain_data_gt[patient_key[idx]]["task1-DG"]
            # print(gt)
            pred = final_result
            R_correct = 1.0 if pred == gt else -0.5

            # ===== 工具调用惩罚 =====
            R_tool = max(0, tool_calls - T_FREE_TOOL)

            R = R_correct - LAMBDA_TOOL * R_tool
            rewards[idx] = R

        except Exception as e:
            # print("执行异常")
            rewards[idx] = -1
            
    print(f"reward: {rewards}\n\n")
    return rewards










