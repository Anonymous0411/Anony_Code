from tools.execute_DAG_task4 import *
import re

# reward
T_FREE_TOOL = 2
LAMBDA_TOOL = 0.2

with open('prompt/generation_DAG/task4/get_score.txt', 'r', encoding='utf-8') as file:
    get_score = file.read()

with open('train_set.json', 'r', encoding='utf-8') as file:
    train_set = json.load(file)

def extract_policy_report_task1_1(DAG_dict):
    DAG_dict['nodes'] = DAG_dict['nodes'][:-1]
    for node_idx, _ in enumerate(DAG_dict['nodes']):
        DAG_dict['nodes'][node_idx].pop("tools", "N/A")
        DAG_dict['nodes'][node_idx].pop("result_tools", "N/A")
    return DAG_dict


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


# def reward_fn(samples, **kwargs):
def reward_fn(prompts, completions, completion_ids=None, patient_key=None, pre_data=None, **kwargs):
    
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
            final_result, tool_calls, DAG_dict = execute_DAG_task4(patient_key[idx], value, pre_data[idx])

            diagnosis_gt = MedChain_data_gt[patient_key[idx]]

            get_score_prompt = get_score.replace("{report}", str(train_set[patient_key[idx]])).replace("{gt_result}", str(diagnosis_gt)).replace("{model_result}", final_result)
            # print(get_score_prompt)
            final_score = get_llama_api(max_token=10000, temperature=0, system_role='', user_input=get_score_prompt)
            try:
                rewards[idx] = float(final_score) / 5
            except:
                rewards[idx] = 0.5

        except Exception as e:
            print(e)
            print('error!')
            rewards[idx] = -1
            
    print(f"reward: {rewards}\n\n")
    return rewards










