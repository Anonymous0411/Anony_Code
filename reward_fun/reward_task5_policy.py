from tools.execute_DAG_task5 import *
import re

# reward
T_FREE_TOOL = 2
LAMBDA_TOOL = 0.2


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
            final_result, tool_calls, DAG_dict = execute_DAG_task5(patient_key[idx], value, pre_data[idx], inference_con=True)
            diagnosis_gt = MedChain_data_gt[patient_key[idx]]
            pre_list = []
            GT_list = ['物理疗法', '免疫疗法', '手术', '基因治疗', '药物治疗', '介入治疗', '放射治疗', '抗生素治疗', '化学治疗', '心理治疗', '中医治疗']

            for single_gt in GT_list:
                if single_gt in final_result:
                    pre_list.append(single_gt)

            iou = intersection_over_union(pre_list, diagnosis_gt)
            rewards[idx] = iou


        except Exception as e:
            print(e)
            print('error!')
            rewards[idx] = -1
            
    print(f"reward: {rewards}\n\n")
    return rewards










