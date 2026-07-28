from tools.execute_DAG_task2 import *
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
            final_result, tool_calls, DAG_dict = execute_DAG_task2(patient_key[idx], value)
            if MedChain_data_gt[patient_key[idx]]['physical'] == '无':
                physical_gt = []
            else:
                physical_gt = list(MedChain_data_gt[patient_key[idx]]['physical'].keys())

            if MedChain_data_gt[patient_key[idx]]['auxiliary'] == '无':
                auxiliary_gt = []
            else:
                auxiliary_gt = list(MedChain_data_gt[patient_key[idx]]['auxiliary'].keys())

            physical_gt_list = ['皮肤检查', '泌尿生殖系统检查', '腹部检查', '头颅眼耳鼻喉检查', '胸部检查', '一般检查', '神经系统检查', '颈部检查', '脊柱和四肢检查']
            auxiliary_gt_list = ['CT', '超声', '内镜检查', '病理检查', '粪便检查', '血液学检查', 'X-ray', '尿液检查', '核医学成像', 'MRI']

            physical_pre = []
            auxiliary_pre = []

            for single_physical_item in physical_gt_list:
                if single_physical_item in final_result:
                    physical_pre.append(single_physical_item)

            for single_auxiliary_item in auxiliary_gt_list:
                if single_auxiliary_item in final_result:
                    auxiliary_pre.append(single_auxiliary_item)

            physical_iou = intersection_over_union(physical_pre, physical_gt)
            auxiliary_iou = intersection_over_union(auxiliary_pre, auxiliary_gt)

            # mean_iou = (physical_iou + auxiliary_iou) / 2.0
            mean_iou = physical_iou
            rewards[idx] = mean_iou

        except Exception as e:
            print(e)
            print('error!')
            rewards[idx] = -1
            
    print(f"reward: {rewards}\n\n")
    return rewards










