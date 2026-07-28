import json
import sys
sys.path.append("/data1/huangguolin/workplace2/graph_agent/graph_agent_v5/")
from util.utils import *
from collections import defaultdict


def extract_json_data(sample):
    if "【病案介绍】" not in sample[1].keys() or "主诉" not in sample[1]["【病案介绍】"].keys():
        zhusu = "None"
    else:
        zhusu = ",".join(sample[1]["【病案介绍】"]["主诉"])
    if "【病案介绍】" not in sample[1].keys() or "既往史" not in sample[1]["【病案介绍】"].keys():
        jiwangshi = "None"
    else:
        jiwangshi = " ".join(sample[1]["【病案介绍】"]["既往史"])
    if "【病案介绍】" not in sample[1].keys() or "现病史" not in sample[1]["【病案介绍】"].keys():
        xianbingshi = "None"
    else:
        xianbingshi = " ".join(sample[1]["【病案介绍】"]["现病史"])
    if "【病案介绍】" not in sample[1].keys() or "查体" not in sample[1]["【病案介绍】"].keys():
        chati = "None"
    else:
        chati = sample[1]["【病案介绍】"]["查体"]
    if "tags" not in sample[1].keys() or "科室" not in sample[1]["tags"].keys():
        keshi = "None"
    else:
        keshi = " ".join(sample[1]["tags"]["科室"])
    if "tags" not in sample[1].keys() or "病种" not in sample[1]["tags"].keys():
        jieguo = "None"
    else:
        jieguo = " ".join(sample[1]["tags"]["病种"])
    return zhusu, jiwangshi, xianbingshi, chati, keshi, jieguo


# dataset_path = "test_set.json"
# with open(dataset_path, 'r', encoding='utf-8') as file:
#     data_all = list(json.load(file).items())

# with open('prompt/excute/task3.txt', 'r', encoding='utf-8') as file:
#     task3_prompt = file.read()

# with open('inference_process_data/task3/pre_data.json', 'r', encoding='utf-8') as file:
#     pre_data = json.load(file)

# with open('ablation_exp/sft_policy_data/pre_data_test_extract.json', 'r', encoding='utf-8') as file:
#     check_extract = json.load(file)

# with open('data/MedChain/patient_condiction.json', 'r', encoding='utf-8') as file:
#     patient_condiction = json.load(file)

# img_src = '/data1/huangguolin/hgl_wrokspace/ourframework/datasets/MedImg/'

# task3_json = {}
# process_data = []

# for sample in tqdm(data_all, desc="Case"):
#     if "【病案介绍】" not in sample[1].keys() or "图像" not in sample[1]["【病案介绍】"].keys():
#         task3_judge = False
#         task3_result = "None"
#     else:
#         process_data.append(sample)

# split_data = split_list_into_parts(process_data, 20)
# for sample in tqdm(split_data[args.process-1], desc="Case"):
# # for sample in tqdm(process_data, desc="Case"):
#     zhusu, jiwangshi, xianbingshi, chati, keshi, jieguo = extract_json_data(sample)

#     if "【病案介绍】" not in sample[1].keys() or "图像" not in sample[1]["【病案介绍】"].keys():
#         task3_judge = False
#         task3_result = "None"

#     else:
#         tuxiang = [item["文件名"] for item in sample[1]["【病案介绍】"]["图像"]]
#         tuxiang_calss = [item["分类"] for item in sample[1]["【病案介绍】"]["图像"]]
#         tuxiang_calss = [item[0] for item in tuxiang_calss]
#         img_path = tuxiang
#         img_class = tuxiang_calss
#         if len(img_path) >= 3:
#             img_path = [f'{img_src}{path}' for path in img_path[:3]]
#             img_class = img_class[:3]
#         else:
#             img_path = [f'{img_src}{path}' for path in img_path]
        
#         # img_path = [img_path[0]]

#         C = defaultdict(list)
#         for category, filename in zip(img_class, img_path):
#             C[category].append(filename)
#         C = dict(C)
#         imgs = [[category, filenames] for category, filenames in C.items()]
#         img_path = imgs

#         down3_sys_role = f"你是一名专业的影像医生,拥有丰富的临床经验，能够很好地生成医学影像报告。"
#         task3_json[sample[0]] = {}

#         for single_case in img_path:

#             # try:
#             #     sys_role = task3_prompt.replace("[img_class]", single_case[0]).replace("[symptom]", pre_data[sample[0]] + f'一些检查：\n{check_extract[sample[0]]}')
#             # except:
#             #     try:
#             #         sys_role = task3_prompt.replace("[img_class]", single_case[0]).replace("[symptom]", patient_condiction[sample[0]] + f'一些检查：\n{check_extract[sample[0]]}')
#             #     except:
#             #         sys_role = task3_prompt.replace("[img_class]", single_case[0]).replace("[symptom]", patient_condiction[sample[0]])

#             sys_role = task3_prompt.replace("[img_class]", single_case[0]).replace("[symptom]", patient_condiction[sample[0]])

#             # try:
#             #     sys_role = task3_prompt.replace("[img_class]", single_case[0]).replace("[symptom]", zhusu)
#             # except:
#             #     sys_role = task3_prompt.replace("[img_class]", single_case[0]).replace("[symptom]", '无')

#             # print(sys_role)
#             try:
#                 img_report = img_api(img_path=single_case[1], user_input=sys_role)
#             except Exception as e:
#                 print(e)
#                 img_report = '无'
#             # print(img_report)


#             task3_json[sample[0]][single_case[0]] = img_report

# with open(f'nips2026_rebuttal/evopatient_eveagent/task3/result_{args.process}.json', 'w', encoding='utf-8') as file:
#     json.dump(task3_json, file, indent=4, ensure_ascii=False)

# with open(f'result_task3.json', 'w', encoding='utf-8') as file:
#     json.dump(task3_json, file, indent=4, ensure_ascii=False)

# ##  合并
# inference_test_all = {}
# for num  in range(1, 21):
#     with open(f'nips2026_rebuttal/evopatient_eveagent/task3/result_{num}.json', 'r', encoding='utf-8') as file:
#         inference_test_part = json.load(file)
#     for key, value in inference_test_part.items():
#         inference_test_all[key] = value

# with open(F'nips2026_rebuttal/evopatient_eveagent/task3/result.json', 'w', encoding='utf-8') as file:
#     json.dump(inference_test_all, file, indent=4, ensure_ascii=False)

# print(len(inference_test_all))










