import json
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))  # tools目录
parent_dir = os.path.dirname(current_dir)  # 项目根目录
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from util.utils import *


with open('prompt/patient_agent/response.txt', 'r', encoding='utf-8') as file:
    query_ori = file.read()

def patient_response(history, dialogue_record, question):
    query_prompt = query_ori.replace("{history}", history).replace("{dialogue_record}", dialogue_record).replace("{question}", question)
    # print(query_prompt)
    response = get_llama_api(max_token=10000, temperature=0, system_role='', user_input=query_prompt)
    dialogue_record += f"医生：{question}\n病人：{response}\n"
    return response, dialogue_record








