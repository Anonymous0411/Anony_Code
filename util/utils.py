import json
# from utils import *
from tqdm import tqdm
import argparse
from openai import OpenAI
from termcolor import colored
import sys
import re
from collections import Counter
import copy
import random
import requests
from PIL import Image
import os
import csv
import numpy as np
from collections import OrderedDict
import mimetypes
import base64
import anthropic
from util.embedding import llm_embedding


parser = argparse.ArgumentParser(description="Example script with command-line arguments.")
parser.add_argument("--process", type=int, required=False, help="First parameter (e.g., a string)", default=1)
parser.add_argument("--port", type=int, required=False, help="First parameter (e.g., a string)", default=23333)
args = parser.parse_args()

# ---------------hyperparameter---------------
try:
    api_key_llama = "YOUR_API_KEY"
    base_url_llama = f"http://0.0.0.0:{args.port}/v1"
    client_llama = OpenAI(api_key=api_key_llama, base_url=base_url_llama)
    model_name_llama = client_llama.models.list().data[0].id

    # api_key_llama = "sk-ib8U2bW3cdeYQykfCfDRjCxXl61ugEbcu8p25cxPMLg7TFpT"
    # base_url_llama = "https://api.chatanywhere.org/v1/"
    # client_llama = OpenAI(api_key=api_key_llama, base_url=base_url_llama)
    # model_name_llama = "gpt-4o-mini"
    # # model_name_llama = "gemini-2.5-pro-thinking"
    # # model_name_llama = "claude-haiku-4-5-20251001-thinking"

    ## embedding model: text-embedding-3-large

    # api_key_critic = "sk-ib8U2bW3cdeYQykfCfDRjCxXl61ugEbcu8p25cxPMLg7TFpT"
    # base_url_critic = "https://api.chatanywhere.org/v1/"
    # client_critic = OpenAI(api_key=api_key_critic, base_url=base_url_critic)
    # model_name_critic = "gpt-4.1-ca"

    # api_key_critic = "YOUR_API_KEY"
    # base_url_critic = f"http://0.0.0.0:{args.port}/v1"
    # client_critic = OpenAI(api_key=api_key_critic, base_url=base_url_critic)
    # model_name_critic = client_critic.models.list().data[0].id
except:
    print(colored("请检查ori_llm与critic_model的api配置！", "red"))
    sys.exit(0)
# --------------------------------------------


def gemma_api(user_input):
    url = f"http://localhost:{args.port}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "gemma3-27b",  # 注意：这个名字要和你 serve 时的 --served-model-name 一致
        "messages": [
            {"role": "user", "content": user_input}
        ]
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()['choices'][0]['message']['content']


def extract_answer(message):
    pattern = r"### Provided Answer\n(.+?)(?=\n### |$)"
    match = re.search(pattern, message, re.DOTALL)
    if match:
        provided_answer = match.group(1).strip()
        # print(provided_answer)
    else:
        print("None")
        sys.exit(0)

    return provided_answer


def extract_question(message):
    pattern = r"### Problem\n(.+?)(?=\n### )"
    match = re.search(pattern, message, re.DOTALL)
    if match:
        provided_question = match.group(1).strip()
        # print(provided_answer)
    else:
        print("None")
        sys.exit(0)

    return provided_question


def extract_option(message):
    pattern = r"### Options\n(.+?)(?=\n### )"
    match = re.search(pattern, message, re.DOTALL)
    if match:
        provided_option = match.group(1).strip()
        # print(provided_answer)
    else:
        print("None")
        sys.exit(0)

    return provided_option


def get_llama_api(max_token, temperature, system_role, user_input):
    response = client_llama.chat.completions.create(
        model=model_name_llama,
        max_tokens=max_token, # 1000
        temperature=temperature, # 0
        messages=[
            {"role": "system", "content": system_role},
            {"role": "user", "content": user_input}
        ],
        extra_body={"chat_template_kwargs": {"thinking_mode": "off"}}  ## 百川关闭深度思考
    )
    return response.choices[0].message.content




def get_critic_api(max_token, temperature, system_role, user_input):
    response = client_critic.chat.completions.create(
        model=model_name_critic,
        max_tokens=max_token, # 1000
        temperature=temperature, # 0
        messages=[
            {"role": "system", "content": system_role},
            {"role": "user", "content": user_input}
        ],
    )
    return response.choices[0].message.content


def get_option_str(data):
    result=""
    for key, value in data.items():
        result += f"{key}. {value}\n"
    return result


def split_list_into_parts(input_list, num_parts):
    # 计算每段的大小
    part_size = len(input_list) // num_parts
    remainder = len(input_list) % num_parts  # 计算剩余元素
    
    result = []
    start = 0
    
    for i in range(num_parts):
        # 如果还有剩余元素，当前段增加一个元素
        end = start + part_size + (1 if i < remainder else 0)
        result.append(input_list[start:end])
        start = end
    
    return result


def filter_critic(input_data):

    with open('prompt/extract_critic.txt', 'r', encoding='utf-8') as file:
        extract_answer_prompt_ori = file.read()

    single_critic = input_data
    try:
        result_sentence = single_critic.split(".")[-2]
    except:
        extract_answer_prompt = extract_answer_prompt_ori.replace("{critic}", single_critic)
        final_anwer = get_llama_api(max_token=50, temperature=0, system_role='', user_input=extract_answer_prompt)
        if "false" in final_anwer.lower():
            filter_data = False
        elif "true" in final_anwer.lower():
            filter_data = True
        else:
            print(final_anwer)

        # format_data += 1

        return filter_data

    if "the answer is correct" in result_sentence:
        filter_data = True
    elif "answer is correct" in result_sentence:
        filter_data = True
    elif "the provided answer is correct" in result_sentence:
        filter_data = True
    elif "the answer is wrong" in result_sentence:
        filter_data = False
    elif "the answer is incorrect" in result_sentence:
        filter_data = False
    elif "the provided answer is wrong" in result_sentence:
        filter_data = False
    elif "the provided answer is incorrect" in result_sentence:
        filter_data = False
    elif "partially" in result_sentence:
        filter_data = False
    elif "error" in result_sentence:
        filter_data = False
    elif "not correct" in result_sentence:
        filter_data = False
    else:
        extract_answer_prompt = extract_answer_prompt_ori.replace("{critic}", single_critic)
        final_anwer = get_llama_api(max_token=50, temperature=0, system_role='', user_input=extract_answer_prompt)
        if "false" in final_anwer.lower():
            filter_data = False
        elif "true" in final_anwer.lower():
            filter_data = True
        else:
            print(final_anwer)

    return filter_data


def deduplicate_with_indices(lst):
    seen = {}
    result = []
    for index, value in enumerate(lst):
        if value not in seen:
            seen[value] = index
            result.append((value, index))
    return result


def filter_answer(input_data):

    with open('prompt/extract_answer.txt', 'r', encoding='utf-8') as file:
        extract_answer_prompt_ori = file.read()

    single_pre = input_data
    result_sentence = single_pre.split(".")[0]
    final_anwer = result_sentence[-1]
    if final_anwer not in ['A', 'B', 'C', 'D', 'E']:
        result_sentence = single_pre.split("\n")[0]
        final_anwer = result_sentence[-1]
        if final_anwer not in ['A', 'B', 'C', 'D', 'E']:
            match = re.search(r"\*\*Final Choice\*\*:\s*([A-E])", single_pre)
            if match:
                final_anwer = match.group(1)
            else:
                final_anwer = "None"
            if final_anwer not in ['A', 'B', 'C', 'D', 'E']:
                extract_answer_prompt = extract_answer_prompt_ori.replace("{answer}", single_pre)
                final_anwer = get_llama_api(max_token=50, temperature=0, system_role='', user_input=extract_answer_prompt)
                if "A" in final_anwer:
                    final_anwer = "A"
                elif "B" in final_anwer:
                    final_anwer = "B"
                elif "C" in final_anwer:
                    final_anwer = "C"
                elif "D" in final_anwer:
                    final_anwer = "D"
                elif "E" in final_anwer:
                    final_anwer = "E"
                else:
                    print(1111)
                    final_anwer = "None"

    return final_anwer


def pad_to_square(img: Image.Image, fill_color=(255, 255, 255)):
    """
    将图像补成正方形，补充的区域为白色。
    fill_color: 背景色，默认为白色
    """
    w, h = img.size
    size = max(w, h)
    
    # 创建一个正方形白底图像
    new_img = Image.new("RGB", (size, size), fill_color)
    
    # 把原图粘贴到新图的左上角 (0, 0) 位置
    new_img.paste(img, ((size - w) // 2, (size - h) // 2))
    
    return new_img


def append_to_csv(csv_path, case_id, label):
    """
    往CSV文件追加一行记录。
    如果文件不存在，会自动写入表头。
    """
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        # 如果文件不存在，先写表头
        if not file_exists:
            writer.writerow(['case_id', 'label'])
        # 写入一行数据
        writer.writerow([case_id, label])


def count_tokens(tokenizer, text):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2) if norm_vec1 != 0 and norm_vec2 != 0 else 0


def find_top_similar(input_vector, retrieval_dict, top_n=5):
    input_vec = np.array(input_vector)
    similarities = {}
    for case_id, feature_vec in retrieval_dict.items():
        vec = np.array(feature_vec)
        sim = cosine_similarity(input_vec, vec)
        similarities[case_id] = sim
    sorted_similarities = OrderedDict(
        sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    )
    
    return list(sorted_similarities.keys())[:top_n]


def split_5fold(data):
    n = len(data)
    fold_size = n // 5
    remainder = n % 5
    
    folds = []
    start = 0

    for i in range(5):
        end = start + fold_size + (1 if i < remainder else 0)
        folds.append(data[start:end])
        start = end

    result = []
    for i in range(5):
        test_set = folds[i]
        train_set = []
        for j in range(5):
            if j != i:
                train_set.extend(folds[j])
        result.append({'train': train_set, 'test': test_set})
    
    return result


## graph_agent 病人agent

def patient_agent(max_token, temperature, system_role, user_input):
    api_key_patient = "YOUR_API_KEY"
    # base_url_patient = f"http://0.0.0.0:23333/v1"
    base_url_patient = f"http://0.0.0.0:{args.port}/v1"
    client_patient = OpenAI(api_key=api_key_patient, base_url=base_url_patient)
    model_name_patient = client_patient.models.list().data[0].id

    # api_key_llama = "sk-4kQ5thuq26NGgwXmyk48fV45al7LoMccZNmk99YD6oD76XRP"
    # base_url_llama = "https://yunwu.ai/v1"
    # client_patient = OpenAI(api_key=api_key_llama, base_url=base_url_llama)
    # model_name_patient = "gpt-5.4-mini"

    response = client_patient.chat.completions.create(
        model=model_name_patient,
        max_tokens=max_token, # 1000
        temperature=temperature, # 0
        messages=[
            {"role": "system", "content": system_role},
            {"role": "user", "content": user_input}
        ],
    )
    return response.choices[0].message.content


def critic_agent(max_token, temperature, system_role, user_input):
    api_key_patient = "YOUR_API_KEY"
    base_url_patient = f"http://0.0.0.0:23334/v1"
    client_patient = OpenAI(api_key=api_key_patient, base_url=base_url_patient)
    model_name_patient = client_patient.models.list().data[0].id

    response = client_patient.chat.completions.create(
        model=model_name_patient,
        max_tokens=max_token, # 1000
        temperature=temperature, # 0
        messages=[
            {"role": "system", "content": system_role},
            {"role": "user", "content": user_input}
        ],
    )
    return response.choices[0].message.content


def inference_agent(max_token, temperature, system_role, user_input):
    api_key_patient = "YOUR_API_KEY"
    base_url_patient = f"http://0.0.0.0:23334/v1"
    client_patient = OpenAI(api_key=api_key_patient, base_url=base_url_patient)
    model_name_patient = client_patient.models.list().data[0].id

    response = client_patient.chat.completions.create(
        model=model_name_patient,
        max_tokens=max_token, # 1000
        temperature=temperature, # 0
        messages=[
            {"role": "system", "content": system_role},
            {"role": "user", "content": user_input}
        ],
    )
    return response.choices[0].message.content


def gpt_4o_mini(max_token, temperature, system_role, user_input):
    # api_key_llama = "sk-ib8U2bW3cdeYQykfCfDRjCxXl61ugEbcu8p25cxPMLg7TFpT"
    # base_url_llama = "https://api.chatanywhere.org/v1/"

    api_key_llama = "sk-4kQ5thuq26NGgwXmyk48fV45al7LoMccZNmk99YD6oD76XRP"
    base_url_llama = "https://yunwu.ai/v1"

    client_llama = OpenAI(api_key=api_key_llama, base_url=base_url_llama)
    model_name_llama = "gpt-4o-mini"

    response = client_llama.chat.completions.create(
        model=model_name_llama,
        max_tokens=max_token, # 1000
        temperature=temperature, # 0
        messages=[
            {"role": "system", "content": system_role},
            {"role": "user", "content": user_input}
        ],
    )
    return response.choices[0].message.content


def gpt_55(max_token, temperature, system_role, user_input):
    # api_key_llama = "sk-ib8U2bW3cdeYQykfCfDRjCxXl61ugEbcu8p25cxPMLg7TFpT"
    # base_url_llama = "https://api.chatanywhere.org/v1/"

    api_key_llama = "sk-4kQ5thuq26NGgwXmyk48fV45al7LoMccZNmk99YD6oD76XRP"
    base_url_llama = "https://yunwu.ai/v1"

    client_llama = OpenAI(api_key=api_key_llama, base_url=base_url_llama)
    model_name_llama = "gpt-5.4-mini"

    response = client_llama.chat.completions.create(
        model=model_name_llama,
        max_tokens=max_token, # 1000
        temperature=temperature, # 0
        messages=[
            {"role": "system", "content": system_role},
            {"role": "user", "content": user_input}
        ],
    )
    return response.choices[0].message.content


def closed_source_model(max_token, temperature, system_role, user_input, model_name):
    # api_key_llama = "sk-ib8U2bW3cdeYQykfCfDRjCxXl61ugEbcu8p25cxPMLg7TFpT"
    # base_url_llama = "https://api.chatanywhere.org/v1/"

    api_key_llama = "sk-4kQ5thuq26NGgwXmyk48fV45al7LoMccZNmk99YD6oD76XRP"
    base_url_llama = "https://yunwu.ai/v1"

    client_llama = OpenAI(api_key=api_key_llama, base_url=base_url_llama)
    model_name_llama = model_name

    response = client_llama.chat.completions.create(
        model=model_name_llama,
        max_tokens=max_token, # 1000
        temperature=temperature, # 0
        messages=[
            {"role": "system", "content": system_role},
            {"role": "user", "content": user_input}
        ],
    )
    return response.choices[0].message.content



def img_api(img_path, user_input) -> str:

    client = OpenAI(
        api_key='YOUR_API_KEY',
        base_url=f'http://0.0.0.0:{args.port}/v1'
    )
    model_name = client.models.list().data[0].id
    
    # client = OpenAI(
    #     api_key="sk-4kQ5thuq26NGgwXmyk48fV45al7LoMccZNmk99YD6oD76XRP",
    #     base_url="https://yunwu.ai/v1",
    #     timeout=120,
    # )
    # model_name = 'gpt-5.1'
    # model_name = 'kimi-k2-instruct'

    # 构建消息内容
    messages = []
    
    # 用户内容容器（支持混合文本+多图）
    user_content = [{"type": "text", "text": user_input}]
    
    base64_list = []

    # 多图处理逻辑
    if img_path:
        # 检查图片数量限制（GPT-4o-mini最多支持10张）
        if len(img_path) > 10:
            raise ValueError("GPT-4o-mini单次请求最多支持10张图片")
        
        for single_img_path in img_path:
            # 自动识别MIME类型
            mime_type, _ = mimetypes.guess_type(single_img_path)
            if not mime_type:
                mime_type = "image/jpeg"  # 默认格式
            
            # 读取并编码为base64
            try:
                with open(single_img_path, "rb") as img_file:
                    base64_img = base64.b64encode(img_file.read()).decode("utf-8")
            except Exception as e:
                raise IOError(f"图片读取失败: {single_img_path} | 错误: {str(e)}")
            
            # 添加到内容数组
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{base64_img}",
                    # "detail": "high"  # 启用细节增强模式
                }
            })
            base64_list.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{base64_img}",
                    # "detail": "high"  # 启用细节增强模式
                }
            })

    # 添加用户消息到对话历史
    messages.append({"role": "user", "content": user_content})
    
    # 调用多模态API
    response = client.chat.completions.create(
        model=model_name,  # 确认支持多模态
        max_tokens=1000,
        temperature=0,
        messages=messages,
        # extra_body={"chat_template_kwargs": {"thinking_mode": "off"}}  ## 百川关闭深度思考
    )

    return response.choices[0].message.content



