import json


with open('inference/qwen3-4B_policy.json', 'r', encoding='utf-8') as file:
    policy = json.load(file)

with open('inference/qwen3-4B_result.json', 'r', encoding='utf-8') as file:
    result = json.load(file)


for key, value in result.items():
    if value['pre'] != value['gt']:
        single_policy = policy[key]









