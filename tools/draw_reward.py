import re
import ast
import json
import matplotlib.pyplot as plt
import numpy as np

def extract_dicts_from_log(log_file_path):
    # 读取整个日志文件内容
    with open(log_file_path, 'r', encoding='utf-8') as file:
        log_content = file.read()
    
    # 使用正则表达式匹配所有字典结构
    # 这个模式可以匹配跨多行的字典内容
    dict_pattern = r"\{[\s\S]*?\}"
    dict_strings = re.findall(dict_pattern, log_content)
    
    # 安全解析字符串为字典对象
    result = []
    for s in dict_strings:
        try:
            # 尝试直接解析
            d = ast.literal_eval(s)
            result.append(d)
        except (SyntaxError, ValueError):
            try:
                # 如果失败，尝试替换单引号为双引号后使用json加载
                d = json.loads(s.replace("'", '"'))
                result.append(d)
            except json.JSONDecodeError:
                # 如果还是失败，记录错误但继续处理其他内容
                print(f"无法解析字典内容: {s[:100]}...")
    
    return result

# 使用示例
if __name__ == "__main__":
    log_file = "train.log"  # 替换为你的日志文件路径
    extracted_dicts = extract_dicts_from_log(log_file)

    reward_list = []
    for single_log in extracted_dicts:
        if 'reward' in list(single_log.keys()):
            reward_list.append(single_log['reward'])

    float_list = reward_list

    plt.figure(figsize=(12, 7))
    plt.plot(float_list, 
            marker='o', 
            linestyle='-', 
            color='steelblue', 
            linewidth=2.5, 
            markersize=9,
            label='Reward')

    for i, value in enumerate(float_list):
        plt.text(i, value + 0.15, f'{value:.2f}', 
                ha='center', 
                va='bottom', 
                fontsize=9,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))

    x = np.arange(len(float_list))
    z = np.polyfit(x, float_list, 10)  # 1次多项式拟合
    p = np.poly1d(z)
    plt.plot(x, p(x), "r--", linewidth=2, label='Trend')

    plt.title(log_file, fontsize=16, pad=20)
    plt.xlabel('step', fontsize=12, labelpad=10)
    plt.ylabel('Reward', fontsize=12, labelpad=10)

    plt.xticks(range(len(float_list)), [f'step {i+1}' for i in range(len(float_list))], rotation=45)
    plt.ylim(min(float_list) - 1, max(float_list) + 1)

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper left', fontsize=11)

    plt.tight_layout()
    plt.savefig(f"{log_file.replace('log', '')}.png", dpi=300, bbox_inches='tight')

