import sys
sys.path.append("/data1/huangguolin/workplace2/graph_agent/graph_agent_v5/")
import os 
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
os.environ["WANDB_MODE"] = "offline"
import re
import torch
import json
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, get_peft_model, TaskType
from reward_fun.reward import *

 
def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()
 
 
def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def build_dataset():

    with open('prompt/generation_DAG/task1/planing.txt', 'r', encoding='utf-8') as file:
        planning_template = file.read()

    with open('data/MedChain/task1/train_set_gt.json', 'r', encoding='utf-8') as file:
        train_set_gt = json.load(file)

    with open('data/MedChain/patient_condiction.json', 'r', encoding='utf-8') as file:
        patient_condition = json.load(file)
    
    prompts = []
    patient_ids = []

    for key in train_set_gt:
        value = planning_template.replace('{description}', patient_condition[key])

        messages = [
            {"role": "system", "content": "你是一个医学分诊策略Agent。"},
            {"role": "user", "content": value}
        ]

        prompts.append(messages)
        patient_ids.append(key)

    return Dataset.from_dict({
        "prompt": prompts,
        "patient_key": patient_ids
    })


model_name = "/data1/huangguolin/hgl_wrokspace/LLM_model/Qwen3-4B-Instruct-2507"

output_dir="models/test"
run_name="Qwen3-4B-Instruct-2507-3-epoch"

training_args = GRPOConfig(
    output_dir=output_dir,
    run_name=run_name,
    learning_rate=5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type='cosine',
    logging_steps=10,
    bf16=True,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_generations=4,
    max_prompt_length=1024,
    max_completion_length=1024,
    num_train_epochs=1,
    save_steps=100,
    max_grad_norm=0.1,
    log_on_each_node=False,
    use_vllm=False,
    vllm_gpu_memory_utilization=.1,
    deepspeed="./ds_z3_offload_config.json",
    report_to="none"
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
)

lora_rank = 32
target_modules = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj"
]

lora_config = LoraConfig(
    r=lora_rank,
    lora_alpha=64,
    target_modules=target_modules,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
tokenizer.pad_token = tokenizer.eos_token

dataset = build_dataset()

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[reward_fn],
    args=training_args,
    train_dataset=dataset,
)
 
trainer.train()
 
trainer.save_model(output_dir)
