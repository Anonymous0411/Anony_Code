import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_path = "/data1/huangguolin/hgl_wrokspace/LLM_model/Qwen3-4B-Instruct-2507"
lora_path = "models/test"
save_path = "/data1/huangguolin/workplace2/graph_agent/model/task5/policy"

model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="cpu"   # ⭐ 强烈建议 merge 时用 CPU
)

tokenizer = AutoTokenizer.from_pretrained(base_model_path)

model = PeftModel.from_pretrained(
    model,
    lora_path,
)

model = model.merge_and_unload()

model.save_pretrained(
    save_path,
    safe_serialization=True
)
tokenizer.save_pretrained(save_path)



