#!/usr/bin/env python3
# 使用 Llama-Factory 框架微调模型

import os
import sys
import json

# 将项目根目录加入 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config


cfg = Config()
MODEL_NAME = cfg.MODEL_NAME
OUTPUT_DIR = cfg.FINETUNED_MODEL_PATH
TRAIN_PATH = "data/sft_train.jsonl"
VAL_PATH = "data/sft_val.jsonl"

# 生成 Llama-Factory 配置文件
config_content = {
    "model_name_or_path": MODEL_NAME,
    "output_dir": OUTPUT_DIR,
    "dataset": "sft_dataset",
    "dataset_dir": ".",
    "train_file": TRAIN_PATH,
    "validation_file": VAL_PATH,
    "max_seq_length": 2048,
    "batch_size": 1,
    "gradient_accumulation_steps": 8,
    "num_train_epochs": 3,
    "learning_rate": 1e-4,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.03,
    "logging_steps": 20,
    "eval_steps": 200,
    "save_steps": 200,
    "save_total_limit": 2,
    "fp16": True,
    "bf16": False,
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj", "v_proj"],
    "use_peft": True,
    "peft_type": "lora",
    "template": "qwen",  # 使用 Qwen 模板，根据模型类型调整
    "cutoff_len": 2048,
    "prompt_template": "default",
    "overwrite_output_dir": True,
    "remove_unused_columns": True,
    "load_in_8bit": False,
    "load_in_4bit": False,
    "device_map": "auto"
}

# 保存配置文件
config_path = "llm/llama_factory_config.json"
with open(config_path, 'w', encoding='utf-8') as f:
    json.dump(config_content, f, indent=2, ensure_ascii=False)

print(f"Llama-Factory 配置文件已生成: {config_path}")
print("\n使用方法:")
print("1. 从 GitHub 克隆 Llama-Factory 仓库:")
print("   git clone https://github.com/hiyouga/LLaMA-Factory.git")
print("   cd LLaMA-Factory")
print("   pip install -e .")
print("\n2. 复制配置文件到 LLaMA-Factory 目录:")
print(f"   cp {config_path} LLaMA-Factory/")
print("\n3. 运行微调命令:")
print("   cd LLaMA-Factory")
print(f"   python -m llama_factory.train --config {os.path.basename(config_path)}")
print("\n4. 微调完成后，模型会保存在:")
print(f"   {OUTPUT_DIR}")
print("\n替代方案:")
print("如果不想使用 Llama-Factory，可以继续使用原有的 finetune_deepseek_r1_distill.py 脚本，")
print("它使用标准的 Transformers 和 PEFT 库实现了 LoRA 微调。")
