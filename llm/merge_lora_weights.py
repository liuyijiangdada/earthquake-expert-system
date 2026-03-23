#!/usr/bin/env python3
# 合并 LoRA 权重到基础模型，生成完整模型权重

import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 将项目根目录加入 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config


def main():
    cfg = Config()
    MODEL_NAME = cfg.MODEL_NAME
    LORA_MODEL_PATH = cfg.FINETUNED_MODEL_PATH
    FULL_MODEL_PATH = os.path.join(os.path.dirname(LORA_MODEL_PATH), "earthquake_expert_deepseek_r1_full")
    
    print(f"加载基础模型: {MODEL_NAME}")
    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"加载 LoRA 权重: {LORA_MODEL_PATH}")
    # 加载 LoRA 权重
    model = PeftModel.from_pretrained(base_model, LORA_MODEL_PATH)
    
    print("合并模型权重...")
    # 合并模型
    model = model.merge_and_unload()
    
    print(f"保存完整模型到: {FULL_MODEL_PATH}")
    # 保存完整模型
    model.save_pretrained(FULL_MODEL_PATH)
    tokenizer.save_pretrained(FULL_MODEL_PATH)
    
    print("合并完成！")


if __name__ == "__main__":
    main()
