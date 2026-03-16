#!/usr/bin/env python3
# 大模型微调脚本

import os
import sys
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import torch

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config

class ModelFinetuner:
    def __init__(self):
        self.config = Config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def load_model(self):
        """加载预训练模型和分词器"""
        print(f"加载模型: {self.config.MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(self.config.MODEL_NAME, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.config.MODEL_NAME,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        return model, tokenizer
    
    def load_data(self):
        """加载微调数据"""
        with open(self.config.TRAIN_DATA_FILE, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        
        with open(self.config.VAL_DATA_FILE, 'r', encoding='utf-8') as f:
            val_data = json.load(f)
        
        return train_data, val_data
    
    def preprocess_function(self, examples, tokenizer):
        """预处理数据"""
        instructions = examples['instruction']
        inputs = examples['input']
        outputs = examples['output']
        
        # 构建对话格式（GLM-5兼容格式）
        prompts