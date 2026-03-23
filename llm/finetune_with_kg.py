#!/usr/bin/env python3
# 结合知识图谱的大模型微调脚本

import os
import sys
import json
import random
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import torch

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config
from kg.neo4j_kg import Neo4jKG

class KGFinetuner:
    def __init__(self):
        self.config = Config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.kg = Neo4jKG()
        self.kg.run()
    
    def generate_training_data(self, num_examples=100):
        """从知识图谱生成训练数据"""
        print("从知识图谱生成训练数据...")
        
        all_earthquakes = self.kg.query_all_earthquakes()
        training_data = []
        
        # 生成关于地震基本信息的问题
        for i, eq in enumerate(all_earthquakes[:num_examples]):
            # 问题1: 询问地震震级
            training_data.append({
                "instruction": "回答关于地震的问题",
                "input": f"{eq['location']}地震的震级是多少？",
                "output": f"{eq['location']}地震的震级是{eq['magnitude']}级。"
            })
            
            # 问题2: 询问地震深度
            training_data.append({
                "instruction": "回答关于地震的问题",
                "input": f"{eq['location']}地震的震源深度是多少？",
                "output": f"{eq['location']}地震的震源深度是{eq['depth']}公里。"
            })
            
            # 问题3: 询问地震发生时间
            training_data.append({
                "instruction": "回答关于地震的问题",
                "input": f"{eq['location']}地震发生在什么时候？",
                "output": f"{eq['location']}地震发生在{eq['time']}。"
            })
            
            # 问题4: 询问地震详细信息
            training_data.append({
                "instruction": "回答关于地震的问题",
                "input": f"请详细介绍{eq['location']}地震",
                "output": f"{eq['name']}发生在{eq['time']}，震级{eq['magnitude']}级，震源深度{eq['depth']}公里，震中位于{eq['latitude']}°N, {eq['longitude']}°E，烈度{eq['intensity']}。{eq['description']}"
            })
        
        # 生成关于震级分类的问题
        training_data.append({
            "instruction": "回答关于地震的问题",
            "input": "地震震级如何分类？",
            "output": "地震震级通常分为：微震（<3.0级）、小震（3.0-3.9级）、中震（4.0-4.9级）、强震（5.0-5.9级）、大地震（6.0-6.9级）、巨大地震（7.0-7.9级）、特大地震（≥8.0级）。"
        })
        
        # 生成关于地震安全的问题
        training_data.append({
            "instruction": "回答关于地震的问题",
            "input": "地震发生时应该如何应对？",
            "output": "地震发生时，应立即采取以下措施：1. 保持冷静，迅速躲到桌子、床等坚固家具下面；2. 远离窗户、玻璃幕墙等易碎物品；3. 如果在室外，远离建筑物、树木、电线杆等；4. 地震停止后，迅速撤离到安全地带；5. 不要使用电梯，走楼梯逃生。"
        })
        
        # 生成关于地震带的问题
        training_data.append({
            "instruction": "回答关于地震的问题",
            "input": "中国哪些地区是地震多发区？",
            "output": "中国的地震多发区主要包括：1. 青藏高原地震区；2. 华北地震区；3. 东南沿海地震带；4. 南北地震带；5. 新疆地震区；6. 台湾地震区。"
        })
        
        # 保存训练数据
        os.makedirs(os.path.dirname(self.config.TRAIN_DATA_FILE), exist_ok=True)
        
        # 随机打乱数据
        random.shuffle(training_data)
        
        # 划分训练集和验证集
        split_idx = int(len(training_data) * 0.8)
        train_data = training_data[:split_idx]
        val_data = training_data[split_idx:]
        
        with open(self.config.TRAIN_DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        
        with open(self.config.VAL_DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, ensure_ascii=False, indent=2)
        
        print(f"生成训练数据完成！")
        print(f"训练数据: {len(train_data)} 条")
        print(f"验证数据: {len(val_data)} 条")
        print(f"保存到: {self.config.TRAIN_DATA_FILE} 和 {self.config.VAL_DATA_FILE}")
        
        return train_data, val_data
    
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
    
    def preprocess_function(self, examples, tokenizer):
        """预处理数据"""
        instructions = examples['instruction']
        inputs = examples['input']
        outputs = examples['output']
        
        # 构建对话格式
        prompts = []
        for instruction, input_text, output_text in zip(instructions, inputs, outputs):
            prompt = f"### 指令:\n{instruction}\n### 输入:\n{input_text}\n### 输出:\n{output_text}"
            prompts.append(prompt)
        
        # 分词
        tokenized = tokenizer(
            prompts,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # 构建标签
        labels = tokenized.input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": tokenized.input_ids,
            "attention_mask": tokenized.attention_mask,
            "labels": labels
        }
    
    def create_peft_model(self, model):
        """创建PEFT模型"""
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        peft_model = get_peft_model(model, lora_config)
        peft_model.print_trainable_parameters()
        return peft_model
    
    def train(self, model, tokenizer, train_data, val_data):
        """训练模型"""
        # 准备数据集
        train_dataset = {
            'instruction': [item['instruction'] for item in train_data],
            'input': [item['input'] for item in train_data],
            'output': [item['output'] for item in train_data]
        }
        
        val_dataset = {
            'instruction': [item['instruction'] for item in val_data],
            'input': [item['input'] for item in val_data],
            'output': [item['output'] for item in val_data]
        }
        
        # 预处理数据
        train_tokenized = self.preprocess_function(train_dataset, tokenizer)
        val_tokenized = self.preprocess_function(val_dataset, tokenizer)
        
        # 构建训练参数
        training_args = TrainingArguments(
            output_dir="./output",
            per_device_train_batch_size=self.config.BATCH_SIZE,
            per_device_eval_batch_size=self.config.BATCH_SIZE,
            learning_rate=self.config.LEARNING_RATE,
            num_train_epochs=self.config.EPOCHS,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            fp16=True
        )
        
        # 构建训练器
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_tokenized,
            eval_dataset=val_tokenized,
            tokenizer=tokenizer
        )
        
        # 开始训练
        print("开始微调模型...")
        trainer.train()
        
        # 保存模型
        print(f"保存微调后的模型到 {self.config.FINETUNED_MODEL_PATH}")
        model.save_pretrained(self.config.FINETUNED_MODEL_PATH)
        tokenizer.save_pretrained(self.config.FINETUNED_MODEL_PATH)
    
    def run(self):
        """运行微调流程"""
        print("开始结合知识图谱的大模型微调...")
        
        # 生成训练数据
        train_data, val_data = self.generate_training_data()
        
        # 加载模型和分词器
        model, tokenizer = self.load_model()
        model.to(self.device)
        
        # 创建PEFT模型
        peft_model = self.create_peft_model(model)
        
        # 开始训练
        self.train(peft_model, tokenizer, train_data, val_data)
        
        print("大模型微调完成！")

if __name__ == "__main__":
    finetuner = KGFinetuner()
    finetuner.run()
