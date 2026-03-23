# 使用 LoRA 微调自己的模型指南

本指南参考 `finetune_deepseek_r1_distill.py` 文件，详细说明如何使用 LoRA 技术微调自己的语言模型。

## 1. 环境准备

### 安装必要的依赖

```bash
pip install torch transformers peft datasets
```

## 2. 数据准备

### 数据格式

创建训练和验证数据文件，格式为 JSONL（每行一个 JSON 对象）：

```json
{"messages": [
  {"role": "system", "content": "你是一个地震专家，专注于回答地震相关问题。"},
  {"role": "user", "content": "什么是地震？"},
  {"role": "assistant", "content": "地震是地球内部能量释放引起的地壳震动现象..."}
]}
```

### 数据文件路径

- 训练数据：`data/sft_train.jsonl`
- 验证数据：`data/sft_val.jsonl`

## 3. 配置文件设置

在 `config/config.py` 中设置模型和训练参数：

```python
class Config:
    # 大模型配置
    MODEL_NAME = "Qwen/Qwen1.5-1.8B"  # 替换为你想使用的基础模型
    FINETUNED_MODEL_PATH = "llm/earthquake_expert_deepseek_r1"  # 微调后模型保存路径
    
    # 微调配置
    BATCH_SIZE = 4
    EPOCHS = 3
    LEARNING_RATE = 1e-4
```

## 4. 代码实现

### 完整的微调脚本

```python
#!/usr/bin/env python3
# 使用 LoRA 微调模型

import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

# 将项目根目录加入 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config


cfg = Config()
MODEL_NAME = cfg.MODEL_NAME
OUTPUT_DIR = cfg.FINETUNED_MODEL_PATH

TRAIN_PATH = "data/sft_train.jsonl"
VAL_PATH = "data/sft_val.jsonl"


@dataclass
class ChatRecord:
    messages: List[Dict[str, str]]


class SFTDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_len: int = 2048):
        self.samples: List[ChatRecord] = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                self.samples.append(ChatRecord(messages=data["messages"]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        msgs = self.samples[idx].messages
        # 简单 chat 模板：[system][user][assistant]
        text = ""
        for m in msgs:
            role = m["role"]
            content = m["content"]
            if role == "system":
                text += f"<|system|>{content}</s>"
            elif role == "user":
                text += f"<|user|>{content}</s>"
            elif role == "assistant":
                text += f"<|assistant|>{content}</s>"

        enc = self.tokenizer(
            text,
            max_length=self.max_len,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = enc.input_ids[0]
        labels = input_ids.clone()
        return {"input_ids": input_ids, "labels": labels}


def collate_fn(batch):
    input_ids = [b["input_ids"] for b in batch]
    labels = [b["labels"] for b in batch]

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

    return {"input_ids": input_ids, "labels": labels}


def main():
    print(f"加载基础模型: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 设备配置
    dtype = torch.float16 if (torch.cuda.is_available() or torch.backends.mps.is_available()) else torch.float32
    device_map = "mps" if torch.backends.mps.is_available() else "auto"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    # LoRA 配置
    lora_config = LoraConfig(
        r=16,  # LoRA 秩，控制可训练参数数量
        lora_alpha=32,  # LoRA 缩放因子
        target_modules=["q_proj", "v_proj"],  # 目标模块
        lora_dropout=0.05,  # Dropout 率
        bias="none",  # 偏置处理方式
        task_type="CAUSAL_LM",  # 任务类型
    )
    model = get_peft_model(model, lora_config)

    # 加载数据
    train_ds = SFTDataset(TRAIN_PATH, tokenizer)
    val_ds = SFTDataset(VAL_PATH, tokenizer)

    print(f"训练样本数: train={len(train_ds)}, val={len(val_ds)}")

    # 训练参数配置
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,  # 降低单卡显存压力
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,  # 通过梯度累积等效放大 batch
        num_train_epochs=3,
        learning_rate=1e-4,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),  # 只有有 CUDA 时才启用 fp16
        bf16=False,
        report_to=[],
    )

    # 创建训练器
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
    )

    # 开始训练
    trainer.train()

    # 保存模型
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"微调完成，模型保存到: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
```

## 5. LoRA 配置参数说明

- `r`: LoRA 秩，控制可训练参数的数量。值越小，参数量越少，训练速度越快，但可能影响性能。
- `lora_alpha`: LoRA 缩放因子，通常设置为 `r` 的 2 倍。
- `target_modules`: 要应用 LoRA 的模块，对于 Transformer 模型，通常选择 `q_proj` 和 `v_proj`。
- `lora_dropout`: Dropout 率，防止过拟合。
- `bias`: 偏置处理方式，可选值为 `none`、`all` 或 `lora_only`。
- `task_type`: 任务类型，对于语言模型通常为 `CAUSAL_LM`。

## 6. 运行微调

### 步骤 1: 准备数据

确保 `data/sft_train.jsonl` 和 `data/sft_val.jsonl` 文件已准备好。

### 步骤 2: 配置模型

在 `config/config.py` 中设置你要使用的基础模型。

### 步骤 3: 运行微调脚本

```bash
python llm/finetune_deepseek_r1_distill.py
```

## 7. 加载和使用微调后的模型

### 加载模型

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# 加载 LoRA 权重
model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL_PATH)

# 合并模型（可选，用于推理加速）
model = model.merge_and_unload()
```

### 示例代码

```python
def generate_response(model, tokenizer, prompt):
    messages = [
        {"role": "system", "content": "你是一个地震专家，专注于回答地震相关问题。"},
        {"role": "user", "content": prompt}
    ]
    
    text = ""
    for m in messages:
        role = m["role"]
        content = m["content"]
        if role == "system":
            text += f"<|system|>{content}</s>"
        elif role == "user":
            text += f"<|user|>{content}</s>"
        elif role == "assistant":
            text += f"<|assistant|>{content}</s>"
    
    # 添加 assistant 标记，表示模型开始生成回答
    text += "<|assistant|>"
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.05
    )
    
    response = tokenizer.decode(output[0], skip_special_tokens=False)
    # 提取 assistant 部分的回答
    assistant_start = response.find("<|assistant|>") + len("<|assistant|")
    assistant_end = response.find("</s>", assistant_start)
    if assistant_end == -1:
        assistant_end = len(response)
    
    return response[assistant_start:assistant_end].strip()

# 使用示例
prompt = "什么是地震？"
response = generate_response(model, tokenizer, prompt)
print(response)
```

## 8. 常见问题和解决方案

### 显存不足

- 减小 `per_device_train_batch_size`
- 增加 `gradient_accumulation_steps`
- 使用较小的模型
- 启用 `fp16` 或 `bf16` 混合精度训练

### 训练速度慢

- 使用 GPU 训练
- 增加 batch size（通过梯度累积）
- 减小 `r` 值，减少可训练参数

### 模型效果不佳

- 增加训练数据量
- 调整学习率和训练轮数
- 优化数据质量和格式
- 调整 LoRA 配置参数

## 9. 总结

使用 LoRA 微调模型的步骤：

1. 准备训练数据（JSONL 格式）
2. 配置模型和训练参数
3. 运行微调脚本
4. 加载和使用微调后的模型

LoRA 技术的优势在于：
- 参数量小，显存占用低
- 训练速度快
- 可移植性强，可以轻松应用到不同模型
- 效果接近全量微调

通过以上步骤，你可以使用 LoRA 技术有效地微调自己的语言模型，适应特定领域的需求。