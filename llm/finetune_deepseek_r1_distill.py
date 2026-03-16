#!/usr/bin/env python3
# 使用 DeepSeek R1 蒸馏模型进行 QLoRA 微调，显存约 16G 可运行

import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

# 将项目根目录加入 Python 路径，方便导入 config
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

    # 在 Mac 上使用 MPS / CPU，避免依赖 bitsandbytes 和 CUDA
    dtype = torch.float16 if (torch.cuda.is_available() or torch.backends.mps.is_available()) else torch.float32
    device_map = "mps" if torch.backends.mps.is_available() else "auto"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    train_ds = SFTDataset(TRAIN_PATH, tokenizer)
    val_ds = SFTDataset(VAL_PATH, tokenizer)

    print(f"训练样本数: train={len(train_ds)}, val={len(val_ds)}")

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,      # 降低单卡显存压力
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,      # 通过梯度累积等效放大 batch
        num_train_epochs=3,
        learning_rate=1e-4,
        logging_steps=20,
        evaluation_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),     # 只有有 CUDA 时才启用 fp16
        bf16=False,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
    )

    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"微调完成，模型保存到: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

