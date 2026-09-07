#!/usr/bin/env python3
# 使用 Qwen2.5-7B-Instruct 进行 LoRA 微调（MPS / 单卡可跑；QLoRA 4-bit 需 CUDA+bitsandbytes）。

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, PeftModel, get_peft_model

# 将项目根目录加入 Python 路径，方便导入 config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

cfg = Config()
MODEL_NAME = cfg.MODEL_NAME
OUTPUT_DIR = cfg.FINETUNED_MODEL_PATH


def _local_snapshot(model_name: str) -> str:
    hub = (
        Path.home()
        / ".cache/huggingface/hub"
        / f"models--{model_name.replace('/', '--')}"
        / "snapshots"
    )
    if hub.is_dir():
        snaps = sorted(p for p in hub.iterdir() if p.is_dir())
        if snaps:
            return str(snaps[-1])
    return model_name

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
        # 使用 Qwen2.5 官方 chat 模板（与评测生成端一致），
        # 旧版手动拼接 <|system|>/<|user|>/<|assistant|> 在 Qwen2.5 词表里并非特殊 token，
        # 会导致遮罩失败、把 prompt 也算进 loss。
        chat = [
            {"role": m["role"], "content": m["content"]}
            for m in msgs
            if m["role"] in ("system", "user", "assistant")
        ]
        full_text = self.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=False
        )
        enc = self.tokenizer(
            full_text,
            max_length=self.max_len,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = enc.input_ids[0]
        labels = input_ids.clone()

        # 只计算助手回答部分的损失：把 system+user（含生成提示）遮罩为 -100
        prompt_msgs = [m for m in chat if m["role"] != "assistant"]
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )
        prompt_len = len(
            self.tokenizer(prompt_text, return_tensors="pt").input_ids[0]
        )
        if prompt_len <= len(labels):
            labels[:prompt_len] = -100

        return {"input_ids": input_ids, "labels": labels}


def collate_fn(batch):
    input_ids = [b["input_ids"] for b in batch]
    labels = [b["labels"] for b in batch]

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

    return {"input_ids": input_ids, "labels": labels}


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--extra-epochs", type=int, default=2, help="在已有 LoRA 上继续训练的 epoch 数")
    ap.add_argument("--from-scratch", action="store_true", help="忽略已有 adapter，重新初始化 LoRA")
    args_cli = ap.parse_args()

    local_model = _local_snapshot(MODEL_NAME)
    print(f"加载基础模型: {local_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        local_model, use_fast=False, trust_remote_code=False, local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 在 Mac 上使用 MPS / CPU，避免依赖 bitsandbytes 和 CUDA
    dtype = torch.float16 if (torch.cuda.is_available() or torch.backends.mps.is_available()) else torch.float32
    device_map = "mps" if torch.backends.mps.is_available() else "auto"

    model = AutoModelForCausalLM.from_pretrained(
        local_model,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=False,
        local_files_only=True,
    )

    lora_config = LoraConfig(
        r=16,  # 7B 档建议 16~32；容量与显存折中
        lora_alpha=32,  # 通常取 r 的 2 倍
        # Qwen2.5 全部线性层都挂 LoRA，比只挂 q/v 吸收更充分
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    adapter_cfg = os.path.join(OUTPUT_DIR, "adapter_config.json")
    continue_adapter = (not args_cli.from_scratch) and os.path.isfile(adapter_cfg)
    if continue_adapter:
        print(f"从已有 LoRA 继续训练: {OUTPUT_DIR}")
        model = PeftModel.from_pretrained(model, OUTPUT_DIR, is_trainable=True)
        epochs = max(1, args_cli.extra_epochs)
        lr = 5e-6
    else:
        model = get_peft_model(model, lora_config)
        epochs = 3
        lr = 2e-4  # 7B + LoRA(r=16) 常用起步 lr；续训分支保持 5e-6

    model.print_trainable_parameters()
    train_ds = SFTDataset(TRAIN_PATH, tokenizer)
    val_ds = SFTDataset(VAL_PATH, tokenizer)

    print(f"训练样本数: train={len(train_ds)}, val={len(val_ds)}, epochs={epochs}, lr={lr}")

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,  # 降低单卡显存压力
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,  # 通过梯度累积等效放大 batch
        num_train_epochs=epochs,
        learning_rate=lr,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=40,
        save_steps=40,
        save_total_limit=3,
        fp16=torch.cuda.is_available(),
        bf16=False,
        report_to=[],
        max_grad_norm=1.0,
        warmup_ratio=0.03,
        dataloader_pin_memory=False,
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

