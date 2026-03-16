#!/usr/bin/env python3
# 将 Ollama 生成的 jsonl 数据转换为监督微调所需的对话格式

import json
import os

SRC_PATH = "data/ollama_finetune_data.jsonl"
TRAIN_OUT = "data/sft_train.jsonl"
VAL_OUT = "data/sft_val.jsonl"


def load_lines(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def convert():
    if not os.path.exists(SRC_PATH):
        raise FileNotFoundError(f"找不到源数据文件: {SRC_PATH}，请先运行 llm/finetune_with_ollama.py 生成数据。")

    lines = load_lines(SRC_PATH)
    if len(lines) < 10:
        print(f"警告：样本数量只有 {len(lines)} 条，可能不足以支撑有效微调。")

    split = int(len(lines) * 0.9)
    train, val = lines[:split], lines[split:]

    def dump(ds, path):
        with open(path, "w", encoding="utf-8") as f:
            for item in ds:
                prompt = item["prompt"]
                answer = item["response"]
                record = {
                    "messages": [
                        {
                            "role": "system",
                            "content": "你是专业的地震学专家，回答要准确、简洁，优先使用给定的地震数据，不要使用***等强调符号。"
                        },
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": answer},
                    ]
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    os.makedirs(os.path.dirname(TRAIN_OUT), exist_ok=True)
    dump(train, TRAIN_OUT)
    dump(val, VAL_OUT)

    print(f"转换完成！train: {len(train)} 条, val: {len(val)} 条")
    print(f"训练集保存到: {TRAIN_OUT}")
    print(f"验证集保存到: {VAL_OUT}")


if __name__ == "__main__":
    convert()

