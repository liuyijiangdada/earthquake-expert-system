#!/usr/bin/env python3
# 将训练数据转换为 JSONL 格式

import json
import os

# 输入输出文件路径
TRAIN_INPUT = "data/train_data.json"
VAL_INPUT = "data/val_data.json"
TRAIN_OUTPUT = "data/sft_train.jsonl"
VAL_OUTPUT = "data/sft_val.jsonl"

# 系统提示
SYSTEM_PROMPT = "你是一个地震知识专家，根据提供的信息回答关于地震的问题。"

def convert_to_jsonl(input_file, output_file):
    """将 JSON 格式转换为 JSONL 格式"""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            # 构建消息格式
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item["input"]},
                {"role": "assistant", "content": item["output"]}
            ]
            # 写入 JSONL 格式
            json.dump({"messages": messages}, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"转换完成: {input_file} -> {output_file}")
    print(f"处理了 {len(data)} 条数据")

def main():
    print("开始转换数据格式...")
    
    # 转换训练数据
    convert_to_jsonl(TRAIN_INPUT, TRAIN_OUTPUT)
    
    # 转换验证数据
    convert_to_jsonl(VAL_INPUT, VAL_OUTPUT)
    
    print("数据转换完成！")

if __name__ == "__main__":
    main()
