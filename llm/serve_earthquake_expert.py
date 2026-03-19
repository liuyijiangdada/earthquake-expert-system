#!/usr/bin/env python3
# 使用微调后的 DeepSeek R1 蒸馏模型提供本地推理服务

import os
import sys

import torch
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config


cfg = Config()
MODEL_NAME = cfg.MODEL_NAME
MODEL_DIR = cfg.FINETUNED_MODEL_PATH

app = Flask(__name__)

print(f"加载基础模型: {MODEL_NAME}")
# 选择推理设备（优先 MPS，其次 CUDA，最后 CPU）
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

dtype = torch.float16 if device.type in {"cuda", "mps"} else torch.float32

# 注意：这里不要用 device_map="auto"，否则容易触发 meta/offload，
# 进而在 peft/accelerate 组合下出现 unhashable type: 'set' 之类的问题。
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=dtype,
    device_map=None,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

print(f"加载微调模型: {MODEL_DIR}")
# 加载 LoRA 权重
model = PeftModel.from_pretrained(base_model, MODEL_DIR)
model.to(device)
model.eval()



@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/generate", methods=["POST"])
def generate():
    data = request.json or {}
    prompt = data.get("prompt", "")
    max_tokens = int(data.get("max_tokens", 512))
    temperature = float(data.get("temperature", 0.7))

    if not prompt:
        return jsonify({"error": "empty prompt"}), 400

    # 使用与微调时相同的聊天格式
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

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    # 检查 pad_token_id
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.95,
        )

    # 只取新生成的部分
    generated = outputs[0][input_ids.shape[-1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()

    return jsonify({"response": text})


if __name__ == "__main__":
    # 默认在本地 9000 端口提供服务
    app.run(host="127.0.0.1", port=9000)

