#!/usr/bin/env python3
# 使用微调后的 DeepSeek R1 蒸馏模型提供本地推理服务

import os
from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from config.config import Config


cfg = Config()
MODEL_DIR = cfg.FINETUNED_MODEL_PATH

app = Flask(__name__)

print(f"加载微调模型: {MODEL_DIR}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True,
)


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

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)

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

