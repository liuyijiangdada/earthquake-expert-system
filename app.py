#!/usr/bin/env python3
# 本地化部署应用

from flask import Flask, request, jsonify
import os
import re
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 设置环境变量，强制使用离线模式
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config.config import Config
from kg.in_memory_kg import InMemoryKG

app = Flask(__name__)
config = Config()

# 初始化知识图谱
kg = InMemoryKG()
kg.run()

# 微调模型：基础权重 + LoRA
MODEL_NAME = config.MODEL_NAME
MODEL_DIR = config.FINETUNED_MODEL_PATH

print(f"加载基础模型: {MODEL_NAME}")
# 选择推理设备（优先 MPS，其次 CUDA，最后 CPU）
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

dtype = torch.float16 if device.type in {"cuda", "mps"} else torch.float32

# 加载基础模型
try:
    # 直接从本地缓存加载模型，避免网络请求
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=dtype,
        device_map=None,
        trust_remote_code=False,  # 禁用远程代码，避免网络请求
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    
    # 直接从本地缓存加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, 
        trust_remote_code=False,  # 禁用远程代码，避免网络请求
        local_files_only=True,
    )
    
    print(f"加载微调模型: {MODEL_DIR}")
    # 加载 LoRA 权重
    model = PeftModel.from_pretrained(base_model, MODEL_DIR)
    model.to(device)
    model.eval()
    print("微调模型加载成功!")
except Exception as e:
    print(f"模型加载失败: {e}")
    sys.exit(1)

# 模型推理函数
def generate_response(instruction, input_text):
    try:
        # 1. 智能实体识别和多维度知识图谱查询
        kg_context = ""

        # 地区识别和查询
        regions = ["四川", "云南", "青海", "西藏", "新疆", "甘肃", "河北", "台湾", "广东", "辽宁", 
                   "北京", "上海", "江苏", "浙江", "安徽", "福建", "江西", "山东", "河南", "湖北", 
                   "湖南", "广西", "海南", "重庆", "贵州", "陕西", "吉林", "黑龙江", "内蒙古", 
                   "宁夏", "香港", "澳门"]
        
        matched_region = None
        for region in regions:
            if region in input_text:
                matched_region = region
                break
        
        # 多维度查询
        if matched_region:
            # 地区查询
            region_results = kg.query_earthquakes_by_region(matched_region)
            if region_results:
                kg_context += f"【知识图谱信息】\n"
                for i, eq in enumerate(region_results[:3]):  # 显示前3个结果
                    kg_context += f"{i+1}. {eq['location']}地震：\n"
                    kg_context += f"   时间：{eq['time']}\n"
                    kg_context += f"   震级：{eq['magnitude']}级\n"
                    kg_context += f"   深度：{eq['depth']}公里\n"
                    kg_context += f"   烈度：{eq['intensity']}\n"
                    kg_context += f"   描述：{eq['description']}\n\n"
        
        # 震级范围查询
        if "震级" in input_text:
            if "大于" in input_text or "高于" in input_text:
                match = re.search(r'(大于|高于)(\d+\.?\d*)', input_text)
                if match:
                    min_mag = float(match.group(2))
                    mag_results = kg.query_earthquakes_by_magnitude(min_mag)
                    if mag_results:
                        kg_context += f"【震级大于{min_mag}级的地震】\n"
                        for i, eq in enumerate(mag_results[:3]):  # 最多显示3个
                            kg_context += f"{i+1}. {eq['location']}：{eq['magnitude']}级 ({eq['time']})\n"
                        kg_context += "\n"
        
        # 构建结构化提示词
        prompt = f"你是一个地震知识专家，请根据以下知识图谱信息回答问题：\n\n"
        if kg_context:
            prompt += f"{kg_context}\n"
        prompt += f"【问题】\n{input_text}\n\n"
        prompt += "请基于上述信息提供准确、专业的回答。\n"
        prompt += "回答要求：\n"
        prompt += "1. 直接回答问题，不要有任何引言或开场白\n"
        prompt += "2. 优先使用知识图谱中的信息\n"
        prompt += "3. 回答要简洁明了，避免冗长\n"
        prompt += "4. 不要使用任何强调符号如***\n"
        prompt += "5. 如果知识图谱中没有相关信息，请基于你的知识提供合理的回答"
        
        # 直接使用本地微调模型进行推理
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
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
            )

        # 只取新生成的部分
        generated = outputs[0][input_ids.shape[-1]:]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        
        # 过滤掉可能的特殊标记
        text = text.replace("</s>", "").replace("<|system|>", "").replace("<|user|>", "").replace("<|assistant|>", "")
        
        # 确保文本干净
        text = text.strip()
        
        return text
    except Exception as e:
        print(f"模型推理错误: {e}")
        return "模型推理暂时失败，请稍后重试。"

# API路由
@app.route("/api/query", methods=["POST"])
def query():
    data = request.json
    if not data:
        return jsonify({"error": "No input data"}), 400
    
    query_type = data.get("query_type")
    params = data.get("params", {})
    
    if query_type == "kg":
        # 知识图谱查询
        kg_query_type = params.get("type")
        
        if kg_query_type == "by_region":
            region = params.get("region")
            results = kg.query_earthquakes_by_region(region)
            return jsonify({"results": results})
        
        elif kg_query_type == "by_magnitude":
            min_magnitude = float(params.get("min_magnitude", 0))
            max_magnitude = float(params.get("max_magnitude", 10))
            results = kg.query_earthquakes_by_magnitude(min_magnitude, max_magnitude)
            return jsonify({"results": results})
        
        elif kg_query_type == "by_depth":
            min_depth = float(params.get("min_depth", 0))
            max_depth = float(params.get("max_depth", 1000))
            results = kg.query_earthquakes_by_depth(min_depth, max_depth)
            return jsonify({"results": results})
        
        elif kg_query_type == "all":
            results = kg.query_all_earthquakes()
            return jsonify({"results": results})
        
        else:
            return jsonify({"error": "Invalid knowledge graph query type"}), 400
    
    elif query_type == "llm":
        # 大模型查询
        input_text = params.get("input", "")
        if not input_text:
            return jsonify({"error": "No input text"}), 400
        
        response = generate_response("回答用户关于地震的问题", input_text)
        return jsonify({"response": response})
    
    else:
        return jsonify({"error": "Invalid query type"}), 400

# 根路由，返回前端页面
@app.route("/")
def index():
    return app.send_static_file("index.html")

# 实时数据更新API
@app.route("/api/update-data", methods=["POST"])
def update_data():
    """更新地震数据"""
    try:
        # 更新内存知识图谱
        updated_count = kg.update_from_realtime_data()
        
        return jsonify({
            "status": "success",
            "updated": updated_count,
            "message": f"成功更新了 {updated_count} 条地震数据"
        })
    except Exception as e:
        print(f"更新数据错误: {e}")
        return jsonify({
            "status": "error",
            "message": f"更新数据失败: {str(e)}"
        }), 500

if __name__ == "__main__":
    print("启动地震知识图谱和大模型应用...")
    print("启动Flask应用...")
    app.run(host=config.DEPLOY_HOST, port=config.DEPLOY_PORT, debug=True)
