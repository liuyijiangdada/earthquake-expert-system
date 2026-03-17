#!/usr/bin/env python3
# 本地化部署应用

from flask import Flask, request, jsonify
import os
import sys
import requests

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config.config import Config
from kg.in_memory_kg import InMemoryKG
from llm.simple_model import SimpleEarthquakeModel

app = Flask(__name__)
config = Config()

# 初始化知识图谱
kg = InMemoryKG()
kg.run()

# 初始化简化的大模型
simple_model = SimpleEarthquakeModel()

# 微调后专家模型服务配置（见 llm/serve_earthquake_expert.py）
FINETUNED_MODEL_ENDPOINT = "http://127.0.0.1:9000/generate"

# 测试微调模型服务连接
def test_finetuned_model_connection():
    try:
        resp = requests.get("http://127.0.0.1:9000/health", timeout=5)
        if resp.status_code == 200:
            print("微调模型服务连接成功!")
            return True
        print(f"微调模型服务健康检查失败，状态码: {resp.status_code}")
        return False
    except Exception as e:
        print(f"微调模型服务连接失败: {e}")
        return False

# 模型推理函数（使用Ollama模型）
def generate_response(instruction, input_text):
    try:
        # 1. 智能实体识别和多维度知识图谱查询
        kg_context = ""

        # 提取输入中的关键信息
        input_lower = input_text.lower()
        
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
                import re
                match = re.search(r'(大于|高于)(\d+\.?\d*)', input_text)
                if match:
                    min_mag = float(match.group(2))
                    mag_results = kg.query_earthquakes_by_magnitude(min_mag)
                    if mag_results:
                        kg_context += f"【震级大于{min_mag}级的地震】\n"
                        for i, eq in enumerate(mag_results[:3]):  # 最多显示3个
                            kg_context += f"{i+1}. {eq['location']}：{eq['magnitude']}级 ({eq['time']})\n"
                        kg_context += "\n"
        
        # 3. 构建结构化提示词
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
        
        # 3. 调用微调后的本地专家模型服务
        resp = requests.post(
            FINETUNED_MODEL_ENDPOINT,
            json={
                "prompt": prompt,
                "temperature": 0.7,
                "max_tokens": 512,
            },
            timeout=120,
        )
        if resp.status_code != 200:
            print(f"微调模型服务返回错误状态码: {resp.status_code}")
            return simple_model.generate_response(input_text)

        data = resp.json()
        return data.get("response", "").strip()
    except Exception as e:
        print(f"Ollama模型推理错误: {e}")
        # 降级到简化模型
        return simple_model.generate_response(input_text)

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
    print("测试微调模型服务连接...")
    finetuned_ok = test_finetuned_model_connection()

    if not finetuned_ok:
        print("微调模型服务不可用，将使用简化规则模型 simple_model")
    
    app.run(host=config.DEPLOY_HOST, port=config.DEPLOY_PORT, debug=True)