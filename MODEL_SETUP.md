# 模型设置说明

## 项目概述

本项目实现了一个地震知识图谱与大模型融合的智能问答系统，主要功能包括：
- 知识图谱构建与查询
- 大模型智能问答
- 实时地震数据获取
- 数据可视化展示

## 当前模型配置

### 主模型
- **模型名称**：`deepseek-r1:8b`（本地Ollama模型）
- **部署方式**：本地Ollama服务
- **模型来源**：通过Ollama本地安装

### 备用模型
- **简化规则模型**：`SimpleEarthquakeModel`
- **功能**：当Ollama模型不可用时自动降级
- **适用场景**：基本的地震知识问答

## 模型设置步骤

### 1. 安装Ollama

1. **下载并安装Ollama**：
   - 访问 [Ollama官网](https://ollama.com)
   - 根据操作系统下载并安装

2. **安装deepseek-r1:8b模型**：
   ```bash
   ollama pull deepseek-r1:8b
   ```

### 2. 配置文件设置

**配置文件**：`config/config.py`

```python
# 大模型配置
MODEL_NAME = "THUDM/glm-5-1.8b"  # 备用模型，通过Hugging Face加载
FINETUNED_MODEL_PATH = "llm/finetuned_model"

# 部署配置
DEPLOY_HOST = "0.0.0.0"
DEPLOY_PORT = 8000
```

### 3. 本地Ollama配置

**app.py中的Ollama配置**：

```python
# Ollama模型配置
OLLAMA_MODEL = "deepseek-r1:8b"

# 测试Ollama连接
def test_ollama_connection():
    try:
        response = ollama.generate(model=OLLAMA_MODEL, prompt="你好", stream=False)
        print("Ollama连接成功!")
        return True
    except Exception as e:
        print(f"Ollama连接失败: {e}")
        return False
```

## 模型融合流程

### 1. 知识图谱与LLM融合

```python
def generate_response(instruction, input_text):
    try:
        # 1. 智能实体识别和多维度知识图谱查询
        kg_context = ""
        
        # 地区识别和查询
        regions = ["四川", "云南", "青海", ...]  # 32个地区
        matched_region = None
        for region in regions:
            if region in input_text:
                matched_region = region
                break
        
        # 多维度查询
        if matched_region:
            region_results = kg.query_earthquakes_by_region(matched_region)
            if region_results:
                eq = region_results[0]
                kg_context += f"【知识图谱信息】\n"
                kg_context += f"地区：{eq['location']}\n"
                kg_context += f"时间：{eq['time']}\n"
                kg_context += f"震级：{eq['magnitude']}级\n"
                kg_context += f"深度：{eq['depth']}公里\n"
                kg_context += f"烈度：{eq['intensity']}\n"
                kg_context += f"描述：{eq['description']}\n\n"
        
        # 2. 构建结构化提示词
        prompt = f"你是一个地震知识专家，请根据以下知识图谱信息回答问题：\n\n"
        if kg_context:
            prompt += f"{kg_context}\n"
        prompt += f"【问题】\n{input_text}\n\n"
        prompt += "请基于上述信息提供准确、专业的回答。如果知识图谱中没有相关信息，请基于你的知识提供合理的回答。"
        
        # 3. 调用Ollama模型
        response = ollama.generate(
            model=OLLAMA_MODEL,
            prompt=prompt,
            stream=False,
            options={
                "temperature": 0.7,
                "top_p": 0.95,
                "max_tokens": 512
            }
        )
        
        return response.get("response", "").strip()
    except Exception as e:
        print(f"Ollama模型推理错误: {e}")
        # 降级到简化模型
        return simple_model.generate_response(input_text)
```

## 实时数据集成

### 1. 数据来源
- **主要来源**：USGS（美国地质调查局）地震数据API
- **数据范围**：全球范围内的地震数据
- **数据更新**：每小时自动更新

### 2. 数据获取脚本
- **文件**：`scripts/fetch_real_earthquake_data.py`
- **功能**：从USGS API获取过去30天的地震数据
- **数据量**：最多1000条记录

### 3. 定时任务
- **文件**：`scripts/scheduler.py`
- **功能**：每小时自动更新地震数据
- **运行方式**：后台运行

## 部署与运行

### 1. 环境配置

```bash
# 创建虚拟环境
python3 -m venv .venv

# 激活虚拟环境
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 安装Ollama客户端
pip install ollama

# 安装定时任务库
pip install schedule
```

### 2. 启动服务

```bash
# 启动定时任务（后台运行）
source .venv/bin/activate && python scripts/scheduler.py &

# 启动主应用
./start.sh
# 或
python app.py
```

### 3. 访问应用

打开浏览器访问：`http://localhost:8000`

## 模型性能与要求

### 1. 硬件要求
- **CPU**：4核以上
- **内存**：8GB以上（推荐16GB）
- **GPU**：可选，有GPU会提高推理速度

### 2. 性能指标
- **响应时间**：通常1-3秒
- **准确率**：基于知识图谱的回答准确率>95%
- **稳定性**：支持并发请求

### 3. 模型文件大小
- **deepseek-r1:8b**：约5GB
- **GLM-5-1.8b**：约3.6GB
- **其他小模型**：通常1-3GB

## 故障排查

### 1. Ollama连接失败

**症状**：启动日志显示"Ollama连接失败"

**解决方案**：
1. 检查Ollama服务是否运行
2. 确认deepseek-r1:8b模型是否已安装
3. 重启Ollama服务：`ollama serve`

### 2. 知识图谱加载失败

**症状**：启动日志显示"数据导入失败"

**解决方案**：
1. 检查data/earthquake_data.json文件是否存在
2. 运行数据生成脚本：`python scripts/fetch_real_earthquake_data.py`
3. 检查数据文件格式是否正确

### 3. 模型推理错误

**症状**：聊天界面显示错误信息

**解决方案**：
1. 检查Ollama服务状态
2. 尝试重启Ollama服务
3. 如果问题持续，系统会自动降级到简化模型

## 扩展与优化

### 1. 模型微调

**文件**：`llm/finetune_with_ollama.py`
**功能**：使用知识图谱数据微调Ollama模型
**运行方式**：
```bash
python llm/finetune_with_ollama.py
```

### 2. 模型切换

**切换到其他Ollama模型**：
```python
# 修改app.py中的配置
OLLAMA_MODEL = "其他模型名称"
```

**支持的Ollama模型**：
- llama3:8b
- gemma:7b
- mistral:7b
- qwen2:7b

### 3. 性能优化

**推荐配置**：
- 使用GPU加速Ollama推理
- 增加系统内存到16GB以上
- 优化知识图谱查询性能

## 应急知识 RAG（可选）

主应用会从 `data/emergency_knowledge.json` 构建句向量索引，默认嵌入模型为 `BAAI/bge-small-zh-v1.5`（见 `config/config.py` 的 `RAG_EMBEDDING_MODEL`）。若 `app.py` 使用离线 Hugging Face 环境，请先在可联网环境下将该模型下载到本机缓存；加载失败时服务仍会启动，仅 `【参考资料】` 显示为「无相关条目」。关闭 RAG：在配置中设 `RAG_ENABLED = False`。

## 总结

本项目成功实现了地震知识图谱与大模型的深度融合，通过以下特点提供优质的地震知识服务：

- **本地化部署**：使用Ollama在本地运行大模型，保证数据隐私
- **实时数据**：从USGS API获取最新地震数据
- **智能融合**：先查询知识图谱获取准确信息，再利用LLM生成自然语言回答
- **用户友好**：直观的可视化界面和聊天交互

通过正确的模型设置和部署，系统可以为用户提供准确、及时的地震知识和信息。