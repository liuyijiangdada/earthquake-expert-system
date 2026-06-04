#!/usr/bin/env python3
# 本地化部署应用

import json
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from flask import Flask, request, jsonify, send_from_directory, abort
import logging
import re
import sys
import io
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

_APP_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_APP_ROOT)
from config.config import Config
from kg.neo4j_kg import Neo4jKG
from rag.emergency_rag import build_emergency_rag_from_config
from llm.qwen_vl_handler import QwenVLHandler
from core.phase_classifier import PhaseClassifier
from core.dynamic_retriever import DynamicRetriever
from core.scheduler import Scheduler
from core.multimodal_output import MultimodalOutput

app = Flask(__name__)
config = Config()

_SPA_DIR = os.path.join(_APP_ROOT, "static", "spa")
_SPA_INDEX = os.path.join(_SPA_DIR, "index.html")

_MAX_IMAGE_SIZE = 10 * 1024 * 1024
_ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}


def _compress_upload_image(image: Image.Image) -> Image.Image:
    """服务端二次压缩，与 Qwen-VL 输入尺寸对齐。"""
    max_edge = int(getattr(config, "UPLOAD_IMAGE_MAX_EDGE", 1280))
    jpeg_quality = int(getattr(config, "UPLOAD_IMAGE_JPEG_QUALITY", 82))

    image = image.convert("RGB")
    w, h = image.size
    if max(w, h) > max_edge:
        scale = max_edge / float(max(w, h))
        image = image.resize(
            (max(1, int(w * scale)), max(1, int(h * scale))),
            Image.Resampling.LANCZOS,
        )

    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


@app.errorhandler(400)
def bad_request(e):
    return jsonify({"error": f"请求参数错误：{e}"}), 400


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "请求的资源不存在"}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "请求方法不允许"}), 405


@app.errorhandler(413)
def request_too_large(e):
    return jsonify({"error": "上传文件过大，图片最大支持10MB"}), 413


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "服务器内部错误，请稍后重试"}), 500


@app.errorhandler(Exception)
def handle_unexpected(e):
    logging.exception("未捕获异常")
    return jsonify({"error": f"服务异常：{type(e).__name__}"}), 500

kg = Neo4jKG()
kg.run()

MODEL_NAME = config.MODEL_NAME
_fp = config.FINETUNED_MODEL_PATH
MODEL_DIR = _fp if os.path.isabs(_fp) else os.path.join(_APP_ROOT, _fp)


def _tokenizer_load_path() -> str:
    tj = os.path.join(MODEL_DIR, "tokenizer.json")
    tc = os.path.join(MODEL_DIR, "tokenizer_config.json")
    if os.path.isfile(tj) and os.path.isfile(tc):
        try:
            with open(tc, encoding="utf-8") as f:
                tcfg = json.load(f)
            if isinstance(tcfg.get("extra_special_tokens"), list):
                return MODEL_NAME
        except OSError:
            pass
        return MODEL_DIR
    return MODEL_NAME


print(f"加载基础模型: {MODEL_NAME}")
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

dtype = torch.float16 if device.type in {"cuda", "mps"} else torch.float32

try:
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=dtype,
        device_map=None,
        trust_remote_code=False,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )

    _tp = _tokenizer_load_path()
    print(f"加载 tokenizer（路径: {_tp}）", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        _tp,
        trust_remote_code=False,
        local_files_only=True,
    )

    print(f"加载微调模型: {MODEL_DIR}", flush=True)
    if not os.path.isfile(os.path.join(MODEL_DIR, "adapter_config.json")):
        raise FileNotFoundError(
            f"本地 LoRA 目录不存在或缺少 adapter_config.json: {MODEL_DIR} "
            f"（请在项目根目录执行 python app.py，或把 FINETUNED_MODEL_PATH 设为绝对路径）"
        )
    model = PeftModel.from_pretrained(base_model, MODEL_DIR, local_files_only=True)
    model.to(device)
    model.eval()
    print("微调模型加载成功!")
except Exception as e:
    import traceback

    traceback.print_exc()
    print(f"模型加载失败: {e}")
    sys.exit(1)

tokenizer.truncation_side = "left"
_llm_input_max = int(getattr(config, "LLM_INPUT_MAX_TOKENS", 4096))

logging.basicConfig(level=getattr(logging, config.LOG_LEVEL, "INFO"))
emergency_rag = None
if getattr(config, "RAG_ENABLED", True):
    emergency_rag = build_emergency_rag_from_config(config)
    if emergency_rag:
        print("应急知识 RAG 索引加载成功")
    else:
        print("应急知识 RAG 未加载（将使用「无相关条目」占位，可检查嵌入模型缓存与 RAG_ENABLED）")

phase_classifier = None
if getattr(config, "PHASE_CLASSIFIER_ENABLED", True):
    phase_classifier = PhaseClassifier()
    print("三阶段问题分类器已初始化")

dynamic_retriever = None
if getattr(config, "DYNAMIC_RETRIEVAL_ENABLED", True):
    dynamic_retriever = DynamicRetriever(config)
    print(f"动态知识检索模块已初始化（启用={dynamic_retriever.enabled}）")

scheduler = None
if getattr(config, "SCHEDULER_ENABLED", True):
    scheduler = Scheduler(config)
    print("不确定性感知调度器已初始化")

multimodal_output = None
if getattr(config, "MULTIMODAL_OUTPUT_ENABLED", True):
    multimodal_output = MultimodalOutput(config)
    print(f"多模态输出模块已初始化（启用={multimodal_output.enabled}）")

qwen_vl_handler = None
if getattr(config, "QWEN_VL_ENABLED", True):
    qwen_vl_handler = QwenVLHandler(config)
    print(
        f"Qwen-VL 多模态输入已配置（模型={qwen_vl_handler.model_name}，首次图片问答时加载）"
    )


_UNRELATED_TOPICS = [
    "聚氯乙烯", "聚乙烯", "期货", "产能", "开工率", "供需", "下游需求", "库存水平",
    "成本的", "弱势震荡", "PVC", "能源化工", "甲醇", "乙二醇", "纯碱",
]

_FORBIDDEN_OPENERS = [
    "给定", "提出一组", "根据您提供的", "Instructions", "Assistant:",
    "instruction:", "input:", "output:", "Napište", "Zlepšení",
    "Human:", "human:", "Human：", "human：",
    "Assistant:", "assistant:", "Assistant：", "assistant：",
]

_CONVERSATION_DELIMITERS = [
    "Human:", "human:", "Human：", "human：",
    "Assistant:", "assistant:", "Assistant：", "assistant：",
    "###", "---", "===",
]


def _contains_chinese(text: str) -> bool:
    """检查文本是否包含中文"""
    return any("\u4e00" <= c <= "\u9fff" for c in text)


def _sanitize_response(text: str, input_text: str) -> str:
    """过滤模型跑偏输出，确保回答是中文地震相关内容"""
    if not text or len(text) < 5:
        return text

    has_chinese = _contains_chinese(text)
    if not has_chinese:
        logging.warning("模型输出不含中文，触发过滤: %s...", text[:80])
        return "抱歉，模型生成了不相关的内容。请重新提问您关心的地震相关问题。"

    cut_index = len(text)
    for delimiter in _CONVERSATION_DELIMITERS:
        idx = text.find(delimiter)
        if idx != -1 and idx < cut_index:
            cut_index = idx

    if cut_index < len(text):
        logging.warning("检测到多轮对话标记，在位置 %d 截断", cut_index)
        text = text[:cut_index]

    lines = text.split("\n")
    clean_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            clean_lines.append(line)
            continue
        should_skip = False
        for pat in _FORBIDDEN_OPENERS:
            if stripped.startswith(pat):
                should_skip = True
                break
        if not should_skip:
            clean_lines.append(line)

    text = "\n".join(clean_lines).strip()

    for topic in _UNRELATED_TOPICS:
        pos = text.find(topic)
        if pos != -1:
            logging.warning("检测到无关话题 '%s'，在位置 %d 截断", topic, pos)
            text = text[:pos].strip()
            break

    if not _contains_chinese(text):
        return "抱歉，模型生成了不相关的内容。请重新提问您关心的地震相关问题。"

    return text


def _normalize_chat_history(raw, max_rounds: int = None) -> list:
    """解析前端传来的 [{role, content}, ...]，保留最近 max_rounds 轮。"""
    if max_rounds is None:
        max_rounds = int(getattr(config, "CHAT_HISTORY_MAX_ROUNDS", 3))
    max_chars = int(getattr(config, "CHAT_HISTORY_MAX_CHARS_PER_MSG", 500))
    if not raw or not isinstance(raw, list):
        return []

    out = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = (item.get("content") or "").strip()
        if role not in ("user", "assistant") or not content:
            continue
        out.append({"role": role, "content": content[:max_chars]})

    cap = max(0, max_rounds) * 2
    return out[-cap:] if cap else []


def _format_history_section(history: list) -> str:
    if not history:
        return ""
    lines = ["【最近对话（供延续上下文，仅供参考）】"]
    for i, h in enumerate(history, 1):
        label = "用户" if h["role"] == "user" else "助手"
        lines.append(f"{i}. {label}：{h['content']}")
    lines.append("")
    return "\n".join(lines)


def _build_rag_section(input_text: str):
    if not getattr(config, "RAG_ENABLED", True):
        return "（本路径已关闭）", []
    if emergency_rag is None:
        return "（无相关条目）", []
    top_k = getattr(config, "RAG_TOP_K", 5)
    max_chars = getattr(config, "RAG_MAX_CHUNK_CHARS", 800)
    hits = emergency_rag.search(input_text, top_k=top_k)
    if not hits:
        return "（无相关条目）", []
    lines = []
    for i, h in enumerate(hits, 1):
        body = (h.get("text") or "")[:max_chars]
        lines.append(f"{i}. [{h.get('topic_id', '')}] {h.get('title', '')}\n{body}")
    return "\n".join(lines), hits


def _prepare_query_context(
    input_text: str,
    *,
    for_vision: bool = False,
    history: list | None = None,
):
    """组装 KG/RAG/动态检索与阶段调度上下文，供文本 LLM 与 Qwen-VL 共用。"""
    kg_context = ""
    normalized_history = _normalize_chat_history(history)
    debug_meta = {
        "kg_enabled": bool(getattr(config, "KG_CONTEXT_ENABLED", True)),
        "rag_enabled": bool(getattr(config, "RAG_ENABLED", True)),
        "rag_topic_ids": [],
        "phase": "通用",
        "phase_confidence": 0.0,
        "urgency": 0.0,
        "need_dynamic": False,
        "schedule_reasoning": "",
        "multimodal_backend": "qwen_vl" if for_vision else "text_llm",
        "history_rounds": len(normalized_history) // 2,
        "history_messages": len(normalized_history),
    }

    phase_result = None
    schedule_decision = None
    if phase_classifier:
        phase_result = phase_classifier.classify(input_text)
        debug_meta["phase"] = phase_result.phase.value
        debug_meta["phase_confidence"] = round(phase_result.confidence, 2)
        debug_meta["urgency"] = round(phase_result.urgency, 2)
        debug_meta["need_dynamic"] = phase_result.need_dynamic

    if scheduler and phase_result:
        schedule_decision = scheduler.decide(phase_result)
        debug_meta["schedule_reasoning"] = schedule_decision.reasoning

    phase_tag = phase_result.phase.value if phase_result else ""

    if getattr(config, "KG_CONTEXT_ENABLED", True):
        use_kg = schedule_decision.use_kg if schedule_decision else True
        if use_kg:
            regions = [
                "四川", "云南", "青海", "西藏", "新疆", "甘肃", "河北", "台湾", "广东", "辽宁",
                "北京", "上海", "江苏", "浙江", "安徽", "福建", "江西", "山东", "河南", "湖北",
                "湖南", "广西", "海南", "重庆", "贵州", "陕西", "吉林", "黑龙江", "内蒙古",
                "宁夏", "香港", "澳门",
            ]

            matched_region = None
            for region in regions:
                if region in input_text:
                    matched_region = region
                    break

            if matched_region:
                region_results = kg.query_earthquakes_by_region(matched_region)
                if region_results:
                    kg_context += "【知识图谱信息】\n"
                    for i, eq in enumerate(region_results[:3]):
                        kg_context += f"{i+1}. {eq['location']}地震：\n"
                        kg_context += f"   时间：{eq['time']}\n"
                        kg_context += f"   震级：{eq['magnitude']}级\n"
                        kg_context += f"   深度：{eq['depth']}公里\n"
                        kg_context += f"   烈度：{eq['intensity']}\n"
                        kg_context += f"   描述：{eq['description']}\n\n"

            if "震级" in input_text:
                if "大于" in input_text or "高于" in input_text:
                    match = re.search(r"(大于|高于)(\d+\.?\d*)", input_text)
                    if match:
                        min_mag = float(match.group(2))
                        mag_results = kg.query_earthquakes_by_magnitude(min_mag)
                        if mag_results:
                            kg_context += f"【震级大于{min_mag}级的地震】\n"
                            for i, eq in enumerate(mag_results[:3]):
                                kg_context += f"{i+1}. {eq['location']}：{eq['magnitude']}级 ({eq['time']})\n"
                            kg_context += "\n"

            emg = kg.query_emergency_context(input_text, phase_tag=phase_tag)
            if emg:
                kg_context += emg

    if getattr(config, "KG_CONTEXT_ENABLED", True):
        kg_section = kg_context.strip() if kg_context.strip() else "（无）"
    else:
        kg_section = "（本路径已关闭）"

    use_rag = schedule_decision.use_rag if schedule_decision else True
    if use_rag:
        rag_section, rag_hits = _build_rag_section(input_text)
    else:
        rag_section = "（调度器判定本路径无需启用）"
        rag_hits = []
    debug_meta["rag_topic_ids"] = [h.get("topic_id", "") for h in rag_hits]

    dynamic_section = ""
    use_dynamic = schedule_decision.use_dynamic if schedule_decision else False
    if use_dynamic and dynamic_retriever and dynamic_retriever.enabled:
        dynamic_result = dynamic_retriever.fetch_for_phase(phase_tag, input_text)
        dynamic_section = dynamic_result.to_context_text()
        debug_meta["dynamic_source"] = dynamic_result.source
        debug_meta["dynamic_items_count"] = len(dynamic_result.items)

    phase_instruction = ""
    if schedule_decision and schedule_decision.prompt_suffix:
        phase_instruction = schedule_decision.prompt_suffix + "\n"

    validity_hint = ""
    if getattr(config, "VALIDITY_HINT_ENABLED", True) and schedule_decision and schedule_decision.validity_hint:
        fetched_at = ""
        if dynamic_retriever and dynamic_retriever._cache:
            fetched_at = dynamic_retriever._cache.fetched_at
        validity_hint = schedule_decision.validity_hint.format(
            fetched_at=fetched_at or "刚刚",
            refresh_minutes=10,
        )

    prompt = "你是一个地震知识专家。请结合【知识图谱】与【参考资料】回答问题。\n\n"

    if phase_tag and phase_tag != "通用":
        prompt += f"当前判定为【{phase_tag}阶段】的问题。\n\n"

    prompt += f"【知识图谱】\n{kg_section}\n\n"
    prompt += f"【参考资料】\n{rag_section}\n\n"

    if dynamic_section:
        prompt += f"{dynamic_section}\n\n"

    prompt += (
        "规则：数值、时间、震级、地点等可验证事实以知识图谱为准；参考资料仅作步骤与表述补充；"
        "动态信息为实时数据，可能随时更新。"
        "若三者均未提供有效条目，可基于常识回答，并简要说明未命中本地知识库。\n"
    )

    if phase_instruction:
        prompt += f"{phase_instruction}\n"

    if for_vision:
        prompt += (
            "回答要求：\n"
            "1. 请直接观察用户上传的图片，结合上述知识上下文作答\n"
            "2. 优先采用知识图谱中的可验证事实，合理利用参考资料与动态信息\n"
            "3. 回答简洁明了，不要使用强调符号如***\n"
            "4. 若图片与地震应急无关，请说明并引导用户上传相关图片\n"
        )
    else:
        prompt += (
            "回答要求：\n"
            "1. 直接回答问题，不要有任何引言或开场白\n"
            "2. 优先采用知识图谱中的可验证事实，合理利用参考资料与动态信息\n"
            "3. 回答要简洁明了，避免冗长\n"
            "4. 不要使用任何强调符号如***\n"
            "5. 如果知识图谱与参考资料均未提供相关信息，请基于你的知识提供合理回答，并说明未命中本地知识库\n"
        )

    if validity_hint:
        prompt += f"6. 在回答末尾附上时效提示：{validity_hint}\n"

    history_section = _format_history_section(normalized_history)
    if history_section:
        prompt += f"\n{history_section}"

    if for_vision:
        prompt += (
            f"【图片问答】请结合图片内容回答；若与上文对话相关请保持连贯。\n"
            f"【用户问题】\n{input_text}\n"
        )
    else:
        prompt += f"【问题】\n{input_text}\n"

    return prompt, debug_meta, phase_tag


def generate_response(instruction, input_text, history=None):
    try:
        prompt, debug_meta, phase_tag = _prepare_query_context(
            input_text, for_vision=False, history=history
        )

        messages = [
            {"role": "system", "content": "你是一个地震专家，专注于回答地震相关问题。"},
            {"role": "user", "content": prompt}
        ]

        text = ""
        for m in messages:
            role = m["role"]
            content = m["content"]
            if role == "system":
                text += f"<|im_start|>{content}</s>"
            elif role == "user":
                text += f"<|im_start|>user\n{content}</s>"
            elif role == "assistant":
                text += f"<|im_start|>assistant\n{content}</s>"

        text += "<|im_start|>assistant\n"

        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=_llm_input_max,
        )
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=int(getattr(config, "LLM_MAX_NEW_TOKENS", 384)),
                temperature=float(getattr(config, "LLM_TEMPERATURE", 0.55)),
                do_sample=bool(getattr(config, "LLM_DO_SAMPLE", True)),
                top_p=float(getattr(config, "LLM_TOP_P", 0.88)),
                repetition_penalty=float(getattr(config, "LLM_REPETITION_PENALTY", 1.15)),
                no_repeat_ngram_size=int(getattr(config, "LLM_NO_REPEAT_NGRAM_SIZE", 4)),
            )

        generated = outputs[0][input_ids.shape[-1]:]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()

        text = text.replace("</s>", "").replace("<|im_start|>", "").replace("<|im_end|>", "")
        text = text.strip()

        text = _sanitize_response(text, input_text)

        media_resources = []
        if multimodal_output and multimodal_output.enabled:
            media_resources = multimodal_output.match_as_dicts(input_text, phase_tag)
            debug_meta["media_resources"] = media_resources

        return text, debug_meta
    except Exception as e:
        print(f"模型推理错误: {e}")
        return "模型推理暂时失败，请稍后重试。", None


@app.route("/api/query", methods=["POST"])
def query():
    data = request.json
    if not data:
        return jsonify({"error": "请求体为空，请提供 JSON 格式数据"}), 400

    query_type = data.get("query_type")
    if not query_type:
        return jsonify({"error": "缺少 query_type 参数，可选值：llm、kg"}), 400

    params = data.get("params", {})

    if query_type == "kg":
        kg_query_type = params.get("type")
        if not kg_query_type:
            return jsonify({"error": "缺少知识图谱查询类型参数 params.type，可选值：all、by_region、by_magnitude、by_depth"}), 400

        try:
            if kg_query_type == "by_region":
                region = params.get("region")
                if not region:
                    return jsonify({"error": "缺少参数 params.region，请指定查询地区"}), 400
                results = kg.query_earthquakes_by_region(region)
                return jsonify({"results": results})

            elif kg_query_type == "by_magnitude":
                try:
                    min_magnitude = float(params.get("min_magnitude", 0))
                    max_magnitude = float(params.get("max_magnitude", 10))
                except (ValueError, TypeError):
                    return jsonify({"error": "震级参数格式错误，min_magnitude 和 max_magnitude 应为数字"}), 400
                if min_magnitude < 0 or max_magnitude < 0:
                    return jsonify({"error": "震级参数不能为负数"}), 400
                if min_magnitude > max_magnitude:
                    return jsonify({"error": f"最小震级({min_magnitude})不能大于最大震级({max_magnitude})"}), 400
                results = kg.query_earthquakes_by_magnitude(min_magnitude, max_magnitude)
                return jsonify({"results": results})

            elif kg_query_type == "by_depth":
                try:
                    min_depth = float(params.get("min_depth", 0))
                    max_depth = float(params.get("max_depth", 1000))
                except (ValueError, TypeError):
                    return jsonify({"error": "深度参数格式错误，min_depth 和 max_depth 应为数字"}), 400
                if min_depth < 0 or max_depth < 0:
                    return jsonify({"error": "深度参数不能为负数"}), 400
                if min_depth > max_depth:
                    return jsonify({"error": f"最小深度({min_depth})不能大于最大深度({max_depth})"}), 400
                results = kg.query_earthquakes_by_depth(min_depth, max_depth)
                return jsonify({"results": results})

            elif kg_query_type == "all":
                results = kg.query_all_earthquakes()
                return jsonify({"results": results})

            else:
                return jsonify({"error": f"不支持的知识图谱查询类型：{kg_query_type}，可选值：all、by_region、by_magnitude、by_depth"}), 400
        except Exception as e:
            logging.error("知识图谱查询失败: %s", e)
            return jsonify({"error": f"知识图谱查询失败，请确认 Neo4j 服务是否正常运行：{type(e).__name__}"}), 500

    elif query_type == "llm":
        input_text = params.get("input", "")
        if not input_text or not input_text.strip():
            return jsonify({"error": "输入内容不能为空，请输入地震相关问题"}), 400
        if len(input_text) > 2000:
            return jsonify({"error": f"输入内容过长（{len(input_text)}字），请控制在2000字以内"}), 400

        history = _normalize_chat_history(params.get("history"))
        response, meta = generate_response(
            "回答用户关于地震的问题", input_text, history=history
        )
        payload = {"response": response}
        if meta is not None:
            payload["debug"] = meta
        return jsonify(payload)

    else:
        return jsonify({"error": f"不支持的查询类型：{query_type}，可选值：llm、kg"}), 400


def _parse_history_form_field() -> list:
    raw = request.form.get("history", "").strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logging.warning("history 字段 JSON 解析失败")
        return []
    return _normalize_chat_history(data)


@app.route("/api/multimodal-query", methods=["POST"])
def multimodal_query():
    if "image" not in request.files:
        return jsonify({"error": "未检测到图片文件，请选择一张图片上传"}), 400

    image_file = request.files["image"]
    if image_file.filename == "":
        return jsonify({"error": "未选择图片文件，请重新选择"}), 400

    if image_file.content_type not in _ALLOWED_IMAGE_TYPES:
        return jsonify({
            "error": f"不支持的图片格式：{image_file.content_type}，支持格式：JPEG、PNG、GIF、WebP"
        }), 400

    image_data = image_file.read()
    if len(image_data) > _MAX_IMAGE_SIZE:
        return jsonify({
            "error": f"图片文件过大（{len(image_data) // 1024 // 1024}MB），最大支持10MB"
        }), 400

    input_text = request.form.get("input", "")
    if not input_text or not input_text.strip():
        return jsonify({"error": "请输入与图片相关的问题描述"}), 400

    try:
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        raw_kb = len(image_data) // 1024
        image = _compress_upload_image(image)
        logging.info(
            "上传图片已压缩: %dKB → %dx%d",
            raw_kb,
            image.size[0],
            image.size[1],
        )
    except Exception as e:
        return jsonify({"error": f"图片解析失败，请确认文件是否损坏：{type(e).__name__}"}), 400

    if not qwen_vl_handler:
        return jsonify({
            "error": "Qwen-VL 多模态未启用，请在 config.py 中设置 QWEN_VL_ENABLED=True"
        }), 503

    try:
        chat_history = _parse_history_form_field()
        context_prompt, debug_meta, phase_tag = _prepare_query_context(
            input_text, for_vision=True, history=chat_history
        )
        response = qwen_vl_handler.generate(image, context_prompt)
        response = _sanitize_response(response, input_text)

        if multimodal_output and multimodal_output.enabled:
            debug_meta["media_resources"] = multimodal_output.match_as_dicts(
                input_text, phase_tag
            )

        payload = {"response": response, "debug": debug_meta}
        return jsonify(payload)
    except Exception as e:
        logging.exception("Qwen-VL 推理失败: %s", e)
        err = str(e)
        if "Invalid buffer size" in err or "out of memory" in err.lower():
            hint = "图片分辨率或显存占用过高，已自动限制尺寸；请重启服务后重试，或换更小图片。"
        elif "本地未找到模型" in err:
            hint = err
        else:
            hint = (
                f"{type(e).__name__}: {err[:200]}。"
                "请确认已安装 qwen-vl-utils、torchvision，并已下载 Qwen2-VL 模型。"
            )
        return jsonify({"error": f"多模态推理失败：{hint}"}), 500


@app.route("/")
def index():
    if os.path.isfile(_SPA_INDEX):
        return send_from_directory(_SPA_DIR, "index.html")
    return app.send_static_file("index.legacy.html")


@app.route("/assets/<path:filename>")
def spa_assets(filename):
    if not os.path.isfile(_SPA_INDEX):
        abort(404)
    assets_dir = os.path.join(_SPA_DIR, "assets")
    path = os.path.join(assets_dir, filename)
    if not os.path.isfile(path):
        abort(404)
    return send_from_directory(assets_dir, filename)


@app.route("/api/update-data", methods=["POST"])
def update_data():
    try:
        updated_count = kg.update_from_realtime_data()

        if dynamic_retriever:
            dynamic_retriever.invalidate_cache()

        return jsonify({
            "status": "success",
            "updated": updated_count,
            "message": f"成功更新了 {updated_count} 条地震数据"
        })
    except Exception as e:
        logging.error("更新数据错误: %s", e)
        return jsonify({
            "status": "error",
            "message": f"更新数据失败：{type(e).__name__}，请确认 USGS API 是否可达"
        }), 500


@app.route("/api/phase-classify", methods=["POST"])
def phase_classify():
    if not phase_classifier:
        return jsonify({"error": "三阶段分类器未启用，请在配置中开启 PHASE_CLASSIFIER_ENABLED"}), 400
    data = request.json
    if not data or "text" not in data:
        return jsonify({"error": "缺少 text 参数，请提供待分类的文本"}), 400
    text = data["text"]
    if not text or not text.strip():
        return jsonify({"error": "待分类文本不能为空"}), 400
    try:
        result = phase_classifier.classify(text)
        return jsonify({
            "phase": result.phase.value,
            "confidence": result.confidence,
            "urgency": result.urgency,
            "need_dynamic": result.need_dynamic,
            "matched_keywords": result.matched_keywords,
            "reasoning": result.reasoning,
        })
    except Exception as e:
        logging.error("阶段分类失败: %s", e)
        return jsonify({"error": f"阶段分类失败：{type(e).__name__}"}), 500


if __name__ == "__main__":
    print("启动地震知识图谱和大模型应用...")
    print("启动Flask应用...")
    _env_dbg = os.environ.get("FLASK_DEBUG", "").strip().lower() in ("1", "true", "yes")
    _flask_debug = _env_dbg or bool(getattr(config, "FLASK_DEBUG", False))
    app.run(
        host=config.DEPLOY_HOST,
        port=config.DEPLOY_PORT,
        debug=_flask_debug,
        use_reloader=_flask_debug,
    )
