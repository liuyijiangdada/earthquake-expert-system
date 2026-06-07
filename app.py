#!/usr/bin/env python3
# 本地化部署应用

import json
import os
from typing import Optional

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from flask import Flask, request, jsonify, send_from_directory, abort, Response
import logging
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
from services.context_builder import QueryContextBuilder, QueryContextDeps
from core.amap_client import AmapClient
from services.output_enricher import OutputEnricher, merge_media_resources
from services.response_guard import guard_response, strip_eval_prefix

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

context_builder = QueryContextBuilder(
    QueryContextDeps(
        config=config,
        kg=kg,
        emergency_rag=emergency_rag,
        phase_classifier=phase_classifier,
        scheduler=scheduler,
        dynamic_retriever=dynamic_retriever,
        multimodal_output=multimodal_output,
    )
)

amap_client = AmapClient.from_config(config)
if amap_client.available:
    print("高德 Web 服务已启用（静态图/地理编码/步行规划）")
else:
    print("高德 Web 服务未配置 Key，地图使用 OSM/URI 降级（见 .env.example）")

output_enricher = None
if getattr(config, "LAYER3_ENABLED", True):
    output_enricher = OutputEnricher(config, amap_client=amap_client)
    print(f"第三层输出增强已初始化（启用={output_enricher.enabled}）")

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
    "用户：", "用户:", "助手：", "助手:",
    "User:", "user:", "User：", "user：",
    "参考答案", "参考答案：", "参考答案:",
]

_CONVERSATION_DELIMITERS = [
    "Human:", "human:", "Human：", "human：",
    "Assistant:", "assistant:", "Assistant：", "assistant：",
    "用户：", "用户:", "助手：", "助手:",
    "User:", "user:", "User：", "user：",
    "<|im_start|>", "</s>",
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
    text = strip_eval_prefix(text)

    for topic in _UNRELATED_TOPICS:
        pos = text.find(topic)
        if pos != -1:
            logging.warning("检测到无关话题 '%s'，在位置 %d 截断", topic, pos)
            text = text[:pos].strip()
            break

    if not _contains_chinese(text):
        return "抱歉，模型生成了不相关的内容。请重新提问您关心的地震相关问题。"

    return text


def _attach_debug(payload: dict, meta: Optional[dict]) -> dict:
    """按 DEBUG_PAYLOAD_ENABLED 决定是否附带 debug 字段。"""
    if meta is not None and getattr(config, "DEBUG_PAYLOAD_ENABLED", True):
        payload["debug"] = meta
    return payload


def _dynamic_items_from_cache() -> list:
    if dynamic_retriever and getattr(dynamic_retriever, "_cache", None):
        return list(dynamic_retriever._cache.items or [])
    return []


def _apply_layer3_prompt(prompt: str, input_text: str, phase_tag: str, debug_meta: dict) -> str:
    if not output_enricher or not output_enricher.enabled:
        debug_meta.setdefault("layer3_media_pending", [])
        return prompt
    enrichment = output_enricher.enrich(
        input_text,
        phase_tag,
        dynamic_items=_dynamic_items_from_cache(),
    )
    if enrichment.layer3_meta:
        debug_meta["layer3"] = enrichment.layer3_meta
    debug_meta["layer3_media_pending"] = enrichment.media_resources
    if enrichment.prompt_section:
        for needle in ("【问题】\n", "【用户问题】\n"):
            if needle in prompt:
                return prompt.replace(needle, enrichment.prompt_section + needle, 1)
    return prompt


def _finalize_media_resources(debug_meta: dict, input_text: str, phase_tag: str):
    layer3 = debug_meta.pop("layer3_media_pending", [])
    keyword_media = []
    if multimodal_output and multimodal_output.enabled:
        keyword_media = multimodal_output.match_as_dicts(input_text, phase_tag)
    max_total = int(getattr(config, "LAYER3_MEDIA_MAX_TOTAL", 8))
    debug_meta["media_resources"] = merge_media_resources(
        keyword_media, layer3, max_total=max_total
    )


def _fallback_on_model_error(
    debug_meta: Optional[dict],
    input_text: str,
    phase_tag: str,
    error: Exception,
    history=None,
) -> tuple:
    """模型推理失败时降级为 RAG 摘要，仍返回 debug 与多模态资源。"""
    meta = debug_meta or {}
    if not meta.get("rag_fallback_text"):
        try:
            _, meta, phase_tag = context_builder.prepare(
                input_text, for_vision=False, history=history
            )
        except Exception:
            return "模型推理暂时失败，请稍后重试。", None

    fallback = (meta.get("rag_fallback_text") or "").strip()
    if fallback:
        meta["response_fallback"] = "rag"
        meta["response_quality"] = "model_error"
        meta["model_error"] = type(error).__name__
        _finalize_media_resources(meta, input_text, phase_tag)
        return fallback, meta

    return "模型推理暂时失败，请稍后重试。", meta or None


def _run_llm_generate(input_ids, attention_mask):
    """执行生成；采样出现 inf/nan 时自动降级为贪婪解码。"""
    max_new = int(getattr(config, "LLM_MAX_NEW_TOKENS", 384))
    sample = bool(getattr(config, "LLM_DO_SAMPLE", True))
    common = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new,
        pad_token_id=tokenizer.pad_token_id,
    )
    with torch.no_grad():
        if sample:
            try:
                return model.generate(
                    **common,
                    do_sample=True,
                    temperature=float(getattr(config, "LLM_TEMPERATURE", 0.55)),
                    top_p=float(getattr(config, "LLM_TOP_P", 0.88)),
                    repetition_penalty=float(getattr(config, "LLM_REPETITION_PENALTY", 1.15)),
                    no_repeat_ngram_size=int(getattr(config, "LLM_NO_REPEAT_NGRAM_SIZE", 4)),
                )
            except RuntimeError as e:
                msg = str(e).lower()
                if "inf" not in msg and "nan" not in msg:
                    raise
                logging.warning("采样生成失败(%s)，降级为贪婪解码", e)
        return model.generate(
            **common,
            do_sample=False,
            repetition_penalty=float(getattr(config, "LLM_REPETITION_PENALTY", 1.15)),
        )


def generate_response(instruction, input_text, history=None):
    debug_meta = None
    phase_tag = ""
    try:
        prompt, debug_meta, phase_tag = context_builder.prepare(
            input_text, for_vision=False, history=history
        )
        prompt = _apply_layer3_prompt(prompt, input_text, phase_tag, debug_meta)

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
            outputs = _run_llm_generate(input_ids, attention_mask)

        generated = outputs[0][input_ids.shape[-1]:]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()

        text = text.replace("</s>", "").replace("<|im_start|>", "").replace("<|im_end|>", "")
        text = text.strip()

        text = _sanitize_response(text, input_text)
        text = guard_response(text, debug_meta)

        _finalize_media_resources(debug_meta, input_text, phase_tag)

        return text, debug_meta
    except Exception as e:
        print(f"模型推理错误: {e}")
        return _fallback_on_model_error(debug_meta, input_text, phase_tag, e, history=history)


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

        history = context_builder.normalize_history(params.get("history"))
        response, meta = generate_response(
            "回答用户关于地震的问题", input_text, history=history
        )
        return jsonify(_attach_debug({"response": response}, meta))

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
    return context_builder.normalize_history(data)


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
        context_prompt, debug_meta, phase_tag = context_builder.prepare(
            input_text, for_vision=True, history=chat_history
        )
        context_prompt = _apply_layer3_prompt(
            context_prompt, input_text, phase_tag, debug_meta
        )
        response = qwen_vl_handler.generate(image, context_prompt)
        response = _sanitize_response(response, input_text)
        response = guard_response(response, debug_meta)

        _finalize_media_resources(debug_meta, input_text, phase_tag)

        return jsonify(_attach_debug({"response": response}, debug_meta))
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


_EMERGENCY_MEDIA_DIR = os.path.join(_APP_ROOT, "static", "emergency")
_GENERATED_MEDIA_DIR = os.path.join(_APP_ROOT, "static", "generated")


@app.route("/emergency-media/<path:filename>")
def emergency_media(filename):
    path = os.path.join(_EMERGENCY_MEDIA_DIR, filename)
    if not os.path.isfile(path):
        abort(404)
    return send_from_directory(_EMERGENCY_MEDIA_DIR, filename)


@app.route("/api/amap/static-map")
def amap_static_map_proxy():
    """代理高德静态图，避免在前端暴露 Web 服务 Key。"""
    if not amap_client.available:
        return jsonify({
            "error": "高德 Web 服务未配置，请在项目根目录 .env 设置 AMAP_WEB_SERVICE_KEY",
        }), 503
    if not getattr(config, "AMAP_STATIC_MAP_PROXY_ENABLED", True):
        return jsonify({"error": "静态图代理已关闭"}), 503
    try:
        lon = float(request.args.get("lon", ""))
        lat = float(request.args.get("lat", ""))
        zoom = int(request.args.get("zoom", 10))
    except (TypeError, ValueError):
        return jsonify({"error": "参数 lon/lat/zoom 格式错误"}), 400

    if not (3 <= zoom <= 18):
        return jsonify({"error": "zoom 须在 3–18 之间"}), 400
    if not (73 <= lon <= 135 and 18 <= lat <= 54):
        return jsonify({"error": "坐标超出服务范围"}), 400

    markers = None
    mlon = request.args.get("mlon")
    mlat = request.args.get("mlat")
    if mlon is not None and mlat is not None:
        try:
            markers = [(float(mlon), float(mlat), "B")]
        except ValueError:
            return jsonify({"error": "mlon/mlat 格式错误"}), 400

    try:
        img = amap_client.fetch_static_map(lon, lat, zoom=zoom, markers=markers)
        return Response(img, mimetype="image/png")
    except Exception as e:
        logging.warning("高德静态图代理失败: %s", e)
        return jsonify({"error": f"静态图获取失败：{type(e).__name__}"}), 502


@app.route("/generated-media/<path:filename>")
def generated_media(filename):
    if ".." in filename or filename.startswith("/"):
        abort(404)
    path = os.path.join(_GENERATED_MEDIA_DIR, filename)
    if not os.path.isfile(path):
        abort(404)
    return send_from_directory(_GENERATED_MEDIA_DIR, filename)


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


@app.route("/api/eval-set", methods=["GET"])
def eval_set():
    """返回分阶段评测问集（论文实验用）。"""
    rel = getattr(config, "EVAL_QUESTIONS_FILE", "data/eval/phase_questions.json")
    path = rel if os.path.isabs(rel) else os.path.join(_APP_ROOT, rel)
    if not os.path.isfile(path):
        return jsonify({"error": f"评测集文件不存在：{path}"}), 404
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        return jsonify({"error": f"评测集读取失败：{type(e).__name__}"}), 500

    phase = (request.args.get("phase") or "").strip()
    phases = data.get("phases", {})
    if phase:
        if phase not in phases:
            return jsonify({
                "error": f"未知阶段：{phase}，可选：{', '.join(phases.keys())}",
            }), 400
        return jsonify({
            "version": data.get("version"),
            "phase": phase,
            "count": len(phases[phase]),
            "questions": phases[phase],
        })
    return jsonify({
        "version": data.get("version"),
        "description": data.get("description"),
        "phases": {k: len(v) for k, v in phases.items()},
        "questions_by_phase": phases,
    })


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
