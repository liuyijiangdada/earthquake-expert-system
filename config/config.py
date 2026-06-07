# 项目配置文件

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass


class Config:
    # 知识图谱配置
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "password"
    
    # 数据配置
    DATA_DIR = "data"
    EARTHQUAKE_DATA_FILE = "data/earthquake_data.csv"
    # 真实地震简录 + 应急知识（优先用于 Neo4j 初始化，见 kg/neo4j_kg.py）
    REAL_EARTHQUAKE_CATALOG_FILE = "data/real_earthquakes_catalog.json"
    EMERGENCY_KNOWLEDGE_FILE = "data/emergency_knowledge.json"
    KG_DATA_FILE = "data/kg_data.json"
    
    # 大模型配置（用于真微调的基础模型）
    # MODEL_NAME = "Qwen/Qwen1.5-4B"
    MODEL_NAME = "Qwen/Qwen1.5-1.8B"

    FINETUNED_MODEL_PATH = "llm/earthquake_expert_deepseek_r1"

    # 推理：与 app.generate 中 getattr 一致，便于调参与文档说明
    LLM_INPUT_MAX_TOKENS = 4096
    LLM_MAX_NEW_TOKENS = 384
    LLM_TEMPERATURE = 0.55
    LLM_TOP_P = 0.88
    LLM_DO_SAMPLE = True
    LLM_REPETITION_PENALTY = 1.15
    LLM_NO_REPEAT_NGRAM_SIZE = 4
    
    # 微调配置
    TRAIN_DATA_FILE = "data/train_data.json"
    VAL_DATA_FILE = "data/val_data.json"
    BATCH_SIZE = 4
    EPOCHS = 3
    LEARNING_RATE = 1e-4
    
    # 部署配置
    DEPLOY_PORT = 8000
    DEPLOY_HOST = "0.0.0.0"
    # False：单进程启动，大模型/RAG 只加载一次；开发时可设环境变量 FLASK_DEBUG=1 或改此处为 True
    FLASK_DEBUG = False
    
    # 爬取配置
    CRAWL_DELAY = 1
    MAX_RETRIES = 3
    
    # 日志配置
    LOG_LEVEL = "INFO"
    LOG_FILE = "logs/app.log"

    # RAG / KG 消融与应急知识向量检索
    KG_CONTEXT_ENABLED = True
    RAG_ENABLED = True
    RAG_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
    RAG_EMBEDDING_LOCAL_FILES_ONLY = True
    RAG_TOP_K = 5
    RAG_MAX_CHUNK_CHARS = 800
    # True：API 响应附带 debug（阶段、RAG 命中、媒体资源等）；False 仅返回 response
    DEBUG_PAYLOAD_ENABLED = True
    # True：将匹配到的示意图摘要写入 LLM prompt（默认 False，仅前端展示）
    MULTIMODAL_INJECT_PROMPT = False
    # True：不连 Milvus，使用内存矩阵（本地开发推荐 True）
    RAG_USE_MEMORY_RAG = False
    # Milvus（与 docker-compose 中 milvus-standalone 一致）
    MILVUS_HOST = "localhost"
    MILVUS_PORT = 19530
    RAG_MILVUS_COLLECTION = "emergency_rag"
    # True：每次启动删表重建（慢）；开发建议 False；无 Docker 时用 RAG_USE_MEMORY_RAG=True
    RAG_MILVUS_REBUILD_ON_START = True

    # 三阶段协同配置
    PHASE_CLASSIFIER_ENABLED = True
    DYNAMIC_RETRIEVAL_ENABLED = True
    DYNAMIC_CACHE_TTL_SECONDS = 300
    DYNAMIC_API_TIMEOUT = 15
    DYNAMIC_MIN_MAGNITUDE = 4.5
    DYNAMIC_MAX_ITEMS = 10
    DYNAMIC_HOURS_WINDOW = 48
    # 动态源优先级：ceic=中国地震台网，usgs=USGS FDSN（CEIC 不可达时自动降级 USGS）
    DYNAMIC_FEED_PROVIDERS = ("ceic", "usgs")
    DYNAMIC_PREFER_CEIC_FOR_CHINA = True
    CEIC_ENABLED = True
    CEIC_BASE_URL = "http://www.ceic.ac.cn"
    # 1=近24h 2=近48h 5=近一年M3+；0 表示按 DYNAMIC_HOURS_WINDOW 自动选择
    CEIC_SPEEDSEARCH_NUM = 0
    USGS_ENABLED = True
    USGS_EVENT_API_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    # USGS 结果限制在中国范围（API bbox + 客户端二次过滤）
    DYNAMIC_CHINA_FILTER_ENABLED = True
    DYNAMIC_CHINA_MIN_LAT = 18.0
    DYNAMIC_CHINA_MAX_LAT = 54.0
    DYNAMIC_CHINA_MIN_LON = 73.0
    DYNAMIC_CHINA_MAX_LON = 135.0
    SCHEDULER_ENABLED = True
    SCHEDULER_DYNAMIC_CONFIDENCE_THRESHOLD = 0.4
    SCHEDULER_STATIC_CONFIDENCE_THRESHOLD = 0.9
    SCHEDULER_URGENCY_HIGH_THRESHOLD = 0.5
    SCHEDULER_URGENCY_CRITICAL_THRESHOLD = 0.7
    VALIDITY_HINT_ENABLED = True
    MULTIMODAL_OUTPUT_ENABLED = True
    MULTIMODAL_MAX_PER_TYPE = 2

    # 多模态输入：Qwen2-VL 端到端（图片 + 文本，懒加载，与文本 LLM 并存）
    QWEN_VL_ENABLED = True
    QWEN_VL_MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
    QWEN_VL_MAX_NEW_TOKENS = 384
    QWEN_VL_TEMPERATURE = 0.5
    QWEN_VL_TOP_P = 0.9
    QWEN_VL_DO_SAMPLE = True
    # 限制视觉 token，避免 MPS/内存爆掉（28 为 Qwen2-VL patch 边长）
    QWEN_VL_MIN_PIXELS = 256 * 28 * 28
    QWEN_VL_MAX_PIXELS = 512 * 28 * 28
    QWEN_VL_MAX_IMAGE_EDGE = 768
    QWEN_VL_MAX_CONTEXT_CHARS = 6000

    # 上传图片压缩（前端 + 服务端）
    UPLOAD_IMAGE_MAX_EDGE = 1280
    UPLOAD_IMAGE_JPEG_QUALITY = 82

    # 多轮对话：注入最近 N 轮 user/assistant 到 prompt（无服务端 session）
    CHAT_HISTORY_MAX_ROUNDS = 3
    CHAT_HISTORY_MAX_CHARS_PER_MSG = 500

    # 高德地图 Web 服务（Key 放 .env：AMAP_WEB_SERVICE_KEY=你的Key）
    # 控制台：https://console.amap.com/dev/key/app  需开通「Web服务」
    AMAP_WEB_SERVICE_KEY = os.environ.get("AMAP_WEB_SERVICE_KEY", "").strip()
    AMAP_WEB_SERVICE_ENABLED = True
    AMAP_API_TIMEOUT = 12
    AMAP_STATIC_MAP_SIZE = "480*280"
    AMAP_COORD_SYSTEM = "gcj02"
    AMAP_STATIC_MAP_PROXY_ENABLED = True

    # 第三层增强：地图/避难所/余震图/政策 RSS
    LAYER3_ENABLED = True
    LAYER3_MAP_ENABLED = True
    LAYER3_SHELTER_ENABLED = True
    LAYER3_AFTERSHOCK_CHART_ENABLED = True
    LAYER3_POLICY_RSS_ENABLED = True
    LAYER3_INJECT_PROMPT = False
    SHELTERS_DATA_FILE = "data/shelters.json"
    GENERATED_MEDIA_DIR = "static/generated"
    GENERATED_MEDIA_URL_PREFIX = "/generated-media"
    LAYER3_AFTERSHOCK_MIN_ITEMS = 2
    LAYER3_POLICY_RSS_TIMEOUT = 12
    LAYER3_POLICY_RSS_CACHE_SECONDS = 3600
    LAYER3_POLICY_RSS_URLS = None
    LAYER3_MEDIA_MAX_TOTAL = 8
    EVAL_QUESTIONS_FILE = "data/eval/phase_questions.json"
