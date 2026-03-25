# 项目配置文件

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
    # 建议使用 DeepSeek R1 的开源蒸馏版本，请在 Hugging Face 上确认最终模型名称
    # 示例： "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    # MODEL_NAME = "Qwen/Qwen1.5-4B"
    MODEL_NAME = "Qwen/Qwen1.5-1.8B"

    FINETUNED_MODEL_PATH = "llm/earthquake_expert_deepseek_r1"
    
    # 微调配置
    TRAIN_DATA_FILE = "data/train_data.json"
    VAL_DATA_FILE = "data/val_data.json"
    BATCH_SIZE = 4
    EPOCHS = 3
    LEARNING_RATE = 1e-4
    
    # 部署配置
    DEPLOY_PORT = 8000
    DEPLOY_HOST = "0.0.0.0"
    
    # 爬取配置
    CRAWL_DELAY = 1
    MAX_RETRIES = 3
    
    # 日志配置
    LOG_LEVEL = "INFO"
    LOG_FILE = "logs/app.log"