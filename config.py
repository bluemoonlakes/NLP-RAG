# API配置（使用阿里云百炼）
OPENAI_API_KEY = "sk-9063f1a5880f43ebb550d9f60de7e91e"  # 从阿里云百炼获取
OPENAI_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen-max"  # 或 "qwen-plus"
OPENAI_EMBEDDING_MODEL = "text-embedding-v2"

# 数据目录配置
DATA_DIR = "./data"

# 向量数据库配置
VECTOR_DB_PATH = "./vector_db"
COLLECTION_NAME = "course_materials"

# 文本处理配置
CHUNK_SIZE = 500  # 每个文本块的长度
CHUNK_OVERLAP = 50  # 重叠长度
MAX_TOKENS = 2000  # 生成回答的最大长度

# RAG配置
TOP_K = 3  # 每次检索的文档块数量