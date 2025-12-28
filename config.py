"""Configuration file for Medical GraphRAG System"""

import os
from dotenv import load_dotenv

load_dotenv()

# Neo4j Configuration
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password123"

# Ollama Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "qwen2.5:3b"
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 512
# Embedding Configuration
EMBEDDING_MODEL_NAME = "AITeamVN/Vietnamese_Embedding"
EMBEDDING_DIMENSION = 384

# Vector Store Configuration
VECTOR_STORE_TYPE = "weaviate"
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")

# Chunking Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
MAX_TOKENS_PER_CHUNK = 500

# Retrieval Configuration
TOP_K_VECTOR_RESULTS = 10
TOP_K_GRAPH_RESULTS = 10
RERANKING_TOP_K = 5
SIMILARITY_THRESHOLD = 0.5

# Graph Configuration
MAX_GRAPH_DEPTH = 3
MAX_NODES_TO_RETURN = 20

# Clustering Configuration
CLUSTERING_ALGORITHM = "leiden"
MIN_CLUSTER_SIZE = 3
RESOLUTION = 1.0

# File Paths
PDF_PATH = "./data/duoc-thu-quoc-gia-viet-nam-2018.pdf"
DATA_DIR = "./data"
CACHE_DIR = "./cache"
LOGS_DIR = "./logs"

# Processing
BATCH_SIZE = 32
NUM_WORKERS = 4

# API Configuration
API_PORT = 8000
API_HOST = "0.0.0.0"

# Logging
LOG_LEVEL = "INFO"

# Vietnamese NLP
VIETNAMESE_STOPWORDS = {
    "và", "hoặc", "nhưng", "mà", "nếu", "thì", "vì", "do", "nên",
    "là", "có", "được", "cho", "để", "từ", "tại", "với", "không",
    "cũng", "chỉ", "đã", "sẽ", "đang", "quá", "rất", "lại", "như",
    "này", "kia", "nước", "phần", "lúc", "cái", "chiếc", "những"
}

# Medical-specific tokens
MEDICAL_ENTITIES = {
    "DRUG": "Thuốc",
    "INDICATION": "Chỉ định",
    "SIDE_EFFECT": "Tác dụng phụ",
    "DOSAGE": "Liều dùng",
    "CONTRAINDICATION": "Chống chỉ định",
    "ACTIVE_INGREDIENT": "Hoạt chất",
    "DRUG_TYPE": "Loại thuốc",
    "SYMPTOM": "Triệu chứng"
}

# Graph relationship types
RELATIONSHIP_TYPES = {
    "HAS_INDICATION": "Chỉ định",
    "CAUSES_SIDE_EFFECT": "Gây tác dụng phụ",
    "HAS_CONTRAINDICATION": "Chống chỉ định",
    "HAS_DOSAGE": "Liều dùng",
    "INTERACTS_WITH": "Tương tác với",
    "CONTAINS": "Chứa hoạt chất",
    "BELONGS_TO": "Thuộc về",
    "PRECAUTION_FOR": "Cảnh báo cho",
    "SIMILAR_TO": "Tương tự với"
}
