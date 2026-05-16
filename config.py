from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "条款要素抽取.xlsx"
CHROMA_DIR = BASE_DIR / "chroma_db"
MODEL_CACHE_DIR = BASE_DIR / "model_cache"

EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
COLLECTION_NAME = "clauses"
TOP_K = 3

RERANKER_MODEL = "BAAI/bge-reranker-base"
RERANK_TOP_N = 10
RERANK_ENABLED = True
