from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "条款要素抽取.xlsx"
CHROMA_DIR = BASE_DIR / "chroma_db"

EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
COLLECTION_NAME = "clauses"
TOP_K = 3
