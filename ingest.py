"""数据导入模块：读取 Excel 并写入 ChromaDB 向量库"""
import pandas as pd
import chromadb

from config import DATA_FILE, CHROMA_DIR, COLLECTION_NAME
from embedding import embed_texts


def load_data():
    df = pd.read_excel(DATA_FILE, engine="openpyxl")
    df = df.fillna("")
    return df


def build_index():
    df = load_data()
    print(f"读取到 {len(df)} 条记录")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    documents = []
    metadatas = []
    ids = []

    for i, row in df.iterrows():
        documents.append(row["原始条款"])
        metadatas.append({
            "证书报告名称": row["证书/报告名称"],
            "内容要素字段": row["内容要素字段"],
            "有效性判定": row["有效性判定"],
        })
        ids.append(f"clause_{i}")

    print("正在生成 embeddings...")
    embeddings = embed_texts(documents)

    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
        embeddings=embeddings,
    )
    print(f"成功导入 {len(documents)} 条条款到向量库")
    return collection


if __name__ == "__main__":
    build_index()
