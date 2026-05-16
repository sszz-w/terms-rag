"""查询模块：输入条款文本，返回最相似条款的关联信息"""
import sys
import chromadb

from config import CHROMA_DIR, COLLECTION_NAME, TOP_K
from embedding import embed_texts


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(name=COLLECTION_NAME)


def query(text: str, top_k: int = TOP_K):
    collection = get_collection()
    query_embedding = embed_texts([text])
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)

    matches = []
    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]
        similarity = 1 - distance
        matches.append({
            "相似度": round(similarity, 4),
            "原始条款": results["documents"][0][i],
            "证书/报告名称": results["metadatas"][0][i]["证书报告名称"],
            "内容要素字段": results["metadatas"][0][i]["内容要素字段"],
            "有效性判定": results["metadatas"][0][i]["有效性判定"],
        })
    return matches


def print_results(matches):
    for i, m in enumerate(matches, 1):
        print(f"\n{'='*60}")
        print(f"匹配 #{i}  (相似度: {m['相似度']})")
        print(f"{'='*60}")
        print(f"原始条款: {m['原始条款']}")
        print(f"\n证书/报告名称: {m['证书/报告名称']}")
        print(f"\n内容要素字段: {m['内容要素字段']}")
        print(f"\n有效性判定:\n{m['有效性判定']}")


def interactive():
    print("条款要素检索系统 (输入 q 退出)")
    print("-" * 40)
    while True:
        text = input("\n请输入条款内容: ").strip()
        if text.lower() == "q":
            break
        if not text:
            continue
        matches = query(text)
        print_results(matches)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        matches = query(text)
        print_results(matches)
    else:
        interactive()
