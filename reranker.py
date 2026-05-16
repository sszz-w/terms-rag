"""Reranker 模块：使用 Cross-Encoder 对候选结果二次精排"""
from fastembed.rerank.cross_encoder import TextCrossEncoder

from config import RERANKER_MODEL, MODEL_CACHE_DIR

_reranker = None


def get_reranker() -> TextCrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = TextCrossEncoder(model_name=RERANKER_MODEL, cache_dir=str(MODEL_CACHE_DIR))
    return _reranker


def rerank(query: str, documents: list[str], top_k: int) -> list[dict]:
    """对 query-document 对打分并返回按分数降序排列的结果。

    返回: [{"index": 原始索引, "score": 相关性分数}, ...]
    """
    reranker = get_reranker()
    scores = list(reranker.rerank(query, documents))
    ranked = [{"index": i, "score": s} for i, s in enumerate(scores)]
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked[:top_k]
