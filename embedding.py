"""Embedding 模块：使用 fastembed (ONNX Runtime) 运行 bge-small-zh-v1.5"""
from fastembed import TextEmbedding

from config import EMBEDDING_MODEL

_model = None


def get_model() -> TextEmbedding:
    global _model
    if _model is None:
        _model = TextEmbedding(model_name=EMBEDDING_MODEL)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_model()
    embeddings = list(model.embed(texts))
    return [e.tolist() for e in embeddings]
