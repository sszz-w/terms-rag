"""Web API 服务：基于 FastAPI 提供条款检索 HTTP 接口"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import TOP_K, RERANK_ENABLED, RERANK_TOP_N
from embedding import embed_texts
from query import get_collection
from reranker import rerank

app = FastAPI(title="条款要素检索 RAG 系统", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    text: str
    top_k: int = TOP_K
    rerank: bool = RERANK_ENABLED


class MatchResult(BaseModel):
    similarity: float
    original_clause: str
    certificate_name: str
    content_elements: str
    validity_rules: str


class QueryResponse(BaseModel):
    query: str
    reranked: bool
    results: list[MatchResult]


@app.post("/query", response_model=QueryResponse)
def query_clauses(req: QueryRequest):
    collection = get_collection()

    recall_n = RERANK_TOP_N if req.rerank else req.top_k
    query_embedding = embed_texts([req.text])
    results = collection.query(query_embeddings=query_embedding, n_results=recall_n)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    matches = []
    if req.rerank and len(documents) > 0:
        ranked = rerank(req.text, documents, req.top_k)
        for item in ranked:
            idx = item["index"]
            matches.append(MatchResult(
                similarity=round(item["score"], 4),
                original_clause=documents[idx],
                certificate_name=metadatas[idx]["证书报告名称"],
                content_elements=metadatas[idx]["内容要素字段"],
                validity_rules=metadatas[idx]["有效性判定"],
            ))
    else:
        for i in range(min(req.top_k, len(documents))):
            similarity = round(1 - distances[i], 4)
            matches.append(MatchResult(
                similarity=similarity,
                original_clause=documents[i],
                certificate_name=metadatas[i]["证书报告名称"],
                content_elements=metadatas[i]["内容要素字段"],
                validity_rules=metadatas[i]["有效性判定"],
            ))

    return QueryResponse(query=req.text, reranked=req.rerank, results=matches)


@app.get("/health")
def health():
    return {"status": "ok"}
