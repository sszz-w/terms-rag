"""Web API 服务：基于 FastAPI 提供条款检索 HTTP 接口"""
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import TOP_K
from embedding import embed_texts
from query import get_collection

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


class MatchResult(BaseModel):
    similarity: float
    original_clause: str
    certificate_name: str
    content_elements: str
    validity_rules: str


class QueryResponse(BaseModel):
    query: str
    results: list[MatchResult]


@app.post("/query", response_model=QueryResponse)
def query_clauses(req: QueryRequest):
    collection = get_collection()
    query_embedding = embed_texts([req.text])
    results = collection.query(query_embeddings=query_embedding, n_results=req.top_k)

    matches = []
    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]
        similarity = round(1 - distance, 4)
        matches.append(MatchResult(
            similarity=similarity,
            original_clause=results["documents"][0][i],
            certificate_name=results["metadatas"][0][i]["证书报告名称"],
            content_elements=results["metadatas"][0][i]["内容要素字段"],
            validity_rules=results["metadatas"][0][i]["有效性判定"],
        ))

    return QueryResponse(query=req.text, results=matches)


@app.get("/health")
def health():
    return {"status": "ok"}
