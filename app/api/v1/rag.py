import logging

from fastapi import APIRouter

from app.common.response import success
from app.schemas.rag import RAGQueryRequest
from app.services.rag_service import rag_query

logger = logging.getLogger("rag-agent.rag_api")
router = APIRouter()


@router.post("/query")
async def rag_query_endpoint(req: RAGQueryRequest):
    result = rag_query(query=req.query, top_k=req.top_k)
    return success(data={"query": req.query, "result": result})
