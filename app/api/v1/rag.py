import logging

from fastapi import APIRouter

from app.agent.tools.request_context import set_perf_request_id
from app.common.response import success
from app.schemas.rag import RAGQueryRequest
from app.services.rag_service import rag_query

logger = logging.getLogger("rag-agent.rag_api")
router = APIRouter()


@router.post("/query")
async def rag_query_endpoint(req: RAGQueryRequest):
    request_id = req.request_id or "rag-" + _generate_short_id()
    set_perf_request_id(request_id)

    result = rag_query(
        query=req.query,
        top_k=req.top_k,
        knowledge_base_id=req.knowledge_base_id,
        request_id=request_id,
    )
    return success(data={"query": req.query, "result": result})


def _generate_short_id() -> str:
    import uuid
    return uuid.uuid4().hex[:12]
