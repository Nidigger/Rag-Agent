"""Agent tools — capabilities available to the ReAct Agent.

Each tool is a LangChain @tool-decorated function that the Agent can
invoke during its reasoning loop. Tools have access to the per-request
context (user_id, month, report flag) via request_context module.
"""

import csv
import logging
import os
import random
from datetime import datetime

from langchain_core.tools import tool

from app.agent.tools.request_context import get_request_context
from app.config import settings
from app.rag.retriever import RagSummarizeService
from app.utils.path_tool import get_abs_path
from app.utils.perf import elapsed_ms, log_perf, now_ms

logger = logging.getLogger("rag-agent.agent_tools")

_rag_service: RagSummarizeService | None = None
external_data: dict = {}

user_ids = [
    "1001", "1002", "1003", "1004", "1005",
    "1006", "1007", "1008", "1009", "1010",
]


def _get_rag_service() -> RagSummarizeService:
    """Lazy-initialize the RAG service singleton."""
    global _rag_service
    if _rag_service is None:
        _rag_service = RagSummarizeService()
    return _rag_service


@tool(description="从向量存储中检索参考资料")
def rag_summarize(query: str) -> str:
    """Query the vector store for relevant document chunks."""
    request_id = get_request_context().get("request_id", "internal")

    logger.info("[tool] rag_summarize: query=%r", query[:80])

    tool_start = now_ms()
    log_perf("rag_tool", "start",
             request_id=request_id,
             query=query[:80],
             top_k=3,
             kb="kb_default")

    result = _get_rag_service().rag_summarize(query=query, request_id=request_id)

    tool_elapsed = elapsed_ms(tool_start)
    log_perf("rag_tool", "done",
             request_id=request_id,
             elapsed_ms=tool_elapsed,
             result_len=len(result))

    return result


@tool(description="获取指定城市的天气，以消息字符串的形式返回")
def get_weather(city: str) -> str:
    """Return simulated weather data for the given city."""
    logger.info("[tool] get_weather: city=%s", city)
    return (
        f"城市{city}天气为晴天，气温26摄氏度，空气湿度50%，"
        f"南风1级，AQI21，最近6小时降雨概率极低"
    )


@tool(description="获取用户所在城市的名称，以纯字符串形式返回")
def get_user_location() -> str:
    """Return a simulated user location."""
    location = random.choice(["深圳", "合肥", "杭州"])
    logger.info("[tool] get_user_location: %s", location)
    return location


@tool(description="获取用户的ID，以纯字符串形式返回")
def get_user_id() -> str:
    """Return the user ID from request context, or a random one."""
    ctx = get_request_context()
    if ctx.get("user_id"):
        logger.info("[tool] get_user_id: %s (from context)", ctx["user_id"])
        return ctx["user_id"]
    uid = random.choice(user_ids)
    logger.info("[tool] get_user_id: %s (random)", uid)
    return uid


@tool(description="获取当前月份，以纯字符串形式返回")
def get_current_month() -> str:
    """Return the current month from context or system clock."""
    ctx = get_request_context()
    if ctx.get("month"):
        logger.info("[tool] get_current_month: %s (from context)", ctx["month"])
        return ctx["month"]
    month = datetime.now().strftime("%Y-%m")
    logger.info("[tool] get_current_month: %s (system)", month)
    return month


def _generate_external_data():
    """Load external CSV data into memory (lazy, once)."""
    global external_data
    if external_data:
        return

    path = get_abs_path(settings.rag.external_data_path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"外部数据文件{path}不存在")

    logger.info("[tool] Loading external data from: %s", path)
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            user_id = row["用户ID"]
            month = row["时间"]
            if user_id not in external_data:
                external_data[user_id] = {}
            external_data[user_id][month] = {
                "特征": row["特征"],
                "清洁效率": row["清洁效率"],
                "耗材": row["耗材"],
                "对比": row["对比"],
            }
    logger.info(
        "[tool] External data loaded: %d users", len(external_data)
    )


@tool(
    description="从外部系统中获取指定用户在指定月份的使用记录，"
    "以纯字符串形式返回，如果未检索到返回空字符串"
)
def fetch_external_data(user_id: str, month: str) -> str:
    """Fetch usage records for a user in a given month from CSV data."""
    _generate_external_data()
    try:
        result = str(external_data[user_id][month])
        logger.info(
            "[tool] fetch_external_data: user=%s, month=%s, found",
            user_id,
            month,
        )
        return result
    except KeyError:
        logger.warning(
            "[tool] fetch_external_data: user=%s, month=%s, not found",
            user_id,
            month,
        )
        return ""


@tool(
    description="无入参，无返回值，调用后触发中间件自动为报告生成的场景"
    "动态注入上下文信息，为后续提示词切换提供上下文信息"
)
def fill_context_for_report():
    """Signal middleware to switch to the report system prompt."""
    logger.info("[tool] fill_context_for_report: report mode activated")
    return "fill_context_for_report已调用"
