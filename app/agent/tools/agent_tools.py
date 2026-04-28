import csv
import logging
import os
import random
from datetime import datetime

from langchain_core.tools import tool

from app.agent.tools.request_context import get_request_context
from app.core.config import settings
from app.rag.retriever import RagSummarizeService
from app.utils.path_tool import get_abs_path

logger = logging.getLogger("rag-agent.agent_tools")

_rag_service: RagSummarizeService | None = None
external_data: dict = {}

user_ids = [
    "1001", "1002", "1003", "1004", "1005",
    "1006", "1007", "1008", "1009", "1010",
]


def _get_rag_service() -> RagSummarizeService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RagSummarizeService()
    return _rag_service


@tool(description="从向量存储中检索参考资料")
def rag_summarize(query: str) -> str:
    return _get_rag_service().rag_summarize(query)


@tool(description="获取指定城市的天气，以消息字符串的形式返回")
def get_weather(city: str) -> str:
    return (
        f"城市{city}天气为晴天，气温26摄氏度，空气湿度50%，"
        f"南风1级，AQI21，最近6小时降雨概率极低"
    )


@tool(description="获取用户所在城市的名称，以纯字符串形式返回")
def get_user_location() -> str:
    return random.choice(["深圳", "合肥", "杭州"])


@tool(description="获取用户的ID，以纯字符串形式返回")
def get_user_id() -> str:
    ctx = get_request_context()
    if ctx.get("user_id"):
        return ctx["user_id"]
    return random.choice(user_ids)


@tool(description="获取当前月份，以纯字符串形式返回")
def get_current_month() -> str:
    ctx = get_request_context()
    if ctx.get("month"):
        return ctx["month"]
    return datetime.now().strftime("%Y-%m")


def _generate_external_data():
    global external_data
    if external_data:
        return

    path = get_abs_path(settings.EXTERNAL_DATA_PATH)
    if not os.path.exists(path):
        raise FileNotFoundError(f"外部数据文件{path}不存在")

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


@tool(
    description="从外部系统中获取指定用户在指定月份的使用记录，"
    "以纯字符串形式返回，如果未检索到返回空字符串"
)
def fetch_external_data(user_id: str, month: str) -> str:
    _generate_external_data()
    try:
        return str(external_data[user_id][month])
    except KeyError:
        logger.warning(
            f"[fetch_external_data] "
            f"未检索到用户：{user_id}在{month}的使用记录"
        )
        return ""


@tool(
    description="无入参，无返回值，调用后触发中间件自动为报告生成的场景"
    "动态注入上下文信息，为后续提示词切换提供上下文信息"
)
def fill_context_for_report():
    return "fill_context_for_report已调用"
