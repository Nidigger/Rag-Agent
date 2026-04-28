import logging
from typing import Callable

from langchain.agents import AgentState
from langchain.agents.middleware import (
    ModelRequest,
    before_model,
    dynamic_prompt,
    wrap_tool_call,
)
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command

from app.agent.tools.request_context import (
    clear_request_context,
    set_request_context,
)
from app.utils.prompt_loader import load_report_prompts, load_system_prompts

logger = logging.getLogger("rag-agent.middleware")


@wrap_tool_call
def monitor_tool(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    logger.info(f"[tool monitor] 执行工具：{request.tool_call['name']}")
    logger.info(f"[tool monitor] 传入参数：{request.tool_call['args']}")

    # Inject runtime context into thread-local so tools can read it
    ctx = dict(request.runtime.context)
    set_request_context(ctx)

    try:
        result = handler(request)
        logger.info(f"[tool monitor] 工具{request.tool_call['name']}调用成功")

        if request.tool_call["name"] == "fill_context_for_report":
            request.runtime.context["report"] = True

        return result
    except Exception as e:
        logger.error(
            f"工具{request.tool_call['name']}调用失败，原因：{str(e)}"
        )
        raise
    finally:
        clear_request_context()


@before_model
def log_before_model(state: AgentState, runtime: Runtime):
    logger.info(
        f"[log_before_model] 即将调用模型，"
        f"带有{len(state['messages'])}条消息。"
    )
    logger.debug(
        f"[log_before_model] "
        f"{type(state['messages'][-1]).__name__} | "
        f"{state['messages'][-1].content.strip()}"
    )
    return None


@dynamic_prompt
def report_prompt_switch(request: ModelRequest):
    is_report = request.runtime.context.get("report", False)
    if is_report:
        return load_report_prompts()
    return load_system_prompts()
