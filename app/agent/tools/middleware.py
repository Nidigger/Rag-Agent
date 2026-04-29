"""Agent middleware — logging, context injection, tool events, and prompt switching.

Middlewares are executed by the LangChain agent framework at specific
points in the agent loop:

- monitor_tool: wraps every tool call, injects request context,
  emits tool_start/tool_done events via AgentEventSink, and logs
  tool invocation results.
- log_before_model: fires before each LLM call for observability.
- report_prompt_switch: dynamically selects system prompt based on
  whether the request is a report generation or a normal chat.
"""

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
from app.services.stream_events import (
    get_tool_status_message,
    tool_done_event,
    tool_start_event,
)
from app.utils.prompt_loader import load_report_prompts, load_system_prompts

logger = logging.getLogger("rag-agent.middleware")


@wrap_tool_call
def monitor_tool(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    """Wrap every tool call with logging, tool events, and request-context injection.

    - Injects the runtime context (e.g. user_id, month, report flag)
      into thread-local storage so tools can access it.
    - Emits tool_start / tool_done events via AgentEventSink when present.
    - Logs the tool name, arguments, and result status.
    - Sets context["report"] = True when fill_context_for_report is called,
      which triggers the report system prompt in subsequent LLM calls.
    """
    tool_name = request.tool_call["name"]
    tool_args = request.tool_call["args"]
    logger.info("[middleware] Tool call: %s(%s)", tool_name, tool_args)

    # Inject runtime context into thread-local so tools can read it
    ctx = dict(request.runtime.context)
    set_request_context(ctx)

    # Emit tool_start event via event sink (V2)
    sink = ctx.get("event_sink")
    if sink is not None:
        sink.emit(
            tool_start_event(
                tool=tool_name,
                message=get_tool_status_message(tool_name, "start"),
            )
        )

    try:
        result = handler(request)
        logger.info(
            "[middleware] Tool %s completed successfully", tool_name
        )

        # Flag report mode after the context-filling tool is called
        if tool_name == "fill_context_for_report":
            request.runtime.context["report"] = True
            logger.debug("[middleware] Report mode activated")

        # Emit tool_done event via event sink (V2)
        if sink is not None:
            sink.emit(
                tool_done_event(
                    tool=tool_name,
                    message=get_tool_status_message(tool_name, "done"),
                )
            )

        return result
    except Exception as e:
        logger.error(
            "[middleware] Tool %s failed: %s", tool_name, e, exc_info=True
        )
        raise
    finally:
        clear_request_context()


@before_model
def log_before_model(state: AgentState, runtime: Runtime):
    """Log a summary before each LLM call within the agent loop."""
    msg_count = len(state["messages"])
    last_msg = state["messages"][-1]
    logger.info(
        "[middleware] Before model: %d messages, last=%s",
        msg_count,
        type(last_msg).__name__,
    )
    logger.debug(
        "[middleware] Last message content: %s",
        last_msg.content.strip()[:200],
    )
    return None


@dynamic_prompt
def report_prompt_switch(request: ModelRequest):
    """Select the appropriate system prompt based on request context.

    If context["report"] is True (set by the fill_context_for_report tool),
    use the report-specific prompt. Otherwise, use the standard chat prompt.
    """
    is_report = request.runtime.context.get("report", False)
    prompt_type = "report" if is_report else "chat"
    logger.info("[middleware] Prompt selected: %s", prompt_type)

    if is_report:
        return load_report_prompts()
    return load_system_prompts()
