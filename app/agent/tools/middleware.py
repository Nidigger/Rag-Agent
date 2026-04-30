"""Agent middleware — logging, context injection, tool events, and prompt switching.

Middlewares are executed by the LangChain agent framework at specific
points in the agent loop:

- monitor_tool: wraps every tool call, injects request context,
  emits tool_start/tool_done events via AgentEventSink, and logs
  tool invocation results with elapsed times.
- log_before_model: fires before each LLM call for observability,
  records model start timing for downstream duration tracking.
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
    clear_perf_request_id,
    clear_request_context,
    get_request_context,
    set_perf_request_id,
    set_request_context,
)
from app.services.stream_events import (
    get_tool_status_message,
    tool_done_event,
    tool_start_event,
)
from app.utils.perf import elapsed_ms, log_perf, now_ms
from app.utils.prompt_loader import load_report_prompts, load_system_prompts

logger = logging.getLogger("rag-agent.middleware")


def emit_pending_model_done(context: dict) -> None:
    """Emit a pending model done log from a plain context dict.

    Used by ``ReactAgent.execute()`` after ``agent.invoke()`` returns
    so that the final model round (which has no subsequent tool call)
    still produces a matching ``[perf][agent_model] done`` event.

    Safe to call multiple times — clears ``_model_start`` on success.
    """
    if "_model_start" not in context:
        return

    model_elapsed = elapsed_ms(context.pop("_model_start"))
    step = context.pop("_perf_step", 0)
    request_id = context.get("request_id", "internal")

    log_perf("agent_model", "done",
             request_id=request_id,
             status="success",
             step=step,
             elapsed_ms=model_elapsed)


def _emit_model_done(runtime: Runtime) -> None:
    """Emit a model done perf log if a model start was recorded.

    Reads ``_model_start`` and ``_perf_step`` from the runtime context,
    computes elapsed time, and logs the event. Clears the tracking
    values afterwards to prevent double-emit.

    This is the runtime-context wrapper called from monitor_tool before
    each tool invocation.
    """
    emit_pending_model_done(runtime.context)


@wrap_tool_call
def monitor_tool(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    """Wrap every tool call with logging, tool events, and request-context injection.

    - Injects the runtime context (e.g. user_id, month, report flag)
      into thread-local storage so tools can access it.
    - Emits tool_start / tool_done events via AgentEventSink when present.
    - Logs performance metrics with elapsed_ms per tool call.
    - Propagates ``request_id`` to the worker thread so deep layers
      (embedding, Qdrant) can read it via ``get_perf_request_id()``.
    - Sets context["report"] = True when fill_context_for_report is called,
      which triggers the report system prompt in subsequent LLM calls.
    """
    tool_name = request.tool_call["name"]
    tool_args = request.tool_call["args"]
    logger.info("[middleware] Tool call: %s(%s)", tool_name, tool_args)

    # Inject runtime context into thread-local so tools can read it
    ctx = dict(request.runtime.context)
    set_request_context(ctx)

    request_id = ctx.get("request_id", "internal")
    set_perf_request_id(request_id)

    query_trunc = str(tool_args.get("query", ""))[:80] if isinstance(tool_args, dict) else ""

    # Emit model_done before tool_start so the sequence is:
    #   model start -> model done -> tool start -> tool done
    _emit_model_done(request.runtime)

    log_perf("agent_tool", "start",
             request_id=request_id,
             status="start",
             tool_name=tool_name,
             query=query_trunc if query_trunc else None)

    # Emit tool_start event via event sink (V2)
    sink = ctx.get("event_sink")
    if sink is not None:
        sink.emit(
            tool_start_event(
                tool=tool_name,
                message=get_tool_status_message(tool_name, "start"),
            )
        )

    tool_start = now_ms()

    try:
        result = handler(request)
        tool_elapsed = elapsed_ms(tool_start)
        logger.info(
            "[middleware] Tool %s completed successfully (elapsed=%dms)",
            tool_name,
            tool_elapsed,
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

        log_perf("agent_tool", "done",
                 request_id=request_id,
                 status="success",
                 tool_name=tool_name,
                 elapsed_ms=tool_elapsed)

        return result
    except Exception as e:
        tool_elapsed = elapsed_ms(tool_start)
        log_perf("agent_tool", "error",
                 request_id=request_id,
                 status="failed",
                 tool_name=tool_name,
                 elapsed_ms=tool_elapsed,
                 error_code="TOOL_ERROR",
                 error=str(e)[:80])
        logger.error(
            "[middleware] Tool %s failed after %dms: %s",
            tool_name,
            tool_elapsed,
            e,
            exc_info=True,
        )
        raise
    finally:
        clear_request_context()
        clear_perf_request_id()


@before_model
def log_before_model(state: AgentState, runtime: Runtime):
    """Log a summary before each LLM call within the agent loop.

    Records the model start timestamp and step number in the runtime
    context so that ``monitor_tool`` (or ``emit_pending_model_done``
    from ``ReactAgent.execute``) can compute the model elapsed time
    and emit ``[perf][agent_model] done``.
    """
    msg_count = len(state["messages"])
    last_msg = state["messages"][-1]
    request_id = runtime.context.get("request_id", "internal")

    # Track step count via runtime context
    step = runtime.context.get("_perf_step", 0) + 1
    runtime.context["_perf_step"] = step

    # Record model start time so downstream can compute elapsed
    runtime.context["_model_start"] = now_ms()

    log_perf("agent_model", "start",
             request_id=request_id,
             status="start",
             step=step,
             message_count=msg_count,
             last=type(last_msg).__name__)

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
