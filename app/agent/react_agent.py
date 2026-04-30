"""ReAct Agent — Phase 1: Non-streaming tool orchestration.

This module is responsible ONLY for the agent phase:
- Determining which tools to call
- Executing RAG, weather, external data queries, etc.
- Collecting results into an AgentExecutionResult

It does NOT produce SSE output. The streaming of the final answer
is handled by FinalAnswerStreamer (Phase 2).
"""

import logging

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage

from app.agent.tools.agent_tools import (
    fetch_external_data,
    fill_context_for_report,
    get_current_month,
    get_user_id,
    get_user_location,
    get_weather,
    rag_summarize,
)
from app.agent.tools.middleware import (
    emit_pending_model_done,
    log_before_model,
    monitor_tool,
    report_prompt_switch,
)
from app.integrations.llm_client import get_agent_model
from app.schemas.common import AgentExecutionResult
from app.utils.perf import elapsed_ms, log_perf, now_ms
from app.utils.prompt_loader import load_system_prompts

logger = logging.getLogger("rag-agent.react_agent")


class ReactAgent:
    """ReAct Agent that orchestrates tool calls without streaming.

    Uses a non-streaming OpenAI-compatible model to ensure tool call
    arguments are always complete, valid JSON.
    """

    def __init__(self):
        self._agent = None

    def _ensure_agent(self):
        if self._agent is None:
            self._agent = create_agent(
                model=get_agent_model(),
                system_prompt=load_system_prompts(),
                tools=[
                    rag_summarize,
                    get_weather,
                    get_user_location,
                    get_user_id,
                    get_current_month,
                    fetch_external_data,
                    fill_context_for_report,
                ],
                middleware=[monitor_tool, log_before_model, report_prompt_switch],
            )
            logger.info("[ReactAgent] agent initialized")

    def execute(
        self,
        query: str,
        context: dict | None = None,
        messages: list | None = None,
    ) -> AgentExecutionResult:
        """Run the full Agent loop and return structured results.

        Args:
            query: The user's message.
            context: Runtime context dict (e.g. {"report": True, "user_id": "1001"}).
            messages: Prior conversation history for multi-turn support.

        Returns:
            AgentExecutionResult containing the draft answer, collected
            tool context, tool names used, total call count, and full
            message history.
        """
        self._ensure_agent()
        if context is None:
            context = {"report": False}

        history = messages or []
        # Keep last 10 messages (5 rounds) to avoid prompt overflow
        if len(history) > 10:
            history = history[-10:]

        request_id = context.get("request_id", "internal")

        input_dict = {
            "messages": [*history, {"role": "user", "content": query}]
        }

        logger.info(
            "[ReactAgent] Starting execution: query=%r, context=%s, "
            "history_len=%d",
            query[:80],
            context,
            len(history),
        )

        agent_start = now_ms()
        log_perf("agent", "start",
                 request_id=request_id,
                 status="start",
                 history_len=len(history),
                 report=context.get("report", False))

        # Non-streaming: let the Agent run all tool calls to completion
        final_state = self._agent.invoke(
            input_dict, context=context
        )

        # Emit the final model done event if log_before_model stored a
        # _model_start that was never consumed by a subsequent tool call.
        emit_pending_model_done(context)

        all_messages = final_state.get("messages", [])

        # Collect tool context and used tool names (unique)
        tool_context_parts: list[str] = []
        used_tools: list[str] = []
        tool_call_names: list[str] = []

        for msg in all_messages:
            if isinstance(msg, ToolMessage):
                tool_context_parts.append(msg.content)
                tool_name = getattr(msg, "name", None) or "unknown"
                tool_call_names.append(tool_name)
                if tool_name not in used_tools:
                    used_tools.append(tool_name)

        tool_context = "\n".join(tool_context_parts)
        tool_call_count = len(tool_call_names)
        unique_tool_count = len(used_tools)

        # Find the final AIMessage (last one without tool_calls)
        final_draft = ""
        for msg in reversed(all_messages):
            if isinstance(msg, AIMessage):
                if not getattr(msg, "tool_calls", None):
                    final_draft = _message_content_to_text(msg.content).strip()
                    break

        agent_elapsed = elapsed_ms(agent_start)
        log_perf("agent", "done",
                 request_id=request_id,
                 status="success",
                 elapsed_ms=agent_elapsed,
                 used_tools=",".join(used_tools),
                 unique_tool_count=unique_tool_count,
                 tool_call_count=tool_call_count,
                 draft_len=len(final_draft))

        logger.info(
            "[ReactAgent] Execution complete: used_tools=%s, "
            "unique_tool_count=%d, tool_call_count=%d, "
            "draft_len=%d, tool_context_len=%d",
            used_tools,
            unique_tool_count,
            tool_call_count,
            len(final_draft),
            len(tool_context),
        )

        return AgentExecutionResult(
            final_draft=final_draft,
            tool_context=tool_context,
            used_tools=used_tools,
            tool_call_count=tool_call_count,
            messages=all_messages,
        )


def _message_content_to_text(content) -> str:
    """Normalize AIMessage.content (str or list[dict]) to plain text."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)

    return str(content)
