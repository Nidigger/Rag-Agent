"""Final Answer Streamer — Phase 2: Token-level streaming for user-facing output.

This module takes the structured results from the Agent phase
(AgentExecutionResult) and streams the final answer token-by-token
using a streaming-capable OpenAI-compatible model — WITHOUT binding
any tools. This guarantees clean SSE output with no intermediate
states, tool call JSON, or raw data leaking to the frontend.
"""

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.integrations.llm_client import get_streaming_model
from app.utils.perf import elapsed_ms, log_perf, now_ms

logger = logging.getLogger("rag-agent.final_answer_streamer")

# Default system prompts for chat and report scenarios
_CHAT_SYSTEM_PROMPT = (
    "你是扫地机器人智能助手。"
    "请只基于给定的上下文和草稿回答用户问题。"
    "不要输出工具调用过程、原始 JSON 或中间推理步骤。"
    "回答应当简洁、自然、口语化。"
)

_REPORT_SYSTEM_PROMPT = (
    "你是扫地机器人智能助手。"
    "请只基于给定的上下文和草稿生成用户使用报告。"
    "不要输出工具调用过程、原始 JSON 或中间推理步骤。"
    "请生成结构清晰、可读性强的用户使用报告，"
    "包含数据摘要、使用分析和合理建议。"
)


class FinalAnswerStreamer:
    """Streams the final answer token-by-token via an OpenAI-compatible model.

    No tools are bound — this is purely for generating the user-facing
    response based on the context collected during the Agent phase.
    """

    def __init__(self):
        self._model = get_streaming_model()

    def stream_final_answer(
        self,
        query: str,
        tool_context: str,
        final_draft: str | None = None,
        report: bool = False,
        request_id: str | None = None,
    ):
        """Stream the final answer as a sync generator of text tokens.

        Args:
            query: The original user question.
            tool_context: Concatenated tool results from the Agent phase.
            final_draft: Draft answer from the Agent, used as reference.
            report: If True, use report-style system prompt.
            request_id: Request ID for performance logging.

        Yields:
            str: Text chunks (typically 1-3 tokens each) for SSE delivery.
        """
        rid = request_id or "internal"
        system_prompt = _REPORT_SYSTEM_PROMPT if report else _CHAT_SYSTEM_PROMPT

        user_prompt = (
            f"用户问题：\n{query}\n\n"
            f"已获取的工具/检索上下文：\n{tool_context}\n\n"
            f"Agent 草稿：\n{final_draft or ''}\n\n"
            f"请输出最终回答："
        )

        logger.info(
            "[FinalAnswerStreamer] Starting stream: report=%s, "
            "context_len=%d, draft_len=%d",
            report,
            len(tool_context),
            len(final_draft or ""),
        )

        log_perf("final_stream", "start",
                 request_id=rid,
                 report=report,
                 draft_len=len(final_draft or ""),
                 context_len=len(tool_context))

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        stream_start = now_ms()
        first_token_yielded = False
        output_len = 0

        for chunk in self._model.stream(messages):
            text = getattr(chunk, "content", "")
            if text:
                if not first_token_yielded:
                    log_perf("final_stream", "first_token",
                             request_id=rid,
                             elapsed_ms=elapsed_ms(stream_start))
                    first_token_yielded = True
                output_len += len(text)
                yield text

        log_perf("final_stream", "done",
                 request_id=rid,
                 elapsed_ms=elapsed_ms(stream_start),
                 output_len=output_len)
        logger.info("[FinalAnswerStreamer] Stream finished")


# Module-level singleton
_streamer: FinalAnswerStreamer | None = None


def get_final_answer_streamer() -> FinalAnswerStreamer:
    """Get or create the singleton FinalAnswerStreamer instance."""
    global _streamer
    if _streamer is None:
        _streamer = FinalAnswerStreamer()
    return _streamer
