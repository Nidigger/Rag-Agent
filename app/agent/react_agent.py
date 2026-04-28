import logging

from langchain.agents import create_agent

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
    log_before_model,
    monitor_tool,
    report_prompt_switch,
)
from app.integrations.dashscope_client import get_chat_model
from app.utils.prompt_loader import load_system_prompts

logger = logging.getLogger("rag-agent.react_agent")


class ReactAgent:
    def __init__(self):
        self._agent = None

    def _ensure_agent(self):
        if self._agent is None:
            self._agent = create_agent(
                model=get_chat_model(),
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

    def execute_stream(
        self,
        query: str,
        context: dict | None = None,
        messages: list | None = None,
    ):
        self._ensure_agent()
        if context is None:
            context = {"report": False}

        history = messages or []
        # Keep last 10 messages (5 rounds) to avoid prompt overflow
        if len(history) > 10:
            history = history[-10:]
        input_dict = {
            "messages": [*history, {"role": "user", "content": query}]
        }

        for chunk in self._agent.stream(
            input_dict, stream_mode="values", context=context
        ):
            latest_message = chunk["messages"][-1]
            if latest_message.content:
                yield latest_message.content.strip() + "\n"
