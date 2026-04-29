"""In-memory session management for multi-turn conversations.

Stores messages keyed by session_id. Each session holds a list of
{role, content} dicts representing the conversation history.

Note: This is a simple in-memory store suitable for single-instance
deployments. For multi-process/multi-instance deployments, replace
with Redis or a database-backed store.
"""

import logging
import uuid
from typing import Any

logger = logging.getLogger("rag-agent.session_service")

_sessions: dict[str, dict[str, Any]] = {}


class SessionService:
    def create_session(self) -> str:
        """Create a new session and return its ID."""
        session_id = str(uuid.uuid4())
        _sessions[session_id] = {"messages": []}
        logger.info("[session] Created: %s", session_id)
        return session_id

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        """Retrieve a session by ID, or None if not found."""
        return _sessions.get(session_id)

    def add_message(
        self, session_id: str, role: str, content: str
    ) -> None:
        """Append a message to the session's conversation history."""
        if session_id not in _sessions:
            _sessions[session_id] = {"messages": []}
        _sessions[session_id]["messages"].append(
            {"role": role, "content": content}
        )
        logger.debug(
            "[session] Added %s message to %s (total: %d)",
            role,
            session_id,
            len(_sessions[session_id]["messages"]),
        )

    def get_messages(self, session_id: str) -> list[dict]:
        """Return the full message history for a session."""
        session = _sessions.get(session_id)
        if not session:
            return []
        return session["messages"]


_session_service: SessionService | None = None


def get_session_service() -> SessionService:
    """Get or create the singleton SessionService instance."""
    global _session_service
    if _session_service is None:
        _session_service = SessionService()
    return _session_service
