import logging
from typing import Optional, List, Dict, Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from app.memory.storage import (
    SessionStorage,
    default_session_storage,
    ChatMessage,
    ConversationSession,
)
from app.llm.model import get_llm_model, MockChatModel

logger = logging.getLogger("datapilot.memory.manager")

QUERY_REWRITER_PROMPT = """You are a Conversational Query Rewriter for DataPilot AI Data Analyst.
Your task is to review the conversation history and a user's follow-up question, then rewrite the follow-up question into a complete, standalone analytical query.

Rules:
1. If the follow-up query is already complete and self-contained, return it unchanged.
2. If the follow-up query references prior context (e.g., "show that as a chart", "filter for Laptop", "what about North region?"), combine it with the previous context to produce a explicit query.
3. Output ONLY the rewritten standalone query. Do not add conversational intro/outro text.
"""


def rewrite_contextual_query(
    query: str,
    session_id: Optional[str] = None,
    storage: Optional[SessionStorage] = None,
    llm: Optional[BaseChatModel] = None,
) -> str:
    """Rewrites a follow-up query into a standalone analytical query using session chat history."""
    if not session_id:
        return query

    sess_storage = storage or default_session_storage
    session = sess_storage.get_session(session_id)
    if not session or not session.messages:
        return query

    recent_history = session.messages[-6:]  # Last 3 turns
    history_lines = []
    for msg in recent_history:
        history_lines.append(f"{msg.role.capitalize()}: {msg.content}")
    history_text = "\n".join(history_lines)

    # Heuristic check for common follow-up phrases
    q_lower = query.lower().strip()
    is_short_followup = len(q_lower.split()) <= 6 or any(
        k in q_lower for k in ["show that", "plot that", "visualize that", "make it", "filter for", "what about", "how about"]
    )

    if not is_short_followup:
        return query

    # LLM Rewriter
    if llm is not None:
        agent_llm = llm
    else:
        try:
            agent_llm = get_llm_model()
        except Exception as e:
            logger.warning(f"Query rewriter failed to initialize LLM model: {e}. Using MockChatModel.")
            agent_llm = MockChatModel()
    if not isinstance(agent_llm, MockChatModel):
        try:
            messages = [
                SystemMessage(content=QUERY_REWRITER_PROMPT),
                HumanMessage(
                    content=f"Conversation History:\n{history_text}\n\nFollow-up Question: '{query}'\n\nStandalone Rewritten Query:"
                ),
            ]
            response = agent_llm.invoke(messages)
            rewritten = response.content if hasattr(response, "content") else str(response)
            if isinstance(rewritten, str) and rewritten.strip():
                clean_rewritten = rewritten.strip().strip('"').strip("'")
                logger.info(f"Contextual Query Rewritten: '{query}' -> '{clean_rewritten}'")
                return clean_rewritten
        except Exception as e:
            logger.warning(f"LLM query rewriter failed: {e}. Using heuristic query rewriting.")

    # Heuristic Rewriter Fallback
    last_user_query = None
    for msg in reversed(recent_history):
        if msg.role == "user":
            last_user_query = msg.content
            break

    if not last_user_query:
        return query

    if any(k in q_lower for k in ["show that as a", "plot that as a", "visualize that as a"]):
        chart_kind = q_lower.replace("show that as a", "").replace("plot that as a", "").replace("visualize that as a", "").strip()
        return f"{last_user_query} as a {chart_kind}"
    elif any(k in q_lower for k in ["show that", "plot that", "visualize that"]):
        return f"Visualize {last_user_query}"
    elif "filter for" in q_lower or "filter by" in q_lower:
        filter_val = q_lower.replace("filter for", "").replace("filter by", "").strip()
        return f"{last_user_query} filtered for {filter_val}"
    elif "what about" in q_lower or "how about" in q_lower:
        val = q_lower.replace("what about", "").replace("how about", "").strip()
        return f"{last_user_query} for {val}"

    return f"{last_user_query} ({query})"


def sync_session_turn(
    session_id: str,
    user_query: str,
    rewritten_query: str,
    assistant_response: str,
    dataset_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    storage: Optional[SessionStorage] = None,
) -> ConversationSession:
    """Records a complete user/assistant turn into session memory."""
    sess_storage = storage or default_session_storage
    session = sess_storage.get_or_create_session(session_id=session_id, dataset_id=dataset_id)

    # 1. Record User Message
    user_meta = {"rewritten_query": rewritten_query} if rewritten_query != user_query else {}
    sess_storage.add_message(session_id=session_id, role="user", content=user_query, metadata=user_meta)

    # 2. Record Assistant Response Message
    sess_storage.add_message(session_id=session_id, role="assistant", content=assistant_response, metadata=metadata or {})

    return sess_storage.get_session(session_id) # type: ignore
