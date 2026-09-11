import os
import io
import pandas as pd
import pytest

from app.memory.storage import (
    SessionStorage,
    ChatMessage,
    ConversationSession,
    default_session_storage,
)
from app.memory.manager import (
    rewrite_contextual_query,
    sync_session_turn,
)
from app.agents.graph import build_datapilot_graph
from app.llm.model import MockChatModel


@pytest.fixture
def memory_storage(tmp_path):
    """Provides a fresh isolated SessionStorage instance in a temp directory."""
    temp_dir = os.path.join(tmp_path, "sessions")
    storage = SessionStorage(storage_dir=temp_dir)
    return storage


def test_session_storage_creation_and_retrieval(memory_storage):
    """Test session creation, message storage, and persistent retrieval."""
    session = memory_storage.create_session(dataset_id="ds_123")
    sid = session.session_id

    memory_storage.add_message(session_id=sid, role="user", content="Hello, analyze my dataset")
    memory_storage.add_message(session_id=sid, role="assistant", content="Ready to analyze", metadata={"confidence": 1.0})

    retrieved = memory_storage.get_session(sid)
    assert retrieved is not None
    assert retrieved.dataset_id == "ds_123"
    assert len(retrieved.messages) == 2
    assert retrieved.messages[0].role == "user"
    assert retrieved.messages[1].role == "assistant"
    assert retrieved.messages[1].metadata == {"confidence": 1.0}

    # Verify JSON file exists on disk
    file_path = os.path.join(memory_storage.storage_dir, f"{sid}.json")
    assert os.path.exists(file_path)

    # Clean up
    memory_storage.delete_session(sid)
    assert not os.path.exists(file_path)


def test_session_storage_chat_history_limit(memory_storage):
    """Test get_chat_history limiting returned message count."""
    session = memory_storage.create_session()
    sid = session.session_id

    for i in range(12):
        role = "user" if i % 2 == 0 else "assistant"
        memory_storage.add_message(session_id=sid, role=role, content=f"Message {i+1}")

    history = memory_storage.get_chat_history(sid, limit=5)
    assert len(history) == 5
    assert history[0].content == "Message 8"
    assert history[-1].content == "Message 12"


def test_contextual_query_rewriting_first_turn(memory_storage):
    """Test query rewriter on first turn query returns query unchanged."""
    query = "What are the top products by revenue?"
    rewritten = rewrite_contextual_query(query=query, storage=memory_storage, llm=MockChatModel())
    assert rewritten == query


def test_contextual_query_rewriting_followup(memory_storage):
    """Test query rewriter reconstructing follow-up questions from session history."""
    session = memory_storage.create_session()
    sid = session.session_id

    memory_storage.add_message(session_id=sid, role="user", content="What are the top products by revenue?")
    memory_storage.add_message(session_id=sid, role="assistant", content="Top product is Laptop generating 1400.0.")

    followup_query = "Show that as a pie chart"
    rewritten = rewrite_contextual_query(query=followup_query, session_id=sid, storage=memory_storage, llm=MockChatModel())

    assert "pie" in rewritten.lower()
    assert "top products by revenue" in rewritten.lower()


def test_sync_session_turn(memory_storage):
    """Test sync_session_turn appending user and assistant messages into session."""
    session = sync_session_turn(
        session_id="sess_sync_test",
        user_query="Show sales trend",
        rewritten_query="Show sales trend over time",
        assistant_response="Trend analysis completed.",
        dataset_id="ds_456",
        metadata={"status": "success"},
        storage=memory_storage,
    )

    assert session.session_id == "sess_sync_test"
    assert session.dataset_id == "ds_456"
    assert len(session.messages) == 2
    assert session.messages[0].metadata["rewritten_query"] == "Show sales trend over time"
    assert session.messages[1].content == "Trend analysis completed."


def test_langgraph_multi_turn_consecutive_queries(memory_storage):
    """Test LangGraph StateGraph executing consecutive multi-turn conversational analysis."""
    graph = build_datapilot_graph()
    session = memory_storage.create_session()
    sid = session.session_id

    df = pd.DataFrame({
        "Product": ["Laptop", "Phone", "Laptop"],
        "Revenue": [1200.0, 800.0, 1400.0],
    })

    # Turn 1
    state_turn1 = {
        "session_id": sid,
        "query": "What are top products by revenue?",
        "df": df,
        "llm": MockChatModel(),
    }
    result_turn1 = graph.invoke(state_turn1)

    sync_session_turn(
        session_id=sid,
        user_query="What are top products by revenue?",
        rewritten_query=result_turn1["query"],
        assistant_response=result_turn1["insight_output"].executive_summary,
        storage=memory_storage,
    )

    # Turn 2: Follow-up question
    state_turn2 = {
        "session_id": sid,
        "query": "Show that as a pie chart",
        "df": df,
        "llm": MockChatModel(),
    }
    result_turn2 = graph.invoke(state_turn2)

    assert "pie" in result_turn2["query"].lower()
    assert result_turn2["vis_output"].is_useful is True
    assert result_turn2["vis_output"].chart_type == "pie"

    # Clean up
    memory_storage.delete_session(sid)
