import os
import json
import uuid
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Literal
from pydantic import BaseModel, Field

logger = logging.getLogger("datapilot.memory.storage")

SESSIONS_DIR = "data/sessions"


class ChatMessage(BaseModel):
    """Represents a single message in a conversational session."""
    role: Literal["user", "assistant", "system"] = Field(description="Role of the message sender")
    content: str = Field(description="Text content of the message")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO timestamp of message creation"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata associated with message (e.g. dataset_id, tool_calls, chart_path)"
    )


class ConversationSession(BaseModel):
    """Represents a multi-turn conversation session."""
    session_id: str = Field(description="Unique session identifier")
    dataset_id: Optional[str] = Field(default=None, description="Currently active dataset ID for session")
    messages: List[ChatMessage] = Field(default_factory=list, description="Ordered conversation history")
    active_dataframe_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Context tracking active filters, columns, or segment selections"
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO timestamp of session creation"
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO timestamp of last update"
    )


class SessionStorageError(Exception):
    """Custom exception raised when session storage operations fail."""
    pass


class SessionStorage:
    """Thread-safe in-memory session manager with persistent JSON backup on disk."""

    def __init__(self, storage_dir: str = SESSIONS_DIR):
        self.storage_dir = storage_dir
        self._sessions: Dict[str, ConversationSession] = {}
        os.makedirs(self.storage_dir, exist_ok=True)

    def create_session(
        self, session_id: Optional[str] = None, dataset_id: Optional[str] = None
    ) -> ConversationSession:
        """Creates a new conversation session."""
        sid = session_id or f"session_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc).isoformat()
        session = ConversationSession(
            session_id=sid,
            dataset_id=dataset_id,
            messages=[],
            active_dataframe_context={},
            created_at=now,
            updated_at=now,
        )
        self._sessions[sid] = session
        self.save_to_disk(session)
        logger.info(f"Created new conversation session: {sid}")
        return session

    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Retrieves a session from memory cache or loads it from disk."""
        if session_id in self._sessions:
            return self._sessions[session_id]

        loaded = self.load_from_disk(session_id)
        if loaded:
            self._sessions[session_id] = loaded
            return loaded
        return None

    def get_or_create_session(
        self, session_id: Optional[str] = None, dataset_id: Optional[str] = None
    ) -> ConversationSession:
        """Retrieves an existing session or creates a new one if not found."""
        if session_id:
            existing = self.get_session(session_id)
            if existing:
                if dataset_id and existing.dataset_id != dataset_id:
                    existing.dataset_id = dataset_id
                    existing.updated_at = datetime.now(timezone.utc).isoformat()
                    self.save_to_disk(existing)
                return existing
        return self.create_session(session_id=session_id, dataset_id=dataset_id)

    def add_message(
        self,
        session_id: str,
        role: Literal["user", "assistant", "system"],
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ChatMessage:
        """Appends a new message to a session and persists to disk."""
        session = self.get_session(session_id)
        if not session:
            session = self.create_session(session_id=session_id)

        msg = ChatMessage(
            role=role,
            content=content,
            metadata=metadata or {},
        )
        session.messages.append(msg)
        session.updated_at = datetime.now(timezone.utc).isoformat()
        self.save_to_disk(session)
        return msg

    def get_chat_history(
        self, session_id: str, limit: int = 10
    ) -> List[ChatMessage]:
        """Returns the most recent N messages from a session history."""
        session = self.get_session(session_id)
        if not session:
            return []
        return session.messages[-limit:]

    def update_session_dataset(self, session_id: str, dataset_id: str) -> bool:
        """Updates the active dataset_id associated with a session."""
        session = self.get_session(session_id)
        if not session:
            return False
        session.dataset_id = dataset_id
        session.updated_at = datetime.now(timezone.utc).isoformat()
        self.save_to_disk(session)
        return True

    def delete_session(self, session_id: str) -> bool:
        """Deletes a session from memory cache and disk."""
        if session_id in self._sessions:
            del self._sessions[session_id]

        file_path = os.path.join(self.storage_dir, f"{session_id}.json")
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Deleted session file: {file_path}")
                return True
            except OSError as e:
                logger.error(f"Failed to delete session file {file_path}: {e}")
                return False
        return True

    def save_to_disk(self, session: ConversationSession) -> None:
        """Saves session object to persistent JSON file on disk."""
        file_path = os.path.join(self.storage_dir, f"{session.session_id}.json")
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(session.model_dump_json(indent=2))
        except Exception as e:
            logger.error(f"Failed to save session {session.session_id} to disk: {e}")

    def load_from_disk(self, session_id: str) -> Optional[ConversationSession]:
        """Loads session object from persistent JSON file on disk."""
        file_path = os.path.join(self.storage_dir, f"{session_id}.json")
        if not os.path.exists(file_path):
            return None
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return ConversationSession(**data)
        except Exception as e:
            logger.error(f"Failed to load session {session_id} from disk: {e}")
            return None


# Global singleton storage instance
default_session_storage = SessionStorage()
