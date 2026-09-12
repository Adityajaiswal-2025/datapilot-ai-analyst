"""Security utilities, CORS settings, and sandbox security rules for DataPilot."""

from typing import Set
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from app.core.config import settings


# AST Security Sandbox Blacklists
DISALLOWED_IMPORTS: Set[str] = {
    "os", "sys", "subprocess", "socket", "urllib", "requests", "httpx",
    "shutil", "importlib", "pathlib", "ctypes", "threading", "multiprocessing",
    "asyncio", "builtins", "signal", "tempfile", "glob", "pickle", "shelve",
    "dbm", "platform", "code", "codeop", "pty", "tty", "termios"
}

DISALLOWED_BUILTINS: Set[str] = {
    "eval", "exec", "open", "__import__", "globals", "locals",
    "getattr", "setattr", "delattr", "compile", "breakpoint",
    "exit", "quit", "input", "memoryview", "vars"
}

DISALLOWED_ATTRIBUTES: Set[str] = {
    "__subclasses__", "__globals__", "__code__", "__mro__",
    "__bases__", "__import__", "__builtins__", "__class__"
}


def setup_security(app: FastAPI) -> None:
    """Configures CORS and security middleware for the FastAPI app."""
    if settings.BACKEND_CORS_ORIGINS:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
