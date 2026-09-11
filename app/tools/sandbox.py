import ast
import io
import sys
import math
import json
import re
import time
import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
from app.core.security import (
    DISALLOWED_IMPORTS,
    DISALLOWED_BUILTINS,
    DISALLOWED_ATTRIBUTES,
)
from app.tools.analysis import _resolve_dataframe
from app.data.metadata import sanitize_for_json


class SandboxSecurityError(Exception):
    """Custom exception raised when code fails security validation."""
    pass


def validate_code_security(code: str) -> List[str]:
    """Inspects Python code via AST parsing and returns a list of security policy violations."""
    violations: List[str] = []

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"SyntaxError: {str(e)}"]

    for node in ast.walk(tree):
        # 1. Inspect import statements (import os, import sys)
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod_base = alias.name.split(".")[0]
                if mod_base in DISALLOWED_IMPORTS:
                    violations.append(f"Prohibited import: module '{alias.name}' is not allowed.")

        # 2. Inspect from-imports (from os import path)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                mod_base = node.module.split(".")[0]
                if mod_base in DISALLOWED_IMPORTS:
                    violations.append(f"Prohibited import: module '{node.module}' is not allowed.")

        # 3. Inspect function calls (eval(), exec(), open())
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in DISALLOWED_BUILTINS:
                    violations.append(f"Prohibited function call: '{node.func.id}()' is not allowed.")
            elif isinstance(node.func, ast.Attribute):
                if node.func.attr in DISALLOWED_BUILTINS:
                    violations.append(f"Prohibited method call: '.{node.func.attr}()' is not allowed.")

        # 4. Inspect dunder attribute access (__subclasses__, __globals__)
        elif isinstance(node, ast.Attribute):
            if node.attr in DISALLOWED_ATTRIBUTES:
                violations.append(f"Prohibited attribute access: '{node.attr}' is not allowed.")

    return violations


def _run_in_restricted_env(code: str, target_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """Internal helper to execute code in restricted namespace and capture stdout."""
    stdout_buffer = io.StringIO()
    old_stdout = sys.stdout

    safe_builtins = {
        "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
        "enumerate": enumerate, "filter": filter, "float": float,
        "int": int, "isinstance": isinstance, "len": len, "list": list,
        "map": map, "max": max, "min": min, "print": print, "range": range,
        "round": round, "set": set, "sorted": sorted, "str": str,
        "sum": sum, "tuple": tuple, "zip": zip, "True": True, "False": False, "None": None,
    }

    safe_globals: Dict[str, Any] = {
        "__builtins__": safe_builtins,
        "pd": pd,
        "np": np,
        "math": math,
        "datetime": datetime,
        "re": re,
        "json": json,
    }

    if target_df is not None:
        safe_globals["df"] = target_df.copy()

    local_vars: Dict[str, Any] = {}

    try:
        sys.stdout = stdout_buffer
        compiled = compile(code, "<sandbox>", "exec")
        exec(compiled, safe_globals, local_vars)
        sys.stdout = old_stdout

        output_text = stdout_buffer.getvalue().strip()

        # Extract result variable if set, or last local variable
        result_val = None
        if "result" in local_vars:
            result_val = local_vars["result"]
        elif "res" in local_vars:
            result_val = local_vars["res"]

        # Sanitize result for JSON serialization safety
        if isinstance(result_val, pd.DataFrame):
            records = []
            for _, r in result_val.head(50).iterrows():
                records.append({str(k): sanitize_for_json(v) for k, v in r.items()})
            clean_result = {
                "type": "DataFrame",
                "row_count": len(result_val),
                "columns": list(result_val.columns),
                "data": records,
            }
        elif isinstance(result_val, pd.Series):
            clean_result = {
                "type": "Series",
                "name": str(result_val.name),
                "data": {str(k): sanitize_for_json(v) for k, v in result_val.head(50).items()},
            }
        else:
            clean_result = sanitize_for_json(result_val)

        return {
            "success": True,
            "stdout": output_text,
            "result": clean_result,
            "error": None,
        }
    except Exception as e:
        sys.stdout = old_stdout
        return {
            "success": False,
            "stdout": stdout_buffer.getvalue().strip(),
            "result": None,
            "error": f"{type(e).__name__}: {str(e)}",
        }


def execute_sandboxed_code(
    code: str,
    df: Optional[pd.DataFrame] = None,
    dataset_id: Optional[str] = None,
    timeout_seconds: int = 5,
) -> Dict[str, Any]:
    """Executes Python code in a safe, restricted execution environment.

    - Performs AST static security checks.
    - Blocks dangerous modules, file IO, network calls, and reflection.
    - Captures stdout prints.
    - Enforces execution timeout.
    """
    start_time = time.time()

    # Step 1: Resolve DataFrame if dataset_id provided
    target_df = None
    if dataset_id or df is not None:
        try:
            target_df = _resolve_dataframe(dataset_id, df)
        except Exception as e:
            return {
                "success": False,
                "security_violations": [],
                "stdout": "",
                "result": None,
                "execution_time_seconds": 0.0,
                "error": str(e),
            }

    # Step 2: AST Security Analysis
    violations = validate_code_security(code)
    if violations:
        return {
            "success": False,
            "security_violations": violations,
            "stdout": "",
            "result": None,
            "execution_time_seconds": round(time.time() - start_time, 4),
            "error": "Code rejected due to security policy violations.",
        }

    # Step 3: Execute in ThreadPoolExecutor for Timeout Enforcement
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_run_in_restricted_env, code, target_df)
        try:
            res = future.result(timeout=timeout_seconds)
            res["security_violations"] = []
            res["execution_time_seconds"] = round(time.time() - start_time, 4)
            return res
        except TimeoutError:
            return {
                "success": False,
                "security_violations": [],
                "stdout": "",
                "result": None,
                "execution_time_seconds": round(time.time() - start_time, 4),
                "error": f"Execution timed out after {timeout_seconds} second(s).",
            }
