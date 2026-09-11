import pandas as pd
import pytest
from app.tools.sandbox import validate_code_security, execute_sandboxed_code


@pytest.fixture
def sample_df():
    """Provides a sample DataFrame for sandbox tests."""
    return pd.DataFrame({
        "Product": ["Laptop", "Mouse", "Keyboard", "Monitor"],
        "Sales": [1200, 25, 75, 300],
        "Units": [2, 10, 4, 1],
    })


def test_valid_python_execution(sample_df):
    """Test executing valid Pandas data analysis code in sandbox."""
    code = """
print("Analyzing sales data...")
high_sales = df[df['Sales'] > 100]
result = high_sales['Sales'].sum()
"""
    res = execute_sandboxed_code(code, df=sample_df)
    assert res["success"] is True
    assert len(res["security_violations"]) == 0
    assert res["stdout"] == "Analyzing sales data..."
    assert res["result"] == 1500


def test_block_unauthorized_imports():
    """Test blocking dangerous module imports."""
    bad_codes = [
        "import os\nos.system('echo hacked')",
        "import sys\nprint(sys.version)",
        "import subprocess\nsubprocess.run(['ls'])",
        "import urllib.request",
        "from os import path",
    ]
    for code in bad_codes:
        violations = validate_code_security(code)
        assert len(violations) > 0
        res = execute_sandboxed_code(code)
        assert res["success"] is False
        assert len(res["security_violations"]) > 0


def test_block_file_io():
    """Test blocking file system open calls."""
    code = "f = open('secret.txt', 'w')\nf.write('hack')"
    violations = validate_code_security(code)
    assert any("open" in v for v in violations)

    res = execute_sandboxed_code(code)
    assert res["success"] is False


def test_block_eval_exec():
    """Test blocking eval, exec, and __import__ calls."""
    code1 = "eval('1 + 1')"
    code2 = "exec('print(123)')"
    code3 = "__import__('os').system('dir')"

    for code in [code1, code2, code3]:
        violations = validate_code_security(code)
        assert len(violations) > 0


def test_block_dunder_reflection():
    """Test blocking dunder reflection attempts (__subclasses__)."""
    code = "classes = ().__class__.__bases__[0].__subclasses__()"
    violations = validate_code_security(code)
    assert any("__subclasses__" in v or "__class__" in v for v in violations)

    res = execute_sandboxed_code(code)
    assert res["success"] is False


def test_syntax_error_handling():
    """Test clean handling of Python syntax errors."""
    code = "def invalid_syntax("
    res = execute_sandboxed_code(code)
    assert res["success"] is False
    assert "SyntaxError" in res["security_violations"][0] or "SyntaxError" in str(res["error"])
