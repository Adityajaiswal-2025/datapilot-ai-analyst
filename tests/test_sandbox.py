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


def test_sandbox_timeout_enforcement():
    """Phase 19: Empirically verifies sandbox execution timeout enforcement."""
    import time
    start = time.time()
    code = "time.sleep(10)"
    res = execute_sandboxed_code(code, timeout_seconds=1)
    duration = time.time() - start

    assert res["success"] is False
    assert "timed out" in str(res["error"]).lower()
    assert res["error"] == "Execution timed out after 1 second(s)."
    assert duration < 3.0, f"Sandbox timeout took too long: {duration}s"


def test_sandbox_indirect_dangerous_constructs(sample_df):
    """Phase 19: Tests blocking indirect dangerous constructs outside sandbox policy."""
    forbidden_snippets = [
        '__import__("os")',
        'getattr(df, "__class__")',
        'object.__subclasses__()',
        'globals()',
        'locals()',
        'vars()',
        'import importlib',
        'import pathlib',
        'import shutil',
        'import socket',
        'import subprocess',
        'import os\nenv = os.environ',
    ]
    for code in forbidden_snippets:
        violations = validate_code_security(code)
        res = execute_sandboxed_code(code, df=sample_df)
        assert res["success"] is False, f"Code was unexpectedly allowed: {code}"
        assert len(violations) > 0 or len(res["security_violations"]) > 0 or "Prohibited" in str(res["error"]) or "NameError" in str(res["error"]) or "Security" in str(res["error"]) or "ImportError" in str(res["error"])


def test_sandbox_valid_analytical_code(sample_df):
    """Phase 19: Verifies valid pandas, numpy, and mathematical calculations work cleanly."""
    code = """
# Pandas grouping & filtering using pre-bound safe globals
grouped = df.groupby("Product")["Sales"].sum()
total_sales = float(np.sum(df["Sales"]))
mean_units = float(np.mean(df["Units"]))
calc_val = 2 + 2 * 10

result = {
    "total_sales": total_sales,
    "mean_units": mean_units,
    "calc_val": calc_val,
    "laptop_sales": int(grouped["Laptop"]),
}
"""
    res = execute_sandboxed_code(code, df=sample_df)
    assert res["success"] is True
    assert len(res["security_violations"]) == 0
    assert res["result"]["total_sales"] == 1600.0
    assert res["result"]["mean_units"] == 4.25
    assert res["result"]["calc_val"] == 22
    assert res["result"]["laptop_sales"] == 1200

