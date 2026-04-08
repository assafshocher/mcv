#!/usr/bin/env python3
"""
Notebook grading test harness.

Supports three spec formats:
  1. "functions"-based (HW1 scratch): structured tests with setup/call/check
  2. "tests"-array with inline "code" (HW2/HW3 scratch): each test has a code block
  3. "code_analysis" mode (HW3 applied): checks code patterns without execution

Usage:
    python run_tests.py <submissions_dir> <spec_file> [--output <results_dir>] [--timeout <seconds>] [--data-dir <path>]
"""

import argparse
import csv
import inspect
import json
import os
import re
import shutil
import sys
import time
import traceback
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError


PYTHON = sys.executable


def load_spec(spec_path: Path) -> dict:
    with open(spec_path) as f:
        return json.load(f)


def detect_spec_format(spec: dict) -> str:
    """Detect which spec format is being used."""
    if spec.get("grading_mode") == "code_analysis":
        return "code_analysis"
    if "functions" in spec:
        return "functions"
    if "tests" in spec:
        return "tests_array"
    raise ValueError("Unknown spec format: must have 'functions', 'tests', or grading_mode='code_analysis'")


def extract_student_id(nb, var_name: str = "STUDENT_ID") -> str:
    """Extract student ID from notebook cells."""
    for cell in nb.cells:
        if cell.cell_type == "code" and var_name in cell.source:
            for line in cell.source.split("\n"):
                line = line.strip()
                if line.startswith(var_name) and "=" in line:
                    val = line.split("=", 1)[1].strip().strip("\"'")
                    if val:
                        return val
    return "MISSING"


def extract_all_code(nb) -> str:
    """Extract all code from a notebook as a single string."""
    return "\n".join(cell.source for cell in nb.cells if cell.cell_type == "code")


# ═══════════════════════════════════════════════════════════════════════════════
# Format 1: functions-based (HW1 scratch)
# ═══════════════════════════════════════════════════════════════════════════════

def build_test_code_functions(spec: dict) -> str:
    """Build test cell code from functions-based spec."""
    lines = []
    lines.append("import json, time, inspect, traceback")
    lines.append(spec.get("setup_code", ""))
    lines.append("")

    ref_code = spec.get("reference_code", "")
    lines.append("# Reference implementations")
    lines.append(ref_code)
    lines.append("")

    lines.append("_results = {'student_id': '', 'tests': {}, 'timing': {}, 'source': {}}")
    lines.append("")

    var_name = spec.get("student_id_variable", "STUDENT_ID")
    lines.append(f"try:")
    lines.append(f"    _results['student_id'] = str({var_name}).strip()")
    lines.append(f"except:")
    lines.append(f"    _results['student_id'] = 'MISSING'")
    lines.append("")

    for func_name in spec.get("functions", {}):
        lines.append(f"try:")
        lines.append(f"    _results['source']['{func_name}'] = inspect.getsource({func_name})")
        lines.append(f"except:")
        lines.append(f"    _results['source']['{func_name}'] = 'NOT_FOUND'")
    lines.append("")

    for func_name, func_spec in spec.get("functions", {}).items():
        lines.append(f"# --- Tests for {func_name} ---")
        lines.append(f"try:")
        lines.append(f"    {func_name}")
        lines.append(f"except NameError:")
        lines.append(f"    _results['tests']['{func_name}_exists'] = 'FAIL: function not defined'")
        lines.append("")

        banned = func_spec.get("banned_ops", [])
        if banned:
            lines.append(f"try:")
            lines.append(f"    _src = inspect.getsource({func_name})")
            lines.append(f"    _banned = {banned}")
            lines.append(f"    _found = [b for b in _banned if b in _src]")
            lines.append(f"    if _found:")
            lines.append(f"        _results['tests']['{func_name}_banned_ops'] = f'FAIL: used banned op(s): {{_found}}'")
            lines.append(f"    else:")
            lines.append(f"        _results['tests']['{func_name}_banned_ops'] = 'PASS'")
            lines.append(f"except Exception as e:")
            lines.append(f"    _results['tests']['{func_name}_banned_ops'] = f'FAIL: {{e}}'")
            lines.append("")

        for test in func_spec.get("tests", []):
            test_name = test["name"]
            lines.append(f"try:")
            if test.get("setup"):
                for setup_line in test["setup"].split("\n"):
                    lines.append(f"    {setup_line}")
            lines.append(f"    result = {test['call']}")
            if test.get("expected_setup"):
                lines.append(f"    expected = {test['expected_setup']}")
            for check_line in test["check"].split("\n"):
                lines.append(f"    {check_line}")
            lines.append(f"    _results['tests']['{test_name}'] = 'PASS'")
            lines.append(f"except Exception as e:")
            lines.append(f"    _results['tests']['{test_name}'] = f'FAIL: {{e}}'")
            lines.append("")

    for ttest in spec.get("timing_tests", []):
        tname = ttest["name"]
        lines.append(f"try:")
        if ttest.get("setup"):
            for setup_line in ttest["setup"].split("\n"):
                lines.append(f"    {setup_line}")
        if ttest.get("warmup"):
            lines.append(f"    {ttest['call']}  # warmup")
        lines.append(f"    _t0 = time.time()")
        lines.append(f"    {ttest['call']}")
        lines.append(f"    _results['timing']['{tname}'] = round((time.time() - _t0) * 1000, 1)")
        lines.append(f"except Exception as e:")
        lines.append(f"    _results['timing']['{tname}'] = f'FAIL: {{e}}'")
        lines.append("")

    lines.append("with open('__grading_result__.json', 'w') as _f:")
    lines.append("    json.dump(_results, _f, indent=2)")
    lines.append("print(json.dumps(_results, indent=2))")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Format 2: flat tests array with inline code (HW2/HW3 scratch)
# ═══════════════════════════════════════════════════════════════════════════════

def build_test_code_array(spec: dict) -> str:
    """Build test cell code from tests-array spec (inline code blocks)."""
    lines = []
    lines.append("import json, time, inspect, traceback")
    lines.append(spec.get("setup_code", ""))
    lines.append("")

    lines.append("_results = {'student_id': '', 'tests': {}, 'timing': {}, 'source': {}}")
    lines.append("")

    var_name = spec.get("student_id_variable", "STUDENT_ID")
    lines.append(f"try:")
    lines.append(f"    _results['student_id'] = str({var_name}).strip()")
    lines.append(f"except:")
    lines.append(f"    _results['student_id'] = 'MISSING'")
    lines.append("")

    # Extract source for functions mentioned in tests
    funcs_seen = set()
    for test in spec.get("tests", []):
        fn = test.get("function")
        if fn and fn not in funcs_seen:
            funcs_seen.add(fn)
            lines.append(f"try:")
            lines.append(f"    _results['source']['{fn}'] = inspect.getsource({fn})")
            lines.append(f"except:")
            lines.append(f"    _results['source']['{fn}'] = 'NOT_FOUND'")
    lines.append("")

    for test in spec.get("tests", []):
        test_name = test["name"]
        code = test.get("code", "")
        if not code:
            continue

        # Skip tests marked as skip or manual
        if test.get("skip"):
            lines.append(f"_results['tests']['{test_name}'] = 'SKIP'")
            lines.append("")
            continue
        if test.get("manual"):
            lines.append(f"_results['tests']['{test_name}'] = 'MANUAL'")
            lines.append("")
            continue

        lines.append(f"# --- {test_name} ---")
        lines.append(f"try:")
        for code_line in code.split("\n"):
            lines.append(f"    {code_line}")
        # Allow test code to set result directly (e.g., "BONUS: N")
        lines.append(f"    if '{test_name}' not in _results['tests']:")
        lines.append(f"        _results['tests']['{test_name}'] = 'PASS'")
        lines.append(f"except Exception as e:")
        lines.append(f"    _results['tests']['{test_name}'] = f'FAIL: {{e}}'")
        lines.append("")

    lines.append("with open('__grading_result__.json', 'w') as _f:")
    lines.append("    json.dump(_results, _f, indent=2)")
    lines.append("print(json.dumps(_results, indent=2))")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Format 3: code_analysis (HW3 applied)
# ═══════════════════════════════════════════════════════════════════════════════

def _has_not_implemented(all_code: str, name: str, kind: str = "def") -> bool:
    """Check if a function/class body raises NotImplementedError."""
    # Find the function/class definition and extract its body
    if kind == "class":
        pattern = rf"class\s+{re.escape(name)}\s*[\(:].*"
    else:
        pattern = rf"def\s+{re.escape(name)}\s*\(.*"
    match = re.search(pattern, all_code)
    if not match:
        return False
    # Get the body after the definition (next ~20 lines)
    body_start = match.end()
    body = all_code[body_start:body_start + 500]
    # Check if the first non-empty, non-docstring line is raise NotImplementedError
    lines = body.split("\n")
    in_docstring = False
    for line in lines[1:15]:  # Skip first line (rest of def), check next 15
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith('"""') or stripped.startswith("'''"):
            if in_docstring:
                in_docstring = False
                continue
            if stripped.count('"""') >= 2 or stripped.count("'''") >= 2:
                continue  # Single-line docstring
            in_docstring = True
            continue
        if in_docstring:
            continue
        # First real line of code
        if "NotImplementedError" in stripped:
            return True
        break  # First real code line doesn't raise NotImplementedError
    return False


def grade_code_analysis(nb_path: Path, spec: dict, output_dir: Path) -> dict:
    """Grade a notebook using code analysis (no execution)."""
    print(f"  Analyzing: {nb_path.name} ... ", end="", flush=True)

    try:
        nb = nbformat.read(str(nb_path), as_version=4)
    except Exception as e:
        print("PARSE ERROR")
        return {"student_id": "PARSE_ERROR", "file": nb_path.name,
                "tests": {}, "source": {}, "error": str(e)}

    var_name = spec.get("student_id_variable", "STUDENT_ID")
    student_id = extract_student_id(nb, var_name)
    all_code = extract_all_code(nb)

    results = {"student_id": student_id, "file": nb_path.name,
               "tests": {}, "source": {"all_code": all_code[:5000]}}

    for test in spec.get("tests", []):
        test_name = test["name"]
        check_type = test.get("check")

        # Manual grading tests — skip
        if test.get("type") == "manual" or test.get("manual"):
            results["tests"][test_name] = "MANUAL"
            continue

        if check_type == "class_exists":
            target = test["target"]
            pattern = rf"class\s+{re.escape(target)}\s*[\(:]"
            if re.search(pattern, all_code):
                base = test.get("base_class")
                if base:
                    pattern_base = rf"class\s+{re.escape(target)}\s*\(\s*.*{re.escape(base)}.*\)"
                    if re.search(pattern_base, all_code):
                        # Check for NotImplementedError in class body
                        if _has_not_implemented(all_code, target, "class"):
                            results["tests"][test_name] = f"FAIL: {target} raises NotImplementedError"
                        else:
                            results["tests"][test_name] = "PASS"
                    else:
                        results["tests"][test_name] = f"FAIL: {target} exists but doesn't inherit from {base}"
                else:
                    results["tests"][test_name] = "PASS"
            else:
                results["tests"][test_name] = f"FAIL: class {target} not found"

        elif check_type == "method_exists":
            target = test["target"]
            method = test["method"]
            pattern = rf"def\s+{re.escape(method)}\s*\("
            if re.search(pattern, all_code):
                if _has_not_implemented(all_code, method, "def"):
                    results["tests"][test_name] = f"FAIL: {method} raises NotImplementedError"
                else:
                    results["tests"][test_name] = "PASS"
            else:
                results["tests"][test_name] = f"FAIL: method {method} not found in {target}"

        elif check_type == "function_exists":
            target = test["target"]
            pattern = rf"def\s+{re.escape(target)}\s*\("
            if re.search(pattern, all_code):
                if _has_not_implemented(all_code, target, "def"):
                    results["tests"][test_name] = f"FAIL: {target} raises NotImplementedError"
                else:
                    results["tests"][test_name] = "PASS"
            else:
                results["tests"][test_name] = f"FAIL: function {target} not found"

        elif check_type == "code_contains":
            patterns = test.get("patterns", [])
            missing = [p for p in patterns if p not in all_code]
            if not missing:
                results["tests"][test_name] = "PASS"
            else:
                results["tests"][test_name] = f"FAIL: missing patterns: {missing}"

        else:
            results["tests"][test_name] = "SKIP: unknown check type"

    # Print summary
    tests = results.get("tests", {})
    n_pass = sum(1 for v in tests.values() if v == "PASS")
    n_total = sum(1 for v in tests.values() if v != "MANUAL")
    print(f"{n_pass}/{n_total} passed  (ID: {student_id})")

    # Save per-student results
    result_file = output_dir / f"{nb_path.stem}_results.json"
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Notebook execution (formats 1 & 2)
# ═══════════════════════════════════════════════════════════════════════════════

def grade_notebook(nb_path: Path, spec: dict, output_dir: Path,
                   timeout: int = 300, data_dir: str = None) -> dict:
    """Execute a student notebook with test cells, return results."""
    print(f"  Testing: {nb_path.name} ... ", end="", flush=True)

    try:
        nb = nbformat.read(str(nb_path), as_version=4)
    except Exception as e:
        print("PARSE ERROR")
        return {"student_id": "PARSE_ERROR", "file": nb_path.name,
                "tests": {}, "timing": {}, "source": {}, "error": str(e)}

    var_name = spec.get("student_id_variable", "STUDENT_ID")
    backup_id = extract_student_id(nb, var_name)

    # Build test code based on spec format
    fmt = detect_spec_format(spec)
    if fmt == "functions":
        test_code = build_test_code_functions(spec)
    else:
        test_code = build_test_code_array(spec)

    test_cell = nbformat.v4.new_code_cell(source=test_code)

    # If definitions_only mode: keep only code cells that define classes/functions/imports
    # This avoids running slow training cells — useful for ZSSR, MNIST training, etc.
    if spec.get("definitions_only", False):
        import re
        def is_definition_cell(cell):
            if cell.cell_type != 'code':
                return False  # skip markdown in definitions_only mode
            src = cell.source.strip()
            if not src:
                return False
            # Keep cells that define functions or classes
            if re.search(r'^(def |class )', src, re.MULTILINE):
                return True
            # Keep cells that are primarily imports/setup
            lines = [l.strip() for l in src.split('\n') if l.strip() and not l.strip().startswith('#')]
            if not lines:
                return False
            setup_prefixes = ('import ', 'from ', '%', 'device', 'torch.manual_seed',
                              'np.random.seed', 'STUDENT_ID', 'warnings', 'print(f"Using')
            if all(any(l.startswith(p) for p in setup_prefixes) for l in lines):
                return True
            return False
        nb.cells = [c for c in nb.cells if is_definition_cell(c)]

    nb.cells.append(test_cell)

    # Execute in a temp working directory
    work_dir = output_dir / f"_work_{nb_path.stem}"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Copy test files (images, data) into work directory
    test_files = spec.get("test_files", [])
    spec_dir = Path(spec.get("_spec_dir", "."))
    for fname in test_files:
        src = spec_dir / fname
        if src.exists():
            shutil.copy2(str(src), str(work_dir / fname))

    # Symlink data directory if provided (for MNIST, CIFAR, etc.)
    if data_dir:
        data_src = Path(data_dir)
        data_dest = work_dir / "data"
        if data_src.exists() and not data_dest.exists():
            os.symlink(str(data_src.resolve()), str(data_dest))

    ep = ExecutePreprocessor(
        timeout=timeout, kernel_name="python3",
        allow_errors=True,
        interrupt_on_timeout=True
    )
    result_path = work_dir / "__grading_result__.json"

    try:
        ep.preprocess(nb, {"metadata": {"path": str(work_dir)}})
    except CellExecutionError as ce:
        import sys
        print(f"\n    [DEBUG CellExec] {ce}", file=sys.stderr)
    except Exception as e:
        import sys
        print(f"\n    [DEBUG] {type(e).__name__}: {e}", file=sys.stderr)
        # Kernel crash, timeout, etc. — try to salvage results
        if not result_path.exists():
            print(f"EXEC ERROR ({type(e).__name__})")
            return {"student_id": backup_id, "file": nb_path.name,
                    "tests": {}, "timing": {}, "source": {},
                    "error": f"Kernel crashed: {type(e).__name__}: {e}"}

    # Read results
    if result_path.exists():
        with open(result_path) as f:
            results = json.load(f)
        result_path.unlink()
    else:
        results = {"student_id": backup_id, "tests": {}, "timing": {}, "source": {},
                   "error": "Notebook crashed before tests could run"}

    # Collect visual test outputs
    results["visual_files"] = {}
    visual_tests = spec.get("visual_tests", [])
    for vt in visual_tests:
        vis_path = work_dir / vt["file"]
        if vis_path.exists():
            dest = output_dir / f"{nb_path.stem}_{vt['name']}.png"
            shutil.copy2(str(vis_path), str(dest))
            results["visual_files"][vt["name"]] = str(dest)

    # Extract image and HTML outputs from the test cell (last code cell in executed nb)
    results["test_cell_images"] = []
    results["test_cell_html"] = []
    test_cell_node = nb.cells[-1] if nb.cells else None
    if test_cell_node and test_cell_node.cell_type == 'code':
        for out in test_cell_node.get('outputs', []):
            if out.get('output_type') in ('display_data', 'execute_result'):
                data = out.get('data', {})
                if 'image/png' in data:
                    results["test_cell_images"].append(data['image/png'])
                if 'text/html' in data:
                    results["test_cell_html"].append(data['text/html'])

    results["file"] = nb_path.name
    if not results.get("student_id") or results["student_id"] == "MISSING":
        results["student_id"] = backup_id

    result_file = output_dir / f"{nb_path.stem}_results.json"
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)

    tests = results.get("tests", {})
    n_pass = sum(1 for v in tests.values() if v == "PASS")
    n_total = len(tests)
    print(f"{n_pass}/{n_total} passed  (ID: {results.get('student_id', '?')})")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Grading report generation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_scores(all_results: list, spec: dict) -> list:
    """Compute scores from test results."""
    fmt = detect_spec_format(spec)
    grades = []

    for r in all_results:
        tests = r.get("tests", {})
        row = {
            "student_id": r.get("student_id", "UNKNOWN"),
            "file": r.get("file", ""),
        }

        if fmt == "functions":
            total = 0
            for func_name, func_spec in spec["functions"].items():
                func_max = func_spec["points"]
                func_tests = func_spec.get("tests", [])
                test_names = [t["name"] for t in func_tests]

                # Check banned ops
                banned_key = f"{func_name}_banned_ops"
                if tests.get(banned_key, "").startswith("FAIL"):
                    row[func_name] = 0
                    total += 0
                    continue

                # Count passes
                n_pass = sum(1 for tn in test_names if tests.get(tn) == "PASS")
                n_total = len(test_names)
                if n_total == 0:
                    row[func_name] = 0
                elif n_pass == n_total:
                    row[func_name] = func_max
                else:
                    # Proportional with minimum deduction of 5
                    row[func_name] = max(0, func_max - max(5, round(func_max * (1 - n_pass / n_total))))
                total += row[func_name]
            row["total_grade"] = total

        else:  # tests_array or code_analysis
            total = 0
            bonus = 0
            for test in spec.get("tests", []):
                tname = test["name"]
                pts = test.get("points", 0)
                is_bonus = test.get("bonus", False)
                result = tests.get(tname, "NOT_RUN")

                if result == "PASS":
                    if is_bonus:
                        bonus += pts
                    else:
                        total += pts
                    row[tname] = pts
                elif isinstance(result, str) and result.startswith("BONUS:"):
                    # Variable bonus: "BONUS: 12.5" — awards that many points
                    try:
                        awarded = min(pts, round(float(result.split(":")[1].strip())))
                    except (ValueError, IndexError):
                        awarded = 0
                    if is_bonus:
                        bonus += awarded
                    else:
                        total += awarded
                    row[tname] = awarded
                elif result == "MANUAL":
                    row[tname] = 0  # Manual grading — don't count
                else:
                    # FAIL or SKIP: 0 points
                    row[tname] = 0

            row["total_grade"] = total
            row["bonus"] = bonus

        grades.append(row)
    return grades


def _get_failure_hints(test_name: str, fail_msg: str, test_info: dict) -> list:
    """Generate helpful hints for common failure patterns."""
    hints = []
    msg = fail_msg.lower()

    # Shape mismatches
    if "shape" in msg or "expected" in msg and ("got" in msg):
        hints.append("Check your tensor dimensions — ensure you're handling batch, channel, height, width correctly")
        if "64" in msg or "spatial" in msg:
            hints.append("For super-resolution, the output spatial size should be `scale_factor` times the input")

    # NotImplementedError
    if "notimplementederror" in msg:
        hints.append("This function/class still has the template `raise NotImplementedError()` — replace it with your implementation")

    # Type errors
    if "typeerror" in msg:
        if "argument" in msg:
            hints.append("Check the function signature matches what's expected in the docstring")
        if "not callable" in msg:
            hints.append("Make sure the class/function is properly defined (not overwritten by a variable)")

    # PSNR / quality failures
    if "psnr" in msg or "bicubic" in msg or "must beat" in msg:
        hints.append("Your model isn't learning well enough. Try: more epochs, better learning rate, check residual connection")
        hints.append("Make sure `train_zssr` uses L1Loss (not MSE) and Adam optimizer")
        hints.append("Verify that your dataset returns HR/LR pairs with correct spatial relationship")

    # Missing keys
    if "missing key" in msg or "key:" in msg:
        hints.append("Your function must return a dict with all required keys — check the docstring")

    # Module / inheritance
    if "nn.module" in msg or "must be" in msg and "module" in msg:
        hints.append("Your class must inherit from `nn.Module` — use `class YourClass(nn.Module):`")

    # Dataset issues
    if "dataset" in msg or "dataloader" in msg:
        hints.append("Make sure your Dataset returns a dict with 'HR' and 'LR' keys")
        hints.append("HR crops should be `scale_factor` times larger than LR crops spatially")

    # Attribute errors
    if "attributeerror" in msg:
        hints.append("A required attribute or method is missing — check your class definition")

    # Import / name errors
    if "nameerror" in msg:
        hints.append("A required function or class is not defined — make sure all cells are run in order")

    return hints


def build_graded_notebook(nb_path: Path, results: dict, spec: dict, grades_row: dict) -> nbformat.NotebookNode:
    """Prepend grading report cells to a student notebook."""
    nb = nbformat.read(str(nb_path), as_version=4)
    cells = []

    fmt = detect_spec_format(spec)
    total = grades_row["total_grade"]
    bonus = grades_row.get("bonus", 0)
    max_pts = spec.get("total_points", 100)
    tests = results.get("tests", {})

    # Header cell
    grade_str = f"{total}/{max_pts}"
    if bonus > 0:
        grade_str += f" + {bonus} bonus"

    header_lines = [
        "# Grading Report",
        "",
        f"## Grade: {grade_str}",
        "",
    ]

    if fmt == "functions":
        header_lines.extend([
            "| Function | Score | Status |",
            "|----------|-------|--------|",
        ])
        for func_name, func_spec in spec["functions"].items():
            func_score = grades_row.get(func_name, 0)
            func_max = func_spec["points"]
            func_tests = func_spec.get("tests", [])
            test_names = [t["name"] for t in func_tests]
            all_pass = all(tests.get(tn) == "PASS" for tn in test_names)
            banned_key = f"{func_name}_banned_ops"
            if tests.get(banned_key, "").startswith("FAIL"):
                status = "✗ banned ops"
            elif all_pass:
                status = "✓ Correct"
            else:
                fails = [tn for tn in test_names if tests.get(tn, "").startswith("FAIL")]
                status = f"~ {len(fails)} test(s) failed"
            header_lines.append(f"| `{func_name}` | {func_score}/{func_max} | {status} |")
    else:
        header_lines.extend([
            "| Test | Points | Result |",
            "|------|--------|--------|",
        ])
        for test in spec.get("tests", []):
            tname = test["name"]
            pts = test.get("points", 0)
            is_bonus = test.get("bonus", False)
            result = tests.get(tname, "NOT_RUN")
            awarded = grades_row.get(tname, 0)
            pts_str = f"{pts} (bonus)" if is_bonus else str(pts)
            if result == "PASS":
                status = "✓ PASS"
            elif result == "MANUAL":
                status = "⏭ Manual"
            elif result.startswith("FAIL"):
                status = f"✗ {result.replace('FAIL: ', '')[:60]}"
            else:
                status = result
            header_lines.append(f"| `{tname}` | {awarded}/{pts_str} | {status} |")

    header_lines.extend(["", "---"])
    cells.append(nbformat.v4.new_markdown_cell("\n".join(header_lines)))

    # Include test cell images and HTML (e.g., ZSSR comparison, GIFs) after grade table
    test_images = results.get("test_cell_images", [])
    test_html = results.get("test_cell_html", [])
    if test_images or test_html:
        img_cell = nbformat.v4.new_code_cell(source="# Grading test results (auto-generated)")
        img_cell.outputs = []
        for img_b64 in test_images:
            img_cell.outputs.append(nbformat.v4.new_output(
                output_type='display_data',
                data={'image/png': img_b64, 'text/plain': ['<grading result image>']},
                metadata={'image/png': {'width': 800}}
            ))
        for html_str in test_html:
            html_data = html_str if isinstance(html_str, str) else str(html_str)
            img_cell.outputs.append(nbformat.v4.new_output(
                output_type='display_data',
                data={'text/html': html_data, 'text/plain': ['<grading result>']},
            ))
        img_cell.execution_count = None
        cells.append(img_cell)

    # Per-test detail cells for failures — include analysis and suggestions
    for test_info in _iter_tests(spec):
        tname = test_info["name"]
        result = tests.get(tname, "NOT_RUN")
        if result not in ("PASS", "MANUAL", "NOT_RUN"):
            fail_msg = result.replace("FAIL: ", "") if result.startswith("FAIL:") else result
            detail = [
                f"### `{tname}` — FAIL",
                "",
                f"**Result:** `{fail_msg}`",
                "",
                f"**Description:** {test_info.get('description', '')}",
                "",
            ]
            # Add hints based on common failure patterns
            hints = _get_failure_hints(tname, fail_msg, test_info)
            if hints:
                detail.append("**Possible issues:**")
                for h in hints:
                    detail.append(f"- {h}")
                detail.append("")
            detail.append("---")
            cells.append(nbformat.v4.new_markdown_cell("\n".join(detail)))

    # Check if student notebook has any outputs — warn if not
    has_student_outputs = any(
        c.cell_type == 'code' and c.get('outputs', [])
        for c in nb.cells
    )

    # Separator
    sep_lines = ["---\n---\n", "# Original Submission\n"]
    if not has_student_outputs:
        sep_lines.append(
            "> **Note:** This notebook was submitted without cell outputs. "
            "Students are required to submit notebooks with all outputs "
            "(training curves, visualizations, results). "
            "Missing outputs may indicate the notebook was not fully executed before submission.\n"
        )
    sep_lines.append("*Everything below is the student's original work, unmodified.*\n\n---")
    cells.append(nbformat.v4.new_markdown_cell("\n".join(sep_lines)))

    nb.cells = cells + nb.cells
    return nb


def _iter_tests(spec: dict):
    """Iterate over all tests regardless of spec format."""
    fmt = detect_spec_format(spec)
    if fmt == "functions":
        for func_name, func_spec in spec["functions"].items():
            for t in func_spec.get("tests", []):
                yield t
    else:
        for t in spec.get("tests", []):
            yield t


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Run notebook grading tests")
    parser.add_argument("submissions_dir", help="Folder of student .ipynb files")
    parser.add_argument("spec_file", help="Assignment spec JSON file")
    parser.add_argument("--output", "-o", default="grading_results",
                        help="Output directory for results (default: grading_results)")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Per-notebook timeout in seconds (default: 300)")
    parser.add_argument("--data-dir", default=None,
                        help="Path to data directory (MNIST, CIFAR, etc.)")
    parser.add_argument("--grade", action="store_true",
                        help="Also produce graded notebooks and CSV report")
    args = parser.parse_args()

    sub_dir = Path(args.submissions_dir)
    spec_path = Path(args.spec_file)
    spec = load_spec(spec_path)
    spec["_spec_dir"] = str(spec_path.parent)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    notebooks = sorted(sub_dir.glob("*.ipynb"))
    if not notebooks:
        print(f"No .ipynb files found in {sub_dir}")
        sys.exit(1)

    fmt = detect_spec_format(spec)
    name = spec.get("assignment_name", spec.get("name", "Unknown"))
    print(f"Grading: {name} [{fmt} format]")
    print(f"Found {len(notebooks)} submission(s)\n")

    all_results = []
    for nb_path in notebooks:
        if fmt == "code_analysis":
            result = grade_code_analysis(nb_path, spec, output_dir)
        else:
            result = grade_notebook(nb_path, spec, output_dir,
                                    timeout=args.timeout, data_dir=args.data_dir)
        all_results.append(result)

    # Write combined results
    combined_path = output_dir / "all_results.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Compute scores
    grades = compute_scores(all_results, spec)

    # Print summary table
    max_pts = spec.get("total_points", 100)
    print(f"\n{'='*60}")
    print(f"{'Student ID':<25} {'Grade':>8}  {'Tests'}")
    print("-" * 60)
    for g, r in zip(grades, all_results):
        tests = r.get("tests", {})
        n_pass = sum(1 for v in tests.values() if v == "PASS")
        n_total = sum(1 for v in tests.values() if v not in ("MANUAL", "SKIP"))
        bonus_str = f" +{g['bonus']}" if g.get("bonus") else ""
        print(f"{g['student_id']:<25} {g['total_grade']:>4}/{max_pts}{bonus_str:<5} {n_pass}/{n_total} passed")

    n_total = len(all_results)
    n_perfect = sum(1 for r in all_results
                    if r.get("tests") and
                    all(v == "PASS" for v in r["tests"].values() if v not in ("MANUAL", "SKIP")))
    n_crash = sum(1 for r in all_results if r.get("error"))
    print(f"\nSummary: {n_perfect}/{n_total} perfect, {n_crash} crashes")
    print(f"Details: {combined_path}")

    # Write CSV report
    csv_path = output_dir / "report.csv"
    if grades:
        fieldnames = list(grades[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for g in sorted(grades, key=lambda x: -x["total_grade"]):
                writer.writerow(g)
        print(f"CSV report: {csv_path}")

    # Produce graded notebooks if requested
    if args.grade:
        graded_dir = output_dir / "graded"
        graded_dir.mkdir(exist_ok=True)
        for r, g in zip(all_results, grades):
            nb_path = sub_dir / r["file"]
            if nb_path.exists():
                graded_nb = build_graded_notebook(nb_path, r, spec, g)
                graded_path = graded_dir / f"{nb_path.stem}_graded.ipynb"
                nbformat.write(graded_nb, str(graded_path))
        print(f"Graded notebooks: {graded_dir}/")


if __name__ == "__main__":
    main()
