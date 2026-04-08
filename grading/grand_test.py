#!/usr/bin/env python3
"""
Grand Test: End-to-end grading pipeline validation.

For each HW notebook (6 total):
1. Start from clean template notebook
2. Generate 2 mock student submissions (perfect + one error case)
3. Execute them to produce outputs (simulating student workflow)
4. Grade the executed submissions
5. Validate grades, graded notebooks, and outputs
6. Produce a final report

Usage:
    python grand_test.py [--hw hw3] [--type applied] [--skip-execute] [--skip-generate] [--fast]

Requires conda env: /Users/assafshocher/anaconda3/envs/torch/bin/python
"""

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from copy import deepcopy
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

REPO_ROOT = Path(__file__).parent.parent
HW_DIR = REPO_ROOT / "hw"
GRADING_DIR = REPO_ROOT / "grading"
DATA_DIR = GRADING_DIR / "data"
ENGINE = GRADING_DIR / "engine" / "run_tests.py"

# Map of (hw, type) -> spec file
SPECS = {
    ("hw1", "scratch"): GRADING_DIR / "hw1" / "specs" / "hw1_scratch_spec.json",
    ("hw1", "applied"): GRADING_DIR / "hw1" / "specs" / "hw1_applied_spec.json",
    ("hw2", "scratch"): GRADING_DIR / "hw2" / "specs" / "hw2_scratch_spec.json",
    ("hw2", "applied"): GRADING_DIR / "hw2" / "specs" / "hw2_applied_spec.json",
    ("hw3", "scratch"): GRADING_DIR / "hw3" / "specs" / "hw3_scratch_spec.json",
    ("hw3", "applied"): GRADING_DIR / "hw3" / "specs" / "hw3_applied_spec.json",
}

# Timeouts per assignment
TIMEOUTS = {
    ("hw1", "scratch"): 120,
    ("hw1", "applied"): 300,
    ("hw2", "scratch"): 300,
    ("hw2", "applied"): 600,
    ("hw3", "scratch"): 300,
    ("hw3", "applied"): 900,
}

# Expected grades for perfect mocks
EXPECTED_PERFECT_GRADE = {
    ("hw1", "scratch"): 100,
    ("hw1", "applied"): 100,
    ("hw2", "scratch"): 100,
    ("hw2", "applied"): 100,
    ("hw3", "scratch"): 100,
    ("hw3", "applied"): 100,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Mock Student Implementations
# ═══════════════════════════════════════════════════════════════════════════════

# Each entry: function/class name pattern -> replacement code
# We match by finding the cell containing "def <name>" or "class <name>"

MOCK_IMPLEMENTATIONS = {}

# --- HW1 Scratch ---
MOCK_IMPLEMENTATIONS[("hw1", "scratch", "perfect")] = {
    "def conv2d_loops": '''def conv2d_loops(image, kernel):
    """2D convolution using for loops with kernel flipping. Valid mode (no padding)."""
    H, W = image.shape
    kH, kW = kernel.shape
    H_out, W_out = H - kH + 1, W - kW + 1
    kernel_flipped = kernel.flip(0).flip(1)
    output = torch.zeros(H_out, W_out)
    for i in range(H_out):
        for j in range(W_out):
            output[i, j] = (image[i:i+kH, j:j+kW] * kernel_flipped).sum()
    return output''',

    "def im2patches": '''def im2patches(image, kH, kW):
    """Extract patches from image for vectorized conv."""
    H, W = image.shape
    H_out, W_out = H - kH + 1, W - kW + 1
    patches = torch.zeros(H_out * W_out, kH * kW)
    for i in range(H_out):
        for j in range(W_out):
            patches[i * W_out + j] = image[i:i+kH, j:j+kW].reshape(-1)
    return patches''',

    "def conv2d_vectorized": '''def conv2d_vectorized(image, kernel):
    """Vectorized 2D convolution using im2patches."""
    kH, kW = kernel.shape
    H_out = image.shape[0] - kH + 1
    W_out = image.shape[1] - kW + 1
    patches = im2patches(image, kH, kW)
    kernel_flipped = kernel.flip(0).flip(1).reshape(-1)
    result = patches @ kernel_flipped
    return result.reshape(H_out, W_out)''',
}

MOCK_IMPLEMENTATIONS[("hw1", "scratch", "error")] = {
    **MOCK_IMPLEMENTATIONS[("hw1", "scratch", "perfect")],
    # Bug: no kernel flip (cross-correlation instead of convolution)
    "def conv2d_loops": '''def conv2d_loops(image, kernel):
    """2D convolution — BUG: no kernel flip (cross-correlation)."""
    H, W = image.shape
    kH, kW = kernel.shape
    H_out, W_out = H - kH + 1, W - kW + 1
    output = torch.zeros(H_out, W_out)
    for i in range(H_out):
        for j in range(W_out):
            output[i, j] = (image[i:i+kH, j:j+kW] * kernel).sum()
    return output''',
}

# --- HW1 Applied ---
MOCK_IMPLEMENTATIONS[("hw1", "applied", "perfect")] = {
    "def compute_energy": '''def compute_energy(gray):
    """Compute energy map using Sobel filters."""
    sx = torch.tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]])
    sy = torch.tensor([[-1.,-2.,-1.],[0.,0.,0.],[1.,2.,1.]])
    gx = conv2d(gray, sx)
    gy = conv2d(gray, sy)
    e = gx.abs() + gy.abs()
    return F.pad(e.unsqueeze(0), (1, 1, 1, 1), mode='replicate').squeeze(0)''',

    "def find_seam": '''def find_seam(energy):
    """Find minimum-energy vertical seam using dynamic programming."""
    H, W = energy.shape
    M = torch.zeros_like(energy)
    M[0] = energy[0]
    backtrack = torch.zeros(H, W, dtype=torch.long)
    for i in range(1, H):
        left = F.pad(M[i-1, :W-1].unsqueeze(0), (1, 0), value=float('inf')).squeeze(0)
        right = F.pad(M[i-1, 1:].unsqueeze(0), (0, 1), value=float('inf')).squeeze(0)
        center = M[i-1]
        stacked = torch.stack([left, center, right], dim=0)
        vals, dirs = stacked.min(dim=0)
        M[i] = energy[i] + vals
        backtrack[i] = dirs - 1
    seam = torch.zeros(H, dtype=torch.long)
    seam[-1] = M[-1].argmin()
    for i in range(H-2, -1, -1):
        seam[i] = seam[i+1] + backtrack[i+1, seam[i+1]]
        seam[i] = seam[i].clamp(0, W-1)
    return seam''',

    "def remove_seam": '''def remove_seam(image, seam):
    """Remove a vertical seam from an image."""
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image).float()
    if image.ndim == 2:
        H, W = image.shape
        mask = torch.ones(H, W, dtype=torch.bool)
        for i in range(H):
            mask[i, seam[i]] = False
        return image[mask].reshape(H, W-1)
    else:
        H, W, C = image.shape
        mask = torch.ones(H, W, dtype=torch.bool)
        for i in range(H):
            mask[i, seam[i]] = False
        mask = mask.unsqueeze(-1).expand_as(image)
        return image[mask].reshape(H, W-1, C)''',

    "def seam_carve": '''def seam_carve(image, num_seams, show_progress=False):
    """Remove num_seams vertical seams from an image."""
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image).float()
    result = image.clone()
    for i in range(num_seams):
        if result.ndim == 3:
            gray = to_grayscale(result)
        else:
            gray = result
        energy = compute_energy(gray)
        seam = find_seam(energy)
        result = remove_seam(result, seam)
    return result''',
}

MOCK_IMPLEMENTATIONS[("hw1", "applied", "error")] = dict(
    MOCK_IMPLEMENTATIONS[("hw1", "applied", "perfect")],
    **{
    # Bug: greedy seam instead of DP
    "def find_seam": '''def find_seam(energy):
    """Find seam — BUG: greedy approach instead of DP."""
    H, W = energy.shape
    seam = torch.zeros(H, dtype=torch.long)
    seam[0] = energy[0].argmin()
    for i in range(1, H):
        j = seam[i-1].item()
        lo = max(0, j-1)
        hi = min(W, j+2)
        seam[i] = lo + energy[i, lo:hi].argmin()
    return seam''',
})

# --- HW2 Scratch ---
# HW2 scratch is complex — many interdependent functions.
# We use the existing mock submissions rather than generating from scratch.

# --- HW2 Applied ---
# HW2 applied needs full training — we generate complete notebooks.

# --- HW3 Scratch ---
# Use existing mocks

# --- HW3 Applied ---
# Use existing mocks


# ═══════════════════════════════════════════════════════════════════════════════
# Core Functions
# ═══════════════════════════════════════════════════════════════════════════════

def find_and_replace_cell(nb, pattern, new_code):
    """Find a code cell matching pattern and replace its source."""
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'code' and pattern in cell.source:
            cell.source = new_code
            return True
    return False


def set_student_id(nb, student_id):
    """Set STUDENT_ID in notebook."""
    for cell in nb.cells:
        if cell.cell_type == 'code' and 'STUDENT_ID' in cell.source:
            lines = cell.source.split('\n')
            for j, line in enumerate(lines):
                if line.strip().startswith('STUDENT_ID'):
                    lines[j] = f'STUDENT_ID = "{student_id}"'
            cell.source = '\n'.join(lines)
            return True
    return False


def generate_mock(template_path, output_path, implementations, student_id):
    """Generate a mock submission from template + implementations."""
    nb = nbformat.read(str(template_path), as_version=4)

    set_student_id(nb, student_id)

    for pattern, code in implementations.items():
        if not find_and_replace_cell(nb, pattern, code):
            print(f"    WARNING: Could not find cell matching '{pattern[:40]}...'")

    # Clear all outputs (fresh start)
    for cell in nb.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None

    nbformat.write(nb, str(output_path))
    return True


def execute_mock(nb_path, timeout=600, fast=False):
    """Execute a mock notebook to produce outputs.

    If fast=True, only execute definition/import cells and skip heavy training
    cells. This is sufficient for grand test validation since the grading engine
    re-executes definitions independently anyway.
    """
    import re
    nb = nbformat.read(str(nb_path), as_version=4)

    # Symlink data dir if needed
    work_dir = nb_path.parent
    data_dest = work_dir / "data"
    if not data_dest.exists() and DATA_DIR.exists():
        os.symlink(str(DATA_DIR.resolve()), str(data_dest))

    if fast:
        # Only keep cells that define functions, classes, imports, or setup.
        # Skip heavy training/experiment cells to avoid long execution times.
        # We still want SOME outputs so the graded notebook doesn't look empty.
        kept_cells = []
        skipped = 0
        for cell in nb.cells:
            if cell.cell_type != 'code':
                kept_cells.append(cell)
                continue
            src = cell.source.strip()
            if not src:
                kept_cells.append(cell)
                continue
            # Always keep: definitions, imports, setup, STUDENT_ID, short cells
            is_definition = bool(re.search(r'^(def |class )', src, re.MULTILINE))
            lines = [l.strip() for l in src.split('\n') if l.strip() and not l.strip().startswith('#')]
            is_import = all(any(l.startswith(p) for p in ('import ', 'from ', '%', 'device', 'STUDENT_ID', 'warnings')) for l in lines) if lines else False
            is_short = len(lines) <= 5

            if is_definition or is_import or is_short:
                kept_cells.append(cell)
            else:
                # Skip heavy cells but add a placeholder output
                cell.outputs = [nbformat.v4.new_output(
                    output_type='stream', name='stdout',
                    text=f'[Grand test: skipped execution of this cell for speed]\n'
                )]
                kept_cells.append(cell)
                skipped += 1

        # Create a temp notebook with only kept cells for execution
        exec_nb = deepcopy(nb)
        exec_cells = [c for c in kept_cells if c.cell_type == 'code' and
                      not any('[Grand test: skipped' in str(o.get('text', '')) for o in c.get('outputs', []))]
        # Build execution notebook: only the cells we want to run
        exec_nb.cells = [c for c in kept_cells if c.cell_type != 'code' or
                         not any('[Grand test: skipped' in str(o.get('text', '')) for o in c.get('outputs', []))]

        ep = ExecutePreprocessor(
            timeout=min(timeout, 120),  # Fast mode: 2 min max
            kernel_name="python3",
            allow_errors=True,
            interrupt_on_timeout=True,
        )
        try:
            ep.preprocess(exec_nb, {"metadata": {"path": str(work_dir)}})
        except Exception as e:
            print(f"    Execution issue: {type(e).__name__}: {str(e)[:80]}")

        # Merge executed outputs back into original notebook
        exec_idx = 0
        for cell in kept_cells:
            if cell.cell_type == 'code' and not any('[Grand test: skipped' in str(o.get('text', '')) for o in cell.get('outputs', [])):
                if exec_idx < len(exec_nb.cells):
                    matching = [c for c in exec_nb.cells if c.cell_type == 'code']
                    if exec_idx < len(matching):
                        cell.outputs = matching[exec_idx].get('outputs', [])
                        cell.execution_count = matching[exec_idx].get('execution_count')
                exec_idx += 1
        nb.cells = kept_cells
        if skipped:
            print(f"    (fast mode: skipped {skipped} heavy cells)")
    else:
        ep = ExecutePreprocessor(
            timeout=timeout,
            kernel_name="python3",
            allow_errors=True,
            interrupt_on_timeout=True,
        )
        try:
            ep.preprocess(nb, {"metadata": {"path": str(work_dir)}})
        except Exception as e:
            print(f"    Execution issue: {type(e).__name__}: {str(e)[:100]}")

    # Count cells with outputs
    code_cells = [c for c in nb.cells if c.cell_type == 'code']
    cells_with_output = sum(1 for c in code_cells if c.get('outputs', []))
    print(f"    {cells_with_output}/{len(code_cells)} code cells produced outputs")

    nbformat.write(nb, str(nb_path))
    return cells_with_output > 0


def run_grading(mock_dir, spec_path, output_dir, timeout=300):
    """Run the grading engine on mock submissions."""
    import subprocess

    cmd = [
        sys.executable,
        str(ENGINE),
        str(mock_dir),
        str(spec_path),
        "--output", str(output_dir),
        "--timeout", str(timeout),
        "--data-dir", str(DATA_DIR),
        "--grade",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 120)
    return result


def validate_results(output_dir, spec_path, expected_perfect_grade):
    """Validate grading results."""
    issues = []

    # Check all_results.json exists
    results_file = output_dir / "all_results.json"
    if not results_file.exists():
        issues.append("all_results.json not found")
        return issues

    with open(results_file) as f:
        all_results = json.load(f)

    # Check report.csv exists
    if not (output_dir / "report.csv").exists():
        issues.append("report.csv not found")

    # Check graded notebooks exist
    graded_dir = output_dir / "graded"
    if not graded_dir.exists():
        issues.append("graded/ directory not found")
    else:
        graded_nbs = list(graded_dir.glob("*_graded.ipynb"))
        if len(graded_nbs) != len(all_results):
            issues.append(f"Expected {len(all_results)} graded notebooks, found {len(graded_nbs)}")

    # Load spec for scoring
    with open(spec_path) as f:
        spec = json.load(f)

    from engine.run_tests import compute_scores, detect_spec_format
    grades = compute_scores(all_results, spec)

    for result, grade in zip(all_results, grades):
        student_id = result.get("student_id", "UNKNOWN")
        total = grade["total_grade"]
        bonus = grade.get("bonus", 0)
        tests = result.get("tests", {})
        n_pass = sum(1 for v in tests.values() if v == "PASS")
        n_total = len(tests)

        # Check perfect student gets expected grade
        is_perfect = "perfect" in student_id.lower() or "perfect" in result.get("file", "").lower()
        if is_perfect and total < expected_perfect_grade:
            issues.append(f"{student_id}: Expected {expected_perfect_grade}, got {total}")

        # Check for crashes
        if result.get("error"):
            issues.append(f"{student_id}: Crashed with {result['error'][:100]}")

        # Check graded notebook has proper structure
        if graded_dir.exists():
            graded_path = graded_dir / f"{Path(result['file']).stem}_graded.ipynb"
            if graded_path.exists():
                gnb = nbformat.read(str(graded_path), as_version=4)

                # Must have grading report at top
                if not gnb.cells or "Grading Report" not in gnb.cells[0].source:
                    issues.append(f"{student_id}: Graded notebook missing 'Grading Report' header")

                # Check for test result images/html
                has_visual = any(
                    c.cell_type == 'code' and
                    any('image/png' in str(o.get('data', {})) or
                        'text/html' in str(o.get('data', {}))
                        for o in c.get('outputs', []))
                    for c in gnb.cells[:5]  # Check first few cells
                )
                is_perfect_nb = "perfect" in student_id.lower() or "perfect" in result.get("file", "").lower()
                if not has_visual and is_perfect_nb and any(t.get("code", "").find("plt.") >= 0 for t in spec.get("tests", [])):
                    issues.append(f"{student_id}: Graded notebook missing visual test results")

                # Check for missing output warning (if student had no outputs)
                student_has_outputs = any(
                    c.cell_type == 'code' and c.get('outputs', [])
                    for c in gnb.cells[4:]  # Skip grading cells
                )
                has_warning = any(
                    "submitted without cell outputs" in c.source
                    for c in gnb.cells if c.cell_type == 'markdown'
                )
                if not student_has_outputs and not has_warning:
                    issues.append(f"{student_id}: Missing 'no outputs' warning in graded notebook")

    return issues


# ═══════════════════════════════════════════════════════════════════════════════
# Grand Test Runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_grand_test(targets=None, skip_generate=False, skip_execute=False, fast=False):
    """Run the complete grand test."""

    if targets is None:
        targets = list(SPECS.keys())

    results_summary = {}
    all_issues = []
    start_time = time.time()

    timing = {}

    for hw, typ in targets:
        key = f"{hw}_{typ}"
        step_start = time.time()
        print(f"\n{'='*70}")
        print(f"  GRAND TEST: {hw} {typ}")
        print(f"{'='*70}")

        spec_path = SPECS[(hw, typ)]
        template_path = HW_DIR / hw / ("applied.ipynb" if typ == "applied" else "from_scratch.ipynb")
        timeout = TIMEOUTS.get((hw, typ), 600)

        # Standard output locations
        mock_dir = GRADING_DIR / hw / f"mock_submissions_{typ}"
        results_dir = GRADING_DIR / hw / f"graded_{typ}"

        if not skip_generate:
            impl_key_perfect = (hw, typ, "perfect")
            impl_key_error = (hw, typ, "error")

            if impl_key_perfect in MOCK_IMPLEMENTATIONS:
                # Generate fresh mocks from template
                print(f"\n  Step 1: Generating mocks from template...")
                if results_dir.exists():
                    shutil.rmtree(str(results_dir))
                # Clear old mocks and regenerate
                if mock_dir.exists():
                    shutil.rmtree(str(mock_dir))
                mock_dir.mkdir(parents=True, exist_ok=True)

                generate_mock(
                    template_path, mock_dir / "student_perfect.ipynb",
                    MOCK_IMPLEMENTATIONS[impl_key_perfect], "perfect_student"
                )
                print(f"    Generated student_perfect.ipynb")

                if impl_key_error in MOCK_IMPLEMENTATIONS:
                    generate_mock(
                        template_path, mock_dir / "student_error.ipynb",
                        MOCK_IMPLEMENTATIONS[impl_key_error], "error_student"
                    )
                    print(f"    Generated student_error.ipynb")
            else:
                # Use existing mocks — copy 2 (perfect + error) to a temp grading dir
                print(f"\n  Step 1: Using existing mock submissions...")
                if not mock_dir.exists():
                    print(f"    ERROR: No existing mocks at {mock_dir}")
                    results_summary[key] = {"status": "SKIP", "reason": "no mocks"}
                    continue
                # Select 2 mocks: perfect + one error
                nbs = sorted(mock_dir.glob("*.ipynb"))
                perfect = [n for n in nbs if "perfect" in n.stem]
                errors = [n for n in nbs if "perfect" not in n.stem]
                selected = perfect[:1] + errors[:1]
                if not selected:
                    selected = nbs[:2]
                # Use a temp working directory for grading (don't touch originals)
                grand_mock_dir = GRADING_DIR / hw / f"_grand_test_{typ}"
                if grand_mock_dir.exists():
                    shutil.rmtree(str(grand_mock_dir))
                grand_mock_dir.mkdir(parents=True, exist_ok=True)
                for nb_path in selected:
                    shutil.copy2(str(nb_path), str(grand_mock_dir / nb_path.name))
                    print(f"    Selected {nb_path.name}")
                mock_dir = grand_mock_dir  # Use temp dir for execution + grading
                if results_dir.exists():
                    shutil.rmtree(str(results_dir))

            # Copy test files if needed
            spec_dir = spec_path.parent
            with open(spec_path) as f:
                spec = json.load(f)
            for fname in spec.get("test_files", []):
                src = spec_dir / fname
                if src.exists():
                    shutil.copy2(str(src), str(mock_dir / fname))

            # Copy support files from template dir (images, data files etc.)
            template_dir = template_path.parent
            for f in template_dir.iterdir():
                if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff',
                                         '.npy', '.pt', '.pth', '.csv', '.txt'):
                    dest = mock_dir / f.name
                    if not dest.exists():
                        shutil.copy2(str(f), str(dest))

        if not skip_execute:
            # Execute mocks to produce outputs (fast mode: skip heavy cells)
            all_nbs = sorted(mock_dir.glob("*.ipynb"))
            mode_label = "fast mode" if fast else "full execution"
            print(f"\n  Step 2: Executing {len(all_nbs)} mocks ({mode_label})...")
            exec_start = time.time()
            for nb_path in all_nbs:
                nb_start = time.time()
                print(f"    Executing {nb_path.name}...", end="", flush=True)
                try:
                    execute_mock(nb_path, timeout=timeout, fast=fast)
                    print(f" ({time.time()-nb_start:.1f}s)")
                except Exception as e:
                    print(f" FAILED ({time.time()-nb_start:.1f}s): {e}")
            print(f"    Execution total: {time.time()-exec_start:.1f}s")

            # Copy executed notebooks (with outputs) back to original mock dir
            orig_mock_dir = GRADING_DIR / hw / f"mock_submissions_{typ}"
            if orig_mock_dir.exists() and mock_dir != orig_mock_dir:
                for nb_path in all_nbs:
                    dest = orig_mock_dir / nb_path.name
                    if dest.exists():
                        shutil.copy2(str(nb_path), str(dest))

        # Grade
        print(f"\n  Step 3: Grading...")
        grade_start = time.time()
        try:
            grade_result = run_grading(mock_dir, spec_path, results_dir, timeout=timeout)
            print(f"    Grading took: {time.time()-grade_start:.1f}s")
            print(grade_result.stdout)
            if grade_result.stderr:
                # Only show non-debug stderr
                for line in grade_result.stderr.split('\n'):
                    if line and 'DEBUG' not in line:
                        print(f"    STDERR: {line}")
        except Exception as e:
            print(f"    Grading failed: {e}")
            results_summary[key] = {"status": "FAIL", "reason": str(e)}
            all_issues.append(f"{key}: Grading crashed: {e}")
            continue

        # Validate
        print(f"\n  Step 4: Validating...")
        expected = EXPECTED_PERFECT_GRADE.get((hw, typ), 100)
        issues = validate_results(results_dir, spec_path, expected)

        if issues:
            for issue in issues:
                print(f"    ISSUE: {issue}")
                all_issues.append(f"{key}: {issue}")
            results_summary[key] = {"status": "ISSUES", "issues": issues}
        else:
            print(f"    All validations passed!")
            results_summary[key] = {"status": "PASS"}

        step_elapsed = time.time() - step_start
        timing[key] = round(step_elapsed, 1)
        print(f"\n  Time for {key}: {step_elapsed:.1f}s")

        # Load grades for summary
        results_file = results_dir / "all_results.json"
        if results_file.exists():
            with open(results_file) as f:
                all_r = json.load(f)
            with open(spec_path) as f:
                spec = json.load(f)
            from engine.run_tests import compute_scores
            grades = compute_scores(all_r, spec)
            results_summary[key]["grades"] = [
                {
                    "student": r.get("student_id", "?"),
                    "total": g["total_grade"],
                    "bonus": g.get("bonus", 0),
                    "tests_passed": sum(1 for v in r.get("tests", {}).values() if v == "PASS"),
                    "tests_total": len(r.get("tests", {})),
                }
                for r, g in zip(all_r, grades)
            ]

    # Final report
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"  GRAND TEST REPORT")
    print(f"{'='*70}")
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}m)\n")

    print(f"{'Assignment':<20} {'Status':<10} {'Time':>6}  {'Details'}")
    print("-" * 80)
    for key, info in results_summary.items():
        status = info["status"]
        t = timing.get(key, 0)
        time_str = f"{t:.0f}s"
        details = ""
        if "grades" in info:
            for g in info["grades"]:
                details += f"{g['student']}: {g['total']}/{100}"
                if g['bonus']:
                    details += f"+{g['bonus']}"
                details += f" ({g['tests_passed']}/{g['tests_total']}), "
            details = details.rstrip(", ")
        elif "reason" in info:
            details = info["reason"]
        elif "issues" in info:
            details = f"{len(info['issues'])} issue(s)"
        print(f"{key:<20} {status:<10} {time_str:>6}  {details}")

    if all_issues:
        print(f"\n{'='*70}")
        print(f"  ALL ISSUES ({len(all_issues)}):")
        print(f"{'='*70}")
        for issue in all_issues:
            print(f"  - {issue}")

    # Clean up temp directories
    for hw in ["hw1", "hw2", "hw3"]:
        for typ in ["scratch", "applied"]:
            tmp = GRADING_DIR / hw / f"_grand_test_{typ}"
            if tmp.exists():
                shutil.rmtree(str(tmp))

    # Save report
    report_path = GRADING_DIR / "grand_test_report.json"
    with open(report_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": round(elapsed),
            "timing_per_assignment": timing,
            "results": results_summary,
            "issues": all_issues,
        }, f, indent=2)
    print(f"\nReport saved: {report_path}")

    return len(all_issues) == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grand Test: End-to-end grading validation")
    parser.add_argument("--hw", type=str, help="Specific HW (hw1, hw2, hw3)")
    parser.add_argument("--type", type=str, help="Specific type (applied, scratch)")
    parser.add_argument("--skip-generate", action="store_true", help="Skip mock generation")
    parser.add_argument("--skip-execute", action="store_true", help="Skip mock execution")
    parser.add_argument("--fast", action="store_true", help="Fast mode: skip heavy cells during mock execution")
    args = parser.parse_args()

    targets = None
    if args.hw and args.type:
        targets = [(args.hw, args.type)]
    elif args.hw:
        targets = [(args.hw, t) for t in ["scratch", "applied"] if (args.hw, t) in SPECS]

    os.chdir(str(GRADING_DIR))
    success = run_grand_test(targets=targets, skip_generate=args.skip_generate,
                             skip_execute=args.skip_execute, fast=args.fast)
    sys.exit(0 if success else 1)
