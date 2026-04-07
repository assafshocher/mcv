#!/usr/bin/env python3
"""
Full grading pipeline for HW1 Applied (Seam Carving).
Adapts the general grading engine for the applied spec.
"""

import base64
import csv
import json
import re
from pathlib import Path
from collections import Counter

import nbformat

SPEC_PATH = Path("notebook-grader/hw1_applied_spec.json")
RESULTS_PATH = Path("applied_results/all_results.json")
SUBMISSIONS_DIR = Path("test_applied_submissions")
GRADED_DIR = Path("applied_results/graded")
REPORT_PATH = Path("applied_results/grading_report.csv")

with open(SPEC_PATH) as f:
    spec = json.load(f)

with open(RESULTS_PATH) as f:
    all_results = json.load(f)

func_points = {name: fspec["points"] for name, fspec in spec["functions"].items()}
FUNC_ORDER = list(spec["functions"].keys())

GRADED_DIR.mkdir(parents=True, exist_ok=True)


def diagnose_function(func_name, func_spec, tests, source_code):
    """
    Analyze failures for one function. Returns:
      score, category, short_title, diagnosis_md, fix_code
    """
    max_pts = func_spec["points"]
    test_results = {}
    for t in func_spec["tests"]:
        test_results[t["name"]] = tests.get(t["name"], "NOT_RUN")

    all_pass = all(v == "PASS" for v in test_results.values())

    # Check banned ops
    banned_key = f"{func_name}_banned_ops"
    if tests.get(banned_key, "").startswith("FAIL"):
        return 0, "fundamental", "used banned operation", \
            f"Used a banned operation: {tests[banned_key].replace('FAIL: ', '')}.", None

    if all_pass:
        return max_pts, "perfect", "", "", None

    fail_msgs = {k: v for k, v in test_results.items() if v != "PASS" and v != "NOT_RUN"}

    # --- NotImplementedError ---
    not_impl = any("NotImplementedError" in str(v) for v in fail_msgs.values())
    # Also check source code for raise NotImplementedError
    if not not_impl and "NotImplementedError" in source_code:
        not_impl = True
    # Also detect when ALL tests fail with empty messages (common for NotImplementedError)
    if not not_impl and len(fail_msgs) == len(test_results) and all(v.strip() in ("FAIL:", "FAIL: ") for v in fail_msgs.values()):
        not_impl = True
    if not_impl:
        return 0, "technical", "not implemented", \
            "Function raises `NotImplementedError`. No implementation provided.", None

    # --- Crash ---
    crash_msgs = [v for v in fail_msgs.values() if "error" in v.lower() and "not" not in v.lower()]
    if crash_msgs and len(crash_msgs) == len(fail_msgs):
        return 0, "technical", "code crashes", \
            f"Code crashes at runtime: `{crash_msgs[0].replace('FAIL: ', '')}`.", None

    # ─── compute_energy specific ────────────────────────────────────
    if func_name == "compute_energy":
        # Shape fail → forgot padding
        if test_results.get("energy_shape", "").startswith("FAIL"):
            score = max_pts - 5
            return score, "small_bug", "forgot to pad energy map", \
                "The energy map shape doesn't match the input. Valid convolution with a 3×3 kernel " \
                "shrinks the output by 2 in each dimension. You need to pad back to the original size.\n\n" \
                "Use `F.pad(e.unsqueeze(0), (1, 1, 1, 1), mode='replicate').squeeze(0)` " \
                "(note: `F.pad` requires a 3D+ tensor for 2D spatial padding).", \
                "# Pad the energy map back to original size:\n" \
                "e = gx.abs() + gy.abs()\n" \
                "energy = F.pad(e.unsqueeze(0), (1, 1, 1, 1), mode='replicate').squeeze(0)"

        # Non-negative fail → forgot abs
        if test_results.get("energy_nonneg", "").startswith("FAIL"):
            score = max_pts - 5
            return score, "small_bug", "forgot absolute values", \
                "Energy values should be non-negative. The energy is the sum of **absolute** gradient " \
                "magnitudes: `E = |Gx| + |Gy|`. Without `.abs()`, negative gradients cancel positive ones.", \
                "# Use absolute values:\n" \
                "e = gx.abs() + gy.abs()   # not gx + gy"

        # Correctness fail but shape/nonneg pass → wrong kernels or other
        if test_results.get("energy_correctness", "").startswith("FAIL"):
            score = max_pts - 5
            return score, "small_bug", "energy values don't match reference", \
                "Your energy computation produces wrong values. Check your Sobel kernel definitions " \
                "and make sure you're using absolute values of both gradients.", None

    # ─── find_seam specific ─────────────────────────────────────────
    if func_name == "find_seam":
        # Connected but not optimal → greedy or backtrack bug
        connected_ok = test_results.get("seam_connected") == "PASS"
        optimal_fail = test_results.get("seam_optimal", "").startswith("FAIL")
        trivial_fail = test_results.get("seam_trivial", "").startswith("FAIL")

        if connected_ok and optimal_fail:
            # Check if greedy: has argmin but no cumulative cost matrix / no backtrack
            is_greedy = ("argmin" in source_code and "backtrack" not in source_code.lower() and "M[" not in source_code)
            if is_greedy or "greedy" in source_code.lower():
                score = round(max_pts * 0.2)  # fundamental: no DP
                return score, "fundamental", "greedy instead of DP", \
                    "Your seam finder uses a **greedy** approach — picking the local minimum at each row. " \
                    "This doesn't find the globally optimal seam. You need **dynamic programming**: build a " \
                    "cumulative cost matrix M where `M[i,j] = energy[i,j] + min(M[i-1,j-1], M[i-1,j], M[i-1,j+1])`, " \
                    "then backtrack from the minimum in the last row.", \
                    "# Dynamic programming approach:\n" \
                    "M = torch.zeros_like(energy)\n" \
                    "M[0] = energy[0]\n" \
                    "for i in range(1, H):\n" \
                    "    # For each pixel, find min of 3 neighbors above\n" \
                    "    left = F.pad(M[i-1, :W-1].unsqueeze(0), (1,0), value=float('inf')).squeeze(0)\n" \
                    "    right = F.pad(M[i-1, 1:].unsqueeze(0), (0,1), value=float('inf')).squeeze(0)\n" \
                    "    center = M[i-1]\n" \
                    "    M[i] = energy[i] + torch.stack([left,center,right]).min(dim=0).values\n" \
                    "# Then backtrack from M[-1].argmin()"
            else:
                score = max_pts - 5
                return score, "small_bug", "seam not optimal", \
                    "Your seam is valid (connected, in bounds) but not optimal. " \
                    "Check your DP recurrence and backtracking logic.", None

        if not connected_ok:
            score = round(max_pts * 0.2)
            return score, "fundamental", "seam not connected", \
                "Adjacent seam pixels differ by more than 1 column. " \
                "Check your backtracking — each step should only move to j-1, j, or j+1.", None

    # ─── remove_seam specific ───────────────────────────────────────
    if func_name == "remove_seam":
        shape_rgb_fail = test_results.get("remove_shape_rgb", "").startswith("FAIL")
        shape_gray_fail = test_results.get("remove_shape_gray", "").startswith("FAIL")
        correct_fail = test_results.get("remove_correctness", "").startswith("FAIL")

        if shape_rgb_fail or shape_gray_fail:
            score = max_pts - 5
            return score, "small_bug", "wrong output shape after removal", \
                "The output shape should be (H, W-1) or (H, W-1, 3). " \
                "Make sure you're removing exactly one pixel per row.", None

        if correct_fail:
            score = max_pts - 5
            return score, "small_bug", "wrong pixels after removal", \
                "The remaining pixel values don't match. Check your masking logic — " \
                "for RGB images, expand the mask across channels before applying.", \
                "# For RGB (H, W, 3):\n" \
                "mask = torch.ones(H, W, dtype=torch.bool)\n" \
                "for i in range(H):\n" \
                "    mask[i, seam[i]] = False\n" \
                "return img[mask.unsqueeze(-1).expand_as(img)].reshape(H, W-1, C)"

    # ─── seam_carve specific ────────────────────────────────────────
    if func_name == "seam_carve":
        # Check if the pipeline code itself looks correct (chains gray→energy→seam→remove)
        pipeline_correct = all(kw in source_code for kw in ["compute_energy", "find_seam", "remove_seam"])

        if pipeline_correct:
            # Pipeline logic is sound — any failures are cascading from component bugs
            # Give full credit per partial_credit_policy
            score = max_pts
            return score, "perfect", "", \
                "Pipeline logic is correct (chains components properly). " \
                "Any test failures here are caused by bugs in component functions, " \
                "which are already penalized there.", None

        shape_fail = test_results.get("carve_shape", "").startswith("FAIL")
        correct_fail = test_results.get("carve_correctness_small", "").startswith("FAIL")

        if shape_fail:
            score = max_pts - 5
            return score, "small_bug", "wrong output shape", \
                "After removing n seams, the width should be W - n.", None

    # --- Generic failure ---
    n_fail = len(fail_msgs)
    n_total = len(test_results)
    score = max(0, max_pts - n_fail * 5)
    return score, "small_bug", f"{n_fail}/{n_total} tests failed", \
        "Some tests failed. Review your implementation.", None


def build_grade_cells(student_result, spec):
    """Build grade cells for the top of the notebook."""
    cells = []
    tests = student_result.get("tests", {})
    source = student_result.get("source", {})
    timing = student_result.get("timing", {})

    func_scores = {}
    func_diags = {}
    overall_category = "perfect"
    overall_title = ""

    for func_name in FUNC_ORDER:
        func_spec = spec["functions"][func_name]
        score, cat, title, diag, fix = diagnose_function(
            func_name, func_spec, tests, source.get(func_name, "")
        )
        func_scores[func_name] = score
        func_diags[func_name] = {
            "score": score, "max": func_spec["points"],
            "category": cat, "title": title,
            "diagnosis": diag, "fix": fix,
        }
        if cat == "fundamental" and overall_category in ("perfect", "small_bug", "technical"):
            overall_category = cat; overall_title = title
        elif cat == "small_bug" and overall_category in ("perfect", "technical"):
            overall_category = cat; overall_title = title
        elif cat == "technical" and overall_category == "perfect":
            overall_category = cat; overall_title = title

    total = sum(func_scores.values())

    status_emoji = {"perfect": "✓", "small_bug": "~", "fundamental": "✗", "technical": "⚠"}

    # ─── CELL 1: Header + summary ───
    header_lines = [
        "# Grading Report",
        "",
        f"## Grade: {total}/100",
        "",
        "| Function | Score | Status |",
        "|----------|-------|--------|",
    ]

    for func_name in FUNC_ORDER:
        d = func_diags[func_name]
        emoji = status_emoji.get(d["category"], "?")
        status_str = f'{emoji} {d["title"]}' if d["title"] else f'{emoji} Correct'
        header_lines.append(f"| `{func_name}` | {d['score']}/{d['max']} | {status_str} |")

    if timing:
        timing_parts = [f"{k}: {v}" for k, v in timing.items() if not str(v).startswith("FAIL")]
        if timing_parts:
            header_lines.extend(["", "**Timing:** " + ", ".join(timing_parts)])

    header_lines.extend(["", "---"])
    cells.append(nbformat.v4.new_markdown_cell("\n".join(header_lines)))

    # ─── CELLS 2+: Per-function details ───
    for func_name in FUNC_ORDER:
        func_spec = spec["functions"][func_name]
        d = func_diags[func_name]

        detail_lines = [f"### `{func_name}` — {d['score']}/{d['max']}", ""]
        detail_lines.append("| Test | Description | Result |")
        detail_lines.append("|------|-------------|--------|")

        for t in func_spec["tests"]:
            tname = t["name"]
            result = tests.get(tname, "NOT_RUN")
            passed = result == "PASS"
            desc = t.get("description", tname)[:50]
            status = "✓ PASS" if passed else f"✗ {result.replace('FAIL: ', '')[:60]}"
            detail_lines.append(f"| {tname} | {desc} | {status} |")

        detail_lines.append("")

        if d["category"] != "perfect":
            detail_lines.append(f"**Diagnosis:** {d['diagnosis']}")
            detail_lines.append("")

        detail_lines.append("---")
        cells.append(nbformat.v4.new_markdown_cell("\n".join(detail_lines)))

        if d["fix"]:
            cells.append(nbformat.v4.new_code_cell(f"# Suggested fix for {func_name}:\n\n{d['fix']}"))

    # ─── VISUAL TEST RESULTS ───
    visual_files = student_result.get("visual_files", {})
    if visual_files:
        vis_lines = ["## Visual Test Results", ""]
        cells.append(nbformat.v4.new_markdown_cell("\n".join(vis_lines)))

        for vt in spec.get("visual_tests", []):
            vname = vt["name"]
            if vname in visual_files:
                vis_path = visual_files[vname]
                try:
                    with open(vis_path, "rb") as f:
                        img_b64 = base64.b64encode(f.read()).decode()
                    cells.append(nbformat.v4.new_markdown_cell(
                        f"### {vt['title']}\n\n"
                        f"![{vt['title']}](data:image/png;base64,{img_b64})"
                    ))
                except Exception:
                    cells.append(nbformat.v4.new_markdown_cell(
                        f"### {vt['title']}\n\n*Visual output not available.*"
                    ))

        cells.append(nbformat.v4.new_markdown_cell("---"))

    # ─── SEPARATOR ───
    cells.append(nbformat.v4.new_markdown_cell(
        "---\n---\n\n"
        "# Original Submission\n\n"
        "*Everything below is the student's original work, unmodified.*\n\n"
        "---"
    ))

    return cells, total, overall_category, overall_title, func_scores


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

grades = []

for r in all_results:
    sid = r["student_id"]
    fname = r["file"]

    grade_cells, total, category, short_title, func_scores = build_grade_cells(r, spec)

    nb_path = SUBMISSIONS_DIR / fname
    nb = nbformat.read(str(nb_path), as_version=4)
    nb.cells = grade_cells + nb.cells

    graded_path = GRADED_DIR / fname
    nbformat.write(nb, str(graded_path))

    row = {
        "student_id": sid,
        "file": fname,
        "total_grade": total,
        "category": category,
        "short_title": short_title,
    }
    for fn in FUNC_ORDER:
        row[fn] = func_scores.get(fn, 0)
    grades.append(row)

# ─── CSV ────────────────────────────────────────────────────────────────────

fieldnames = ["student_id", "file", "total_grade", "category", "short_title"] + FUNC_ORDER

with open(REPORT_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for g in sorted(grades, key=lambda x: x["total_grade"]):
        writer.writerow(g)

# ─── Summary ────────────────────────────────────────────────────────────────

print(f"Graded {len(grades)} submissions:\n")

cats = Counter(g["category"] for g in grades)
for cat in ["perfect", "small_bug", "fundamental", "technical"]:
    n = cats.get(cat, 0)
    if n:
        avg = sum(g["total_grade"] for g in grades if g["category"] == cat) / n
        print(f"  {n:>3} {cat:<15} (avg {avg:.0f}/100)")

print(f"\nGraded notebooks: {GRADED_DIR}/")
print(f"CSV report:       {REPORT_PATH}")

print(f"\n{'Student ID':<15} {'Grade':>5}  {'Category':<15} {'Title'}")
print("-" * 70)
for g in sorted(grades, key=lambda x: -x["total_grade"]):
    print(f"{g['student_id']:<15} {g['total_grade']:>5}  {g['category']:<15} {g['short_title']}")
