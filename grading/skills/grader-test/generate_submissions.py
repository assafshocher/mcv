#!/usr/bin/env python3
"""
Generate synthetic student notebooks for testing the grading pipeline.

Usage:
    python generate_submissions.py <spec_file> [--output <folder>]
"""

import argparse
import json
from pathlib import Path
import nbformat


def make_notebook(student_id: str, cells_code: list[str]) -> nbformat.NotebookNode:
    """Create a minimal notebook with student ID and code cells."""
    nb = nbformat.v4.new_notebook()
    nb.cells.append(nbformat.v4.new_code_cell(f'STUDENT_ID = "{student_id}"'))
    nb.cells.append(nbformat.v4.new_code_cell("import torch\nimport time"))
    for code in cells_code:
        nb.cells.append(nbformat.v4.new_code_cell(code))
    return nb


# ─── Synthetic implementations ───────────────────────────────────────────────

PERFECT_CONV2D_LOOPS = """
def conv2d_loops(image, kernel):
    H, W = image.shape
    kH, kW = kernel.shape
    H_out, W_out = H - kH + 1, W - kW + 1
    kernel_flipped = kernel.flip(0).flip(1)
    out = torch.zeros(H_out, W_out)
    for m in range(H_out):
        for n in range(W_out):
            out[m, n] = (image[m:m+kH, n:n+kW] * kernel_flipped).sum()
    return out
""".strip()

PERFECT_IM2PATCHES = """
def im2patches(image, kH, kW):
    H, W = image.shape
    H_out, W_out = H - kH + 1, W - kW + 1
    cols = []
    for di in range(kH):
        for dj in range(kW):
            cols.append(image[di:di+H_out, dj:dj+W_out].reshape(-1))
    return torch.stack(cols, dim=1)
""".strip()

PERFECT_CONV2D_VEC = """
def conv2d_vectorized(image, kernel):
    kH, kW = kernel.shape
    H_out = image.shape[0] - kH + 1
    W_out = image.shape[1] - kW + 1
    patches = im2patches(image, kH, kW)
    kernel_flipped = kernel.flip(0).flip(1).reshape(-1)
    return (patches @ kernel_flipped).reshape(H_out, W_out)
""".strip()

# --- Buggy variants ---

XCORR_LOOPS = """
def conv2d_loops(image, kernel):
    H, W = image.shape
    kH, kW = kernel.shape
    H_out, W_out = H - kH + 1, W - kW + 1
    out = torch.zeros(H_out, W_out)
    for m in range(H_out):
        for n in range(W_out):
            out[m, n] = (image[m:m+kH, n:n+kW] * kernel).sum()
    return out
""".strip()

XCORR_VEC = """
def conv2d_vectorized(image, kernel):
    kH, kW = kernel.shape
    H_out = image.shape[0] - kH + 1
    W_out = image.shape[1] - kW + 1
    patches = im2patches(image, kH, kW)
    kernel_flat = kernel.reshape(-1)
    return (patches @ kernel_flat).reshape(H_out, W_out)
""".strip()

OFF_BY_ONE_LOOPS = """
def conv2d_loops(image, kernel):
    H, W = image.shape
    kH, kW = kernel.shape
    H_out, W_out = H - kH, W - kW  # off by one!
    kernel_flipped = kernel.flip(0).flip(1)
    out = torch.zeros(H_out, W_out)
    for m in range(H_out):
        for n in range(W_out):
            out[m, n] = (image[m:m+kH, n:n+kW] * kernel_flipped).sum()
    return out
""".strip()

USES_UNFOLD = """
def im2patches(image, kH, kW):
    # Using banned unfold operation
    H, W = image.shape
    patches = image.unfold(0, kH, 1).unfold(1, kW, 1)
    return patches.reshape(-1, kH * kW)
""".strip()

CRASH_CODE = """
def conv2d_loops(image, kernel):
    return undefined_variable + image  # NameError
""".strip()


def generate(output_dir: Path):
    """Generate all synthetic submissions."""
    output_dir.mkdir(parents=True, exist_ok=True)

    notebooks = {
        # Perfect student
        "perfect_student": {
            "id": "PERFECT_001",
            "cells": [PERFECT_CONV2D_LOOPS, PERFECT_IM2PATCHES, PERFECT_CONV2D_VEC],
            "expected": "all_pass"
        },

        # Cross-correlation (forgot to flip kernel in both functions)
        "student_xcorr": {
            "id": "XCORR_002",
            "cells": [XCORR_LOOPS, PERFECT_IM2PATCHES, XCORR_VEC],
            "expected": "fails xcorr and correctness tests for both conv functions"
        },

        # Off-by-one in loop bounds
        "student_off_by_one": {
            "id": "OFFBYONE_003",
            "cells": [OFF_BY_ONE_LOOPS, PERFECT_IM2PATCHES, PERFECT_CONV2D_VEC],
            "expected": "fails shape and correctness for conv2d_loops, vec_matches_loops may also fail"
        },

        # Uses banned unfold in im2patches
        "student_unfold": {
            "id": "UNFOLD_004",
            "cells": [PERFECT_CONV2D_LOOPS, USES_UNFOLD, PERFECT_CONV2D_VEC],
            "expected": "fails banned_ops for im2patches, also banned_ops for conv2d_vectorized (uses im2patches)"
        },

        # Crash - undefined variable
        "student_crash": {
            "id": "CRASH_005",
            "cells": [CRASH_CODE, PERFECT_IM2PATCHES, PERFECT_CONV2D_VEC],
            "expected": "conv2d_loops tests fail, im2patches and conv2d_vectorized may still pass"
        },

        # Missing student ID
        "student_missing_id": {
            "id": "",
            "cells": [PERFECT_CONV2D_LOOPS, PERFECT_IM2PATCHES, PERFECT_CONV2D_VEC],
            "expected": "all_pass but student_id is MISSING"
        },

        # Partial implementation (only loops, rest NotImplemented)
        "student_partial": {
            "id": "PARTIAL_007",
            "cells": [
                PERFECT_CONV2D_LOOPS,
                "def im2patches(image, kH, kW):\n    raise NotImplementedError",
                "def conv2d_vectorized(image, kernel):\n    raise NotImplementedError"
            ],
            "expected": "conv2d_loops passes, im2patches and conv2d_vectorized fail"
        },
    }

    manifest = {}
    for name, info in notebooks.items():
        nb = make_notebook(info["id"], info["cells"])
        path = output_dir / f"{name}.ipynb"
        nbformat.write(nb, str(path))
        manifest[name] = {
            "student_id": info["id"] or "MISSING",
            "expected": info["expected"]
        }
        print(f"  Created: {path.name}")

    # Save manifest for validation
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  Manifest: {manifest_path}")

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic student notebooks")
    parser.add_argument("spec_file", help="Assignment spec (for reference)")
    parser.add_argument("--output", "-o", default="test_submissions",
                        help="Output folder (default: test_submissions)")
    args = parser.parse_args()

    print(f"Generating synthetic submissions...\n")
    generate(Path(args.output))
    print(f"\nDone! Run the grader on {args.output}/ to test.")


if __name__ == "__main__":
    main()
