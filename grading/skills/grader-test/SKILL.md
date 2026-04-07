---
name: grader-test
description: >
  Generate synthetic student notebook submissions to test the notebook-grader pipeline.
  Use this skill when the user wants to validate their grading setup, create test submissions
  with known errors, stress-test the grading system, or verify that the grading spec
  catches the right things. Triggers include: "test the grader", "create fake submissions",
  "generate test notebooks", "validate grading", or any mention of testing the grading pipeline.
---

# Grader Test

Generate a folder of synthetic student notebooks to validate the notebook-grader pipeline.

## Purpose

Before grading real student submissions, you want to know the grading system actually works — that it catches the mistakes you care about, gives the right scores, and produces useful diagnostics. This skill creates synthetic notebooks that cover the important cases: a perfect submission, each common mistake from the spec, edge cases, and a crash.

## Workflow

### Step 1: Read the assignment spec

Load the assignment spec JSON. The `grading_notes.common_mistakes` section tells you exactly what failure modes to generate synthetic submissions for.

### Step 2: Generate synthetic notebooks

Create a folder (e.g., `test_submissions/`) with these notebooks:

#### Always generate:

1. **`perfect_student.ipynb`** — Correct implementation of everything. Should get 100/100. Use the reference implementations from the spec (adapted to match the student function signatures).

2. **One notebook per common mistake** — For each entry in `common_mistakes`, create a student notebook that exhibits exactly that mistake. Name it descriptively (e.g., `student_xcorr.ipynb`, `student_off_by_one.ipynb`).

3. **`crash_student.ipynb`** — A notebook that crashes mid-execution (e.g., undefined variable, bad import). Tests that the grader handles crashes gracefully.

4. **`missing_id.ipynb`** — Correct code but STUDENT_ID is empty. Tests ID extraction fallback.

#### For each synthetic notebook:

- Set `STUDENT_ID` to something descriptive (e.g., `"PERFECT_001"`, `"XCORR_002"`)
- Include the `import` cell
- Implement the required functions (correct or intentionally buggy)
- Keep it minimal — just the function definitions, no extra cells

### Step 3: Run the grader

Execute the grading pipeline on the synthetic submissions:

```bash
python <notebook-grader-path>/scripts/run_tests.py test_submissions/ <spec-file> --output test_results/
```

### Step 4: Validate results

Check that:

- **Perfect student** passes all tests
- **Each mistake notebook** fails exactly the tests you'd expect (not more, not fewer)
- **Crash notebook** is handled gracefully (no grader crash, reports the error)
- **Missing ID** falls back to `"MISSING"` or filename

For each synthetic notebook, compare actual test results to expected results. Print a validation summary:

```
Validation Results:
  perfect_student.ipynb    ✓ All tests pass as expected
  student_xcorr.ipynb      ✓ Fails xcorr tests as expected
  student_off_by_one.ipynb ✗ UNEXPECTED: patches_shape passed (should have failed)
  crash_student.ipynb      ✓ Crash handled gracefully
  missing_id.ipynb         ✓ ID correctly shows as MISSING
```

If any validation fails, the spec or the grading script needs fixing. Diagnose and report.

### Step 5: Generate the validation report

Write a summary of what was tested and what passed/failed. Include recommendations for spec improvements if any synthetic submissions revealed gaps in the grading.

## Generating Buggy Code

When creating intentionally buggy implementations, make them realistic — the kind of mistakes actual students make:

- **Cross-correlation instead of convolution**: Skip the `kernel.flip()` call
- **Off-by-one in loop bounds**: Use `range(H - kH)` instead of `range(H - kH + 1)`
- **Wrong reshape order**: Flatten patches in wrong order (column-major vs row-major)
- **Partial implementation**: One function correct, another raises NotImplementedError
- **Using banned ops**: Sneak in `torch.nn.functional.conv2d` or `unfold`

The bug should be isolated — only one thing wrong per notebook, so you can verify the grader catches exactly that issue.
