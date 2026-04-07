---
name: notebook-grader
description: >
  Automated Jupyter notebook grading engine for course assignments. Use this skill whenever
  the user wants to grade, check, evaluate, or score student notebook submissions. Triggers
  include: grading a batch of .ipynb files, checking student code for correctness, running
  tests against submitted notebooks, producing grading reports, analyzing student errors,
  or anything involving assignment evaluation. Also use when the user mentions "grade",
  "submissions", "check notebooks", "student code", "grading report", or references
  a folder of .ipynb files to evaluate.
---

# Notebook Grader

Grade a batch of student Jupyter notebook submissions against an assignment specification.

## How It Works

This skill has two layers:

1. **This file (general engine)** — handles executing notebooks, running tests, analyzing failures, writing grades back, and producing reports. Reusable across all assignments.
2. **An assignment spec** (JSON file) — defines what to test for a specific assignment: function signatures, reference implementations, test cases, point allocations, banned operations, and grading notes. One spec per homework.

## Workflow

### Step 1: Locate the assignment spec

Ask the user which assignment they're grading. Look for a spec file — a JSON file typically named like `hw1_scratch_spec.json`, living alongside the assignment materials. Read `references/spec_format.md` for the full schema.

If no spec exists yet, help the user create one (see "Creating a New Spec" below).

### Step 2: Run the test harness

Use `scripts/run_tests.py` to execute each submitted notebook and run the test cases from the spec:

```bash
python <skill-path>/scripts/run_tests.py \
  <submissions-folder> \
  <spec-file> \
  --output <results-folder>
```

This will:
- Find all `.ipynb` files in the submissions folder
- Execute each notebook in isolation
- Inject test cells that call student functions with the spec's test inputs
- Compare outputs against expected results (using tolerances from the spec)
- Check for banned operations and measure timing
- Write a `results.json` per student in the output folder, including their source code

### Step 3: Analyze results and assign grades

Read `all_results.json`. For each student:

**Scoring philosophy — be fair, not harsh:**
- All tests pass → full points
- Small bug (correct approach, one implementation mistake) → **lose only 5 points** for that function, regardless of how many tests that one bug breaks
- Fundamental error (wrong concept entirely) → lose 80% of that function's points
- Banned operations → 0 for that function
- Not implemented / crash → 0 for that function

The key insight: when multiple tests fail from the same root cause (e.g., off-by-one causes 3 shape failures), that's ONE small bug, not three separate deductions.

**For failing students, read their actual code** and diagnose what went wrong. This is the key value — not just "wrong answer" but understanding *why*. Classify each issue:

- **Fundamental error** — misunderstands the concept (e.g., cross-correlation instead of convolution)
- **Small bug** — right approach, implementation mistake (e.g., off-by-one, wrong axis)
- **Technical issue** — crashes for unrelated reasons (e.g., typo, bad import)

Use the `grading_notes.common_mistakes` field in the spec to pattern-match failures.

### Step 4: Write graded notebooks

For each student, create a graded copy with feedback **inserted at the TOP** of the notebook, followed by a separator, then their original work unmodified.

The grade section has this structure:

**Cell 1 — Header + summary table (markdown):**
```markdown
# Grading Report

## Grade: 90/100

| Function | Score | Status |
|----------|-------|--------|
| `conv2d_loops` | 25/30 | ~ off-by-one in output size |
| `im2patches` | 30/30 | ✓ Correct |
| `conv2d_vectorized` | 35/40 | ~ correct but inconsistent with loops |

**Timing:** loops_64: 14.5ms, vec_256: 1.9ms

---
```

**Cells 2+ — Per-function detail (one markdown cell each):**

Each shows a table of test results with the actual inputs tested and pass/fail, then a diagnosis for failures:

```markdown
### `conv2d_loops` — 25/30

| Test | Input | Result |
|------|-------|--------|
| loops_delta | 10×10 image, 3×3 delta | ✗ Wrong shape: (7,7) |
| loops_correctness | 15×15 image, 5×5 kernel | ✗ Shape (10,10) vs (11,11) |
| loops_is_conv_not_xcorr | 10×10 image, 3×3 kernel | ✓ PASS |

**Diagnosis:** Output shape is wrong — you used `H - kH` instead of `H - kH + 1`...
```

**Code cells for fixes** — when there's a specific bug to fix, add a code cell with the suggested correction:

```python
# Suggested fix for conv2d_loops:

# The valid output size is:
H_out, W_out = H - kH + 1, W - kW + 1   # not H - kH, W - kW
```

**Separator cell:**
```markdown
---
---

# Original Submission

*Everything below is the student's original work, unmodified.*

---
```

**For perfect scores**, keep it simple:
```markdown
# Grading Report

## Grade: 100/100

All tests passed. Well done!

---
```

Save graded notebooks to a `graded/` subfolder — never modify originals.

### Step 5: Produce the CSV report

Generate a CSV report with columns:
- `student_id`, `file`, `total_grade`
- `category` — "perfect", "fundamental", "small_bug", "technical"
- `short_title` — 3-5 word diagnosis (empty for perfect)
- One column per graded function with that function's score

Also print a summary:
```
Graded 45 submissions:

   32 perfect         (avg 100/100)
    5 small_bug       (avg 92/100)
    4 fundamental     (avg 45/100)
    3 technical       (avg 30/100)

Student ID       Grade  Category        Title
-----------------------------------------------------------------
PERFECT_001        100  perfect
OFFBYONE_003        90  small_bug       off-by-one in output size
XCORR_002           44  fundamental     forgot to flip kernel
```

## Creating a New Spec

If the user wants to grade an assignment that doesn't have a spec yet, help them create one. You need:

1. The assignment notebook (to know what functions students implement)
2. Reference implementations of each function
3. Test cases — specific inputs (with seeds) and expected outputs
4. Point allocation per function
5. Any banned operations
6. Common mistakes to watch for (symptoms, diagnosis, category, typical deduction)

See `references/spec_format.md` for the full JSON schema and an annotated example.

## Important Notes

- Always execute notebooks in a clean environment — student code might have side effects
- The `STUDENT_ID` variable should be in one of the first few code cells
- Timing tests are informational, not pass/fail (unless the spec says otherwise)
- When analyzing code, be fair — if the approach is correct but has a minor bug, that's ONE deduction (5 points), not one per failed test. Multiple tests failing from the same root cause = one bug.
- Never modify the original submission files — always write to a separate graded/ folder
- Explanations should be educational: explain the concept, show exactly what went wrong, and provide a fix
