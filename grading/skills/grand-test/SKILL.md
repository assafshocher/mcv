---
name: grand-test
description: >
  Full end-to-end grading pipeline validation across ALL homework notebooks. Use this skill
  when the user asks for a "grand test", wants to validate the entire grading system, needs
  to re-test all notebooks after changes, or wants a comprehensive grading report. Triggers
  include: "grand test", "test everything", "run all grading", "validate all notebooks",
  "full grading test", or any request to comprehensively test the grading pipeline.
---

# Grand Test

Run a complete end-to-end validation of the grading pipeline across all 6 homework notebooks.

## What It Does

For each homework notebook (HW1-HW3, scratch + applied = 6 total):

1. **Generate fresh mocks** from clean template notebooks (2 per assignment: perfect + error)
2. **Execute mocks** to produce outputs (simulating real student workflow)
3. **Grade the executed submissions** using the grading engine
4. **Validate results**: correct scores, proper graded notebooks, visual outputs, failure analysis
5. **Produce a final report** with grades, issues, and validation status

## How to Run

### Quick version (recommended after spec/engine changes):

```bash
cd /Users/assafshocher/Downloads/mcv/grading
/Users/assafshocher/anaconda3/envs/torch/bin/python grand_test.py
```

### Specific homework only:

```bash
python grand_test.py --hw hw3 --type applied
python grand_test.py --hw hw1
```

### Skip execution (faster, uses existing mock outputs):

```bash
python grand_test.py --skip-execute
```

### Skip generation (re-grade existing mocks):

```bash
python grand_test.py --skip-generate --skip-execute
```

## What Gets Validated

For each assignment:

| Check | Description |
|-------|-------------|
| Perfect score | student_perfect mock must get 100/100 |
| Crash handling | No grader crashes, errors reported gracefully |
| Graded notebook structure | Has "Grading Report" header, grade table, test details |
| Visual outputs | Grading images/GIFs appear in graded notebook (for assignments with plots) |
| Missing outputs warning | When student notebook has no outputs, warning appears |
| Failure analysis | Failed tests include hints/suggestions |
| CSV report | report.csv produced with correct format |

## Expected Output

```
Assignment           Status     Details
----------------------------------------------------------------------
hw1_scratch          PASS       error_student: 65/100 (10/14), perfect_student: 100/100 (14/14)
hw1_applied          PASS       error_student: 79/100 (17/21), perfect_student: 100/100 (21/21)
hw2_scratch          PASS       NOBIAS_202: 95/100 (14/16), PERFECT_201: 100/100 (15/16)
hw2_applied          PASS       ARGMAX_213: 100/100+10 (7/9), PERFECT_211: 100/100+25 (8/9)
hw3_scratch          PASS       MISSING: 55/100 (7/13), MISSING: 100/100 (12/13)
hw3_applied          PASS       MISSING: 55/100 (5/9), MISSING: 100/100 (8/9)
```

## Key Files

| File | Purpose |
|------|---------|
| `grading/grand_test.py` | Main grand test script |
| `grading/engine/run_tests.py` | Grading engine |
| `grading/hw*/specs/*.json` | Assignment grading specs |
| `hw/hw*/applied.ipynb`, `hw/hw*/from_scratch.ipynb` | Template notebooks |
| `grading/grand_test_report.json` | Output report (JSON) |

## When to Run Grand Test

- After modifying ANY grading spec (`*_spec.json`)
- After modifying the grading engine (`run_tests.py`)
- After modifying template notebooks
- After adding new HW assignments
- Before grading real student submissions
- When the user says "grand test" or "test everything"

## Adding New Assignments

To add a new HW to the grand test:

1. Add spec path to `SPECS` dict in `grand_test.py`
2. Add timeout to `TIMEOUTS` dict
3. Add mock implementations to `MOCK_IMPLEMENTATIONS` dict
   - Key: `(hw_name, type, "perfect")` and `(hw_name, type, "error")`
   - Value: dict mapping `"def function_name"` pattern to replacement code
4. If implementations are too complex, place existing mocks in `grading/hwN/mock_submissions_TYPE/`

## Troubleshooting

- **Perfect student doesn't get 100**: Check mock implementation matches reference in spec
- **Grading crashes**: Check `interrupt_on_timeout=True`, check for `_results` variable shadowing
- **Missing visual outputs**: Ensure matplotlib backend is `module://matplotlib_inline.backend_inline`
- **Slow execution**: HW3 applied takes ~5 min (ZSSR training). Use `--skip-execute` when testing grading-only changes
