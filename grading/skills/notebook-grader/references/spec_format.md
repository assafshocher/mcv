# Assignment Spec Format

Each assignment has a JSON spec file that tells the grader what to test. Here's the full schema with annotations.

## Schema

```json
{
  "assignment_name": "HW1 - From Scratch",
  "description": "2D discrete convolution implementation",
  "total_points": 100,

  "student_id_variable": "STUDENT_ID",

  "setup_code": "import torch\nimport time",

  "reference_code": "Python code defining reference implementations (used internally by the test harness, never shown to students)",

  "functions": {
    "function_name": {
      "points": 30,
      "description": "What this function should do",

      "tests": [
        {
          "name": "descriptive_test_name",
          "setup": "Code to run before the test (e.g., set seed, create inputs)",
          "call": "How to call the student's function (Python expression)",
          "check": "Assertion code — has access to `result` (student output) and `expected` (reference output)",
          "expected_setup": "Code to compute expected output using reference implementation",
          "points": 10,
          "description": "Human-readable description of what this test checks"
        }
      ],

      "banned_ops": ["list", "of", "banned", "strings"],
      "banned_ops_penalty": 100
    }
  },

  "timing_tests": [
    {
      "name": "vectorized_256",
      "setup": "Code to create inputs",
      "call": "Function call to time",
      "warmup": true,
      "description": "Timing 256x256 vectorized convolution"
    }
  ],

  "grading_notes": {
    "common_mistakes": [
      {
        "symptom": "What the test failure looks like",
        "diagnosis": "What the student likely did wrong",
        "category": "fundamental | small_bug | technical",
        "typical_deduction": "How many points to take off"
      }
    ],
    "partial_credit_policy": "Description of how to handle partial correctness"
  }
}
```

## Field Details

### `setup_code`
Runs at the start of the test cell, before any tests. Use for imports that students are expected to have.

### `reference_code`
Python code defining correct implementations of all functions. This gets injected into the test environment with `_ref_` prefixed names so they don't collide with student code. The test harness uses these to compute expected outputs.

### `functions`
Each key is a function name that students must define. The grader checks that the function exists and runs its tests.

#### `tests[]`
Each test has:
- **`setup`** — Code to prepare inputs (seeds, tensor creation)
- **`call`** — Expression that calls the student function, result stored in `result`
- **`expected_setup`** — Expression that computes the expected output, stored in `expected`
- **`check`** — Assertion code. Has access to `result`, `expected`, and any variables from `setup`. Should raise AssertionError with a descriptive message on failure.
- **`points`** — Points for this specific test (must sum to the function's total points)

#### `banned_ops`
Strings to search for in the function's source code. If found, the entire function gets zero points (or `banned_ops_penalty` deduction).

### `timing_tests`
Optional. These measure performance but don't affect the grade unless explicitly configured. Results are recorded in the report for informational purposes.

### `grading_notes`
Guidance for the LLM analysis phase. The `common_mistakes` list helps pattern-match failures to known issues, which makes the diagnosis more accurate and consistent across students.

## Example: Minimal Spec

```json
{
  "assignment_name": "HW1 - From Scratch",
  "total_points": 100,
  "student_id_variable": "STUDENT_ID",
  "setup_code": "import torch",
  "reference_code": "def _ref_add(a, b): return a + b",
  "functions": {
    "my_add": {
      "points": 100,
      "tests": [
        {
          "name": "basic_add",
          "setup": "a, b = torch.tensor(1.0), torch.tensor(2.0)",
          "call": "my_add(a, b)",
          "expected_setup": "_ref_add(a, b)",
          "check": "assert torch.allclose(result, expected)",
          "points": 100
        }
      ]
    }
  }
}
```
