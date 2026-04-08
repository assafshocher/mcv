#!/usr/bin/env python3
"""
Unified mock submission generator for all HW assignments.
Generates 2 mocks per notebook (perfect + one error case),
then executes them to produce outputs (simulating real student submissions).

Usage:
    python generate_all_mocks.py [--hw hw3] [--type applied] [--no-execute]
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
HW_DIR = REPO_ROOT / "hw"
GRADING_DIR = REPO_ROOT / "grading"
PYTHON = "/Users/assafshocher/anaconda3/envs/torch/bin/python"


def read_notebook(path):
    """Read a notebook file."""
    import nbformat
    return nbformat.read(str(path), as_version=4)


def write_notebook(nb, path):
    """Write a notebook file."""
    import nbformat
    nbformat.write(nb, str(path))


def find_code_cell(nb, pattern):
    """Find a code cell containing a pattern. Returns cell index."""
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'code' and pattern in cell.source:
            return i
    return None


def replace_cell_code(nb, pattern, new_code):
    """Replace the source of a code cell matching a pattern."""
    idx = find_code_cell(nb, pattern)
    if idx is not None:
        nb.cells[idx].source = new_code
        return True
    return False


def set_student_id(nb, student_id):
    """Set the STUDENT_ID in the notebook."""
    for cell in nb.cells:
        if cell.cell_type == 'code' and 'STUDENT_ID' in cell.source:
            lines = cell.source.split('\n')
            for j, line in enumerate(lines):
                if line.strip().startswith('STUDENT_ID'):
                    lines[j] = f'STUDENT_ID = "{student_id}"'
            cell.source = '\n'.join(lines)
            return True
    return False


def execute_notebook(nb_path, timeout=600, data_dir=None):
    """Execute a notebook in place and save with outputs."""
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor

    nb = nbformat.read(str(nb_path), as_version=4)

    ep = ExecutePreprocessor(
        timeout=timeout,
        kernel_name="python3",
        allow_errors=True,
        interrupt_on_timeout=True
    )

    # Set up resources with data dir
    resources = {"metadata": {"path": str(nb_path.parent)}}

    # Symlink data directory if needed
    work_dir = nb_path.parent
    data_dest = work_dir / "data"
    if data_dir and not data_dest.exists():
        data_src = Path(data_dir)
        if data_src.exists():
            os.symlink(str(data_src.resolve()), str(data_dest))

    try:
        ep.preprocess(nb, resources)
        print(f"  Executed successfully")
    except Exception as e:
        print(f"  Execution warning: {type(e).__name__}: {str(e)[:100]}")

    # Count outputs
    code_cells = [c for c in nb.cells if c.cell_type == 'code']
    cells_with_output = sum(1 for c in code_cells if c.get('outputs', []))
    print(f"  {cells_with_output}/{len(code_cells)} code cells have outputs")

    nbformat.write(nb, str(nb_path))


# ═══════════════════════════════════════════════════════════════════════════════
# HW1 Applied: Seam Carving
# ═══════════════════════════════════════════════════════════════════════════════

HW1_APPLIED_PERFECT = {
    "compute_energy": '''def compute_energy(image):
    """Compute energy map using Sobel filters."""
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image).float()
    if image.ndim == 3:
        gray = 0.299 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]
    else:
        gray = image.float()
    gray = gray.unsqueeze(0).unsqueeze(0)
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    gx = F.conv2d(gray, sobel_x, padding=1)
    gy = F.conv2d(gray, sobel_y, padding=1)
    energy = (gx.abs() + gy.abs()).squeeze()
    return energy''',

    "find_seam": '''def find_seam(energy):
    """Find minimum-energy vertical seam using dynamic programming."""
    H, W = energy.shape
    dp = energy.clone()
    backtrack = torch.zeros(H, W, dtype=torch.long)
    for i in range(1, H):
        for j in range(W):
            candidates = [dp[i-1, j]]
            offsets = [0]
            if j > 0:
                candidates.append(dp[i-1, j-1])
                offsets.append(-1)
            if j < W - 1:
                candidates.append(dp[i-1, j+1])
                offsets.append(1)
            min_idx = int(torch.tensor(candidates).argmin())
            dp[i, j] = energy[i, j] + candidates[min_idx]
            backtrack[i, j] = j + offsets[min_idx]
    # Backtrace
    seam = torch.zeros(H, dtype=torch.long)
    seam[-1] = dp[-1].argmin()
    for i in range(H-2, -1, -1):
        seam[i] = backtrack[i+1, seam[i+1]]
    return seam''',

    "remove_seam": '''def remove_seam(image, seam):
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

    "seam_carve": '''def seam_carve(image, num_seams):
    """Remove num_seams vertical seams from an image."""
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image).float()
    result = image.clone()
    for _ in range(num_seams):
        energy = compute_energy(result)
        seam = find_seam(energy)
        result = remove_seam(result, seam)
    return result''',
}

HW1_APPLIED_GREEDY = {
    **HW1_APPLIED_PERFECT,
    "find_seam": '''def find_seam(energy):
    """Find seam using GREEDY approach (BUG: not DP)."""
    H, W = energy.shape
    seam = torch.zeros(H, dtype=torch.long)
    seam[0] = energy[0].argmin()
    for i in range(1, H):
        j = seam[i-1].item()
        lo = max(0, j-1)
        hi = min(W, j+2)
        seam[i] = lo + energy[i, lo:hi].argmin()
    return seam''',
}

# ═══════════════════════════════════════════════════════════════════════════════
# HW1 Scratch: 2D Convolution
# ═══════════════════════════════════════════════════════════════════════════════

HW1_SCRATCH_PERFECT = {
    "conv2d_loops": '''def conv2d_loops(image, kernel):
    """2D convolution using for loops with kernel flipping."""
    kernel = torch.flip(kernel, [0, 1])
    kH, kW = kernel.shape
    pH, pW = kH // 2, kW // 2
    H, W = image.shape
    padded = torch.zeros(H + 2*pH, W + 2*pW)
    padded[pH:pH+H, pW:pW+W] = image
    output = torch.zeros(H, W)
    for i in range(H):
        for j in range(W):
            output[i, j] = (padded[i:i+kH, j:j+kW] * kernel).sum()
    return output''',

    "im2patches": '''def im2patches(image, kH, kW):
    """Extract patches from image."""
    H, W = image.shape
    H_out, W_out = H - kH + 1, W - kW + 1
    patches = torch.zeros(H_out * W_out, kH * kW)
    for i in range(H_out):
        for j in range(W_out):
            patches[i * W_out + j] = image[i:i+kH, j:j+kW].reshape(-1)
    return patches''',

    "conv2d_vectorized": '''def conv2d_vectorized(image, kernel):
    """Vectorized 2D convolution using im2patches."""
    kernel_flipped = torch.flip(kernel, [0, 1])
    kH, kW = kernel.shape
    pH, pW = kH // 2, kW // 2
    H, W = image.shape
    padded = torch.zeros(H + 2*pH, W + 2*pW)
    padded[pH:pH+H, pW:pW+W] = image
    patches = im2patches(padded, kH, kW)
    result = patches @ kernel_flipped.reshape(-1)
    return result.reshape(H, W)''',
}

HW1_SCRATCH_XCORR = {
    **HW1_SCRATCH_PERFECT,
    "conv2d_loops": '''def conv2d_loops(image, kernel):
    """2D convolution — BUG: no kernel flip (cross-correlation)."""
    kH, kW = kernel.shape
    pH, pW = kH // 2, kW // 2
    H, W = image.shape
    padded = torch.zeros(H + 2*pH, W + 2*pW)
    padded[pH:pH+H, pW:pW+W] = image
    output = torch.zeros(H, W)
    for i in range(H):
        for j in range(W):
            output[i, j] = (padded[i:i+kH, j:j+kW] * kernel).sum()
    return output''',
}

# ═══════════════════════════════════════════════════════════════════════════════
# HW2 Applied: Digit Addition
# ═══════════════════════════════════════════════════════════════════════════════

HW2_APPLIED_PERFECT_CODE = '''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
STUDENT_ID = "perfect_student"

# Model
class DigitClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

model = DigitClassifier().to(device)

# Training
tf = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
train_ds = torchvision.datasets.MNIST('./data', train=True, download=True, transform=tf)
test_ds = torchvision.datasets.MNIST('./data', train=False, download=True, transform=tf)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=1000)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train_losses = []
test_accs = []

for epoch in range(5):
    model.train()
    total_loss = 0; n = 0
    for imgs, labs in train_loader:
        imgs, labs = imgs.to(device), labs.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = F.cross_entropy(out, labs)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        n += imgs.size(0)
    train_losses.append(total_loss / n)

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labs in test_loader:
            imgs, labs = imgs.to(device), labs.to(device)
            correct += (model(imgs).argmax(1) == labs).sum().item()
            total += labs.size(0)
    test_accs.append(correct / total)
    print(f"Epoch {epoch+1}: loss={train_losses[-1]:.4f}, acc={test_accs[-1]:.2%}")

def predict_sum(model, img1, img2):
    model.eval()
    with torch.no_grad():
        logits1 = model(img1)
        logits2 = model(img2)
        d1 = logits1.argmax(1)
        d2 = logits2.argmax(1)
    return d1 + d2
'''

HW2_APPLIED_UNDERTRAINED_CODE = '''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
STUDENT_ID = "undertrained_student"

# Model - too simple
class DigitClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(784, 10)
    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))

model = DigitClassifier().to(device)

tf = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
train_ds = torchvision.datasets.MNIST('./data', train=True, download=True, transform=tf)
test_ds = torchvision.datasets.MNIST('./data', train=False, download=True, transform=tf)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=1000)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
train_losses = []
test_accs = []

for epoch in range(2):
    model.train()
    total_loss = 0; n = 0
    for imgs, labs in train_loader:
        imgs, labs = imgs.to(device), labs.to(device)
        optimizer.zero_grad()
        loss = F.cross_entropy(model(imgs), labs)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0); n += imgs.size(0)
    train_losses.append(total_loss / n)

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labs in test_loader:
            imgs, labs = imgs.to(device), labs.to(device)
            correct += (model(imgs).argmax(1) == labs).sum().item()
            total += labs.size(0)
    test_accs.append(correct / total)
    print(f"Epoch {epoch+1}: loss={train_losses[-1]:.4f}, acc={test_accs[-1]:.2%}")

def predict_sum(model, img1, img2):
    model.eval()
    with torch.no_grad():
        d1 = model(img1).argmax(1)
        d2 = model(img2).argmax(1)
    return d1 + d2
'''


# ═══════════════════════════════════════════════════════════════════════════════
# HW2 Scratch: Build Your Own Neural Network
# ═══════════════════════════════════════════════════════════════════════════════

# These are complex — read from template and fill in implementations
# We'll handle them by reading the existing template and replacing stubs


# ═══════════════════════════════════════════════════════════════════════════════
# HW3 Applied: ZSSR — use existing generator code
# ═══════════════════════════════════════════════════════════════════════════════

# Imported from existing generator pattern


# ═══════════════════════════════════════════════════════════════════════════════
# HW3 Scratch: CNN from scratch — use existing generator code
# ═══════════════════════════════════════════════════════════════════════════════


def generate_hw1_applied(output_dir):
    """Generate HW1 applied mocks."""
    import nbformat
    template = read_notebook(HW_DIR / "hw1" / "applied.ipynb")
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, impls in [("student_perfect", HW1_APPLIED_PERFECT),
                         ("student_greedy", HW1_APPLIED_GREEDY)]:
        nb = deepcopy(template)
        set_student_id(nb, name)
        for func_name, code in impls.items():
            if not replace_cell_code(nb, f"def {func_name}", code):
                print(f"  WARNING: Could not find cell for {func_name}")
        write_notebook(nb, output_dir / f"{name}.ipynb")
        print(f"  Generated {name}.ipynb")


def generate_hw1_scratch(output_dir):
    """Generate HW1 scratch mocks."""
    import nbformat
    template = read_notebook(HW_DIR / "hw1" / "from_scratch.ipynb")
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, impls in [("student_perfect", HW1_SCRATCH_PERFECT),
                         ("student_xcorr", HW1_SCRATCH_XCORR)]:
        nb = deepcopy(template)
        set_student_id(nb, name)
        for func_name, code in impls.items():
            if not replace_cell_code(nb, f"def {func_name}", code):
                print(f"  WARNING: Could not find cell for {func_name}")
        write_notebook(nb, output_dir / f"{name}.ipynb")
        print(f"  Generated {name}.ipynb")


def generate_hw2_applied(output_dir):
    """Generate HW2 applied mocks as single-cell notebooks."""
    import nbformat
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, code in [("student_perfect", HW2_APPLIED_PERFECT_CODE),
                        ("student_undertrained", HW2_APPLIED_UNDERTRAINED_CODE)]:
        nb = nbformat.v4.new_notebook()
        nb.cells = [nbformat.v4.new_code_cell(source=code)]
        write_notebook(nb, output_dir / f"{name}.ipynb")
        print(f"  Generated {name}.ipynb")


def generate_hw2_scratch(output_dir):
    """Generate HW2 scratch mocks from existing generator."""
    gen_path = GRADING_DIR / "hw2" / "generators" / "gen_hw2_scratch_submissions.py"
    if gen_path.exists():
        # Run existing generator
        result = subprocess.run(
            [PYTHON, str(gen_path)],
            capture_output=True, text=True, cwd=str(GRADING_DIR / "hw2")
        )
        if result.returncode == 0:
            print(f"  Used existing generator")
            # Keep only perfect and one error variant
            mock_dir = GRADING_DIR / "hw2" / "mock_submissions_scratch"
            if mock_dir.exists():
                for f in mock_dir.glob("*.ipynb"):
                    if f.stem not in ("student_perfect", "student_no_bias"):
                        pass  # Keep all for now
        else:
            print(f"  Generator failed: {result.stderr[:200]}")
    else:
        print(f"  No generator found, using existing mocks")

    # Copy to output dir if different
    src_dir = GRADING_DIR / "hw2" / "mock_submissions_scratch"
    if src_dir.exists() and src_dir != output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        for f in sorted(src_dir.glob("*.ipynb"))[:2]:
            shutil.copy2(str(f), str(output_dir / f.name))
            print(f"  Copied {f.name}")


def generate_hw3_applied(output_dir):
    """Generate HW3 applied mocks from existing generator."""
    gen_path = GRADING_DIR / "hw3" / "generators" / "gen_hw3_applied_submissions.py"
    if gen_path.exists():
        # The existing generator uses hardcoded paths, so we run it with modifications
        pass

    # Use existing mocks
    src_dir = GRADING_DIR / "hw3" / "mock_submissions_applied"
    if src_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        # Keep student_perfect and student_no_residual
        for name in ["student_perfect", "student_no_residual"]:
            src = src_dir / f"{name}.ipynb"
            if src.exists():
                shutil.copy2(str(src), str(output_dir / f"{name}.ipynb"))
                print(f"  Copied {name}.ipynb")


def generate_hw3_scratch(output_dir):
    """Generate HW3 scratch mocks."""
    src_dir = GRADING_DIR / "hw3" / "mock_submissions_scratch"
    if src_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        for name in ["student_perfect", "student_xcorr"]:
            src = src_dir / f"{name}.ipynb"
            if src.exists():
                shutil.copy2(str(src), str(output_dir / f"{name}.ipynb"))
                print(f"  Copied {name}.ipynb")


def execute_all_in_dir(mock_dir, timeout=600, data_dir=None):
    """Execute all notebooks in a directory to produce outputs."""
    import nbformat
    for nb_path in sorted(mock_dir.glob("*.ipynb")):
        print(f"  Executing {nb_path.name}...")
        execute_notebook(nb_path, timeout=timeout, data_dir=data_dir)


GENERATORS = {
    ("hw1", "applied"): generate_hw1_applied,
    ("hw1", "scratch"): generate_hw1_scratch,
    ("hw2", "applied"): generate_hw2_applied,
    ("hw2", "scratch"): generate_hw2_scratch,
    ("hw3", "applied"): generate_hw3_applied,
    ("hw3", "scratch"): generate_hw3_scratch,
}


def main():
    parser = argparse.ArgumentParser(description="Generate mock submissions for grading tests")
    parser.add_argument("--hw", type=str, help="Specific HW to generate (e.g., hw1, hw2, hw3)")
    parser.add_argument("--type", type=str, help="Specific type (applied, scratch)")
    parser.add_argument("--no-execute", action="store_true", help="Skip execution step")
    parser.add_argument("--data-dir", type=str, default=str(GRADING_DIR / "data"),
                        help="Path to data directory")
    args = parser.parse_args()

    data_dir = args.data_dir

    targets = []
    if args.hw and args.type:
        targets = [(args.hw, args.type)]
    elif args.hw:
        targets = [(args.hw, t) for t in ["applied", "scratch"] if (args.hw, t) in GENERATORS]
    else:
        targets = list(GENERATORS.keys())

    for hw, typ in targets:
        print(f"\n{'='*60}")
        print(f"Generating {hw} {typ} mocks")
        print(f"{'='*60}")

        output_dir = GRADING_DIR / hw / f"mock_submissions_{typ}"
        gen_func = GENERATORS[(hw, typ)]
        gen_func(output_dir)

        if not args.no_execute:
            print(f"\nExecuting {hw} {typ} notebooks...")
            timeout = 900 if hw == "hw3" and typ == "applied" else 600
            execute_all_in_dir(output_dir, timeout=timeout, data_dir=data_dir)

    print(f"\n{'='*60}")
    print("Done! All mocks generated" + (" and executed" if not args.no_execute else ""))


if __name__ == "__main__":
    main()
