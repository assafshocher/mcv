#!/usr/bin/env python3
"""Generate synthetic student submissions for HW2 from-scratch."""

import copy
import nbformat
from pathlib import Path

TEMPLATE_PATH = Path("hw2_from_scratch.ipynb")
template_nb = nbformat.read(str(TEMPLATE_PATH), as_version=4)


def find_cell_by_content(nb, fragment):
    for i, cell in enumerate(nb.cells):
        if fragment in cell.source:
            return i
    return None


def replace_cell_source(nb, fragment, new_source):
    idx = find_cell_by_content(nb, fragment)
    if idx is not None:
        nb.cells[idx].source = new_source
        return True
    return False


def make_submission(student_id, replacements):
    nb = copy.deepcopy(template_nb)
    replace_cell_source(nb, 'STUDENT_ID = ""', f'STUDENT_ID = "{student_id}"')
    for fragment, new_source in replacements.items():
        if not replace_cell_source(nb, fragment, new_source):
            print(f"  WARNING: Could not find cell containing '{fragment[:50]}...'")
    return nb


# ─── Correct implementations ────────────────────────────────────────────

LINEAR_CORRECT = '''def linear_backward(y, x, w, b):
    x.grad += y.grad @ w
    w.grad += y.grad.T @ x
    b.grad += y.grad.sum(dim=0)


def linear(x, w, b, ctx=None):
    y = x @ w.T + b
    if ctx is not None:
        ctx.append([linear_backward, [y, x, w, b]])
    return y'''

RELU_CORRECT = '''def relu_backward(y, x):
    x.grad += y.grad * (x > 0).float()


def relu(x, ctx=None):
    y = x.clamp(min=0).clone()
    if ctx is not None:
        ctx.append([relu_backward, [y, x]])
    return y'''

CE_CORRECT = '''def cross_entropy_loss_backward(loss, z, targets):
    B = z.shape[0]
    probs = torch.softmax(z, dim=1)
    grad = probs.clone()
    grad[torch.arange(B), targets] -= 1.0
    z.grad += loss.grad * grad / B


def cross_entropy_loss(z, targets, ctx=None):
    B = z.shape[0]
    z_max = z.max(dim=1, keepdim=True).values
    log_sum_exp = (z - z_max).exp().sum(dim=1).log() + z_max.squeeze(1)
    loss = (log_sum_exp - z[torch.arange(B), targets]).mean()
    if ctx is not None:
        ctx.append([cross_entropy_loss_backward, [loss, z, targets]])
    return loss'''

LINEAR_LAYER_CORRECT = '''class Linear(Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self._parameters = ['weight', 'bias']
        k = 1.0 / (in_dim ** 0.5)
        self.weight = torch.empty(out_dim, in_dim).uniform_(-k, k)
        self.bias = torch.empty(out_dim).uniform_(-k, k)

    def forward(self, x, ctx=None):
        return linear(x, self.weight, self.bias, ctx=ctx)'''

MODELS_CORRECT = '''class SoftmaxClassifier(Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self._modules = ['fc']
        self.fc = Linear(in_dim, num_classes)

    def forward(self, x, ctx=None):
        return self.fc(x, ctx=ctx)


class MLP(Module):
    def __init__(self, in_dim, hidden_dim, num_classes):
        super().__init__()
        self._modules = ['fc1', 'fc2']
        self.fc1 = Linear(in_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, num_classes)

    def forward(self, x, ctx=None):
        x = self.fc1(x, ctx=ctx)
        x = relu(x, ctx=ctx)
        return self.fc2(x, ctx=ctx)'''

SGD_CORRECT = '''class SGD:
    def __init__(self, parameters, lr):
        self.parameters = parameters
        self.lr = lr

    def zero_grad(self):
        for p in self.parameters:
            p.grad = None

    def step(self):
        for p in self.parameters:
            if p.grad is not None:
                p.data -= self.lr * p.grad'''

TRAIN_CORRECT = '''def train_epoch(model, optimizer, loader):
    total_loss = 0
    n_batches = 0
    for images, labels in loader:
        x = images.view(images.size(0), -1)
        ctx = []
        optimizer.zero_grad()
        logits = model(x, ctx=ctx)
        loss = cross_entropy_loss(logits, labels, ctx=ctx)
        backward(loss, ctx)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


def evaluate(model, loader):
    correct = 0
    total = 0
    for images, labels in loader:
        x = images.view(images.size(0), -1)
        logits = model(x)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total'''

# ─── Buggy variants ──────────────────────────────────────────────────

LINEAR_NO_BIAS = '''def linear_backward(y, x, w, b):
    x.grad += y.grad @ w
    w.grad += y.grad.T @ x
    # Forgot bias gradient!


def linear(x, w, b, ctx=None):
    y = x @ w.T + b
    if ctx is not None:
        ctx.append([linear_backward, [y, x, w, b]])
    return y'''

RELU_WRONG_MASK = '''def relu_backward(y, x):
    x.grad += y.grad  # No masking! Gradient passes through everywhere


def relu(x, ctx=None):
    y = x.clamp(min=0).clone()
    if ctx is not None:
        ctx.append([relu_backward, [y, x]])
    return y'''

CE_NO_STABILITY = '''def cross_entropy_loss_backward(loss, z, targets):
    B = z.shape[0]
    probs = torch.softmax(z, dim=1)
    grad = probs.clone()
    grad[torch.arange(B), targets] -= 1.0
    z.grad += loss.grad * grad / B


def cross_entropy_loss(z, targets, ctx=None):
    B = z.shape[0]
    # No numerical stability (no max subtraction)
    log_sum_exp = z.exp().sum(dim=1).log()
    loss = (log_sum_exp - z[torch.arange(B), targets]).mean()
    if ctx is not None:
        ctx.append([cross_entropy_loss_backward, [loss, z, targets]])
    return loss'''

# ─── Fragments to match ──────────────────────────────────────────────

LINEAR_FRAG = "def linear_backward(y, x, w, b):"
RELU_FRAG = "def relu_backward(y, x):"
CE_FRAG = "def cross_entropy_loss_backward(loss, z, targets):"
LAYER_FRAG = "class Linear(Module):"
MODELS_FRAG = "class SoftmaxClassifier(Module):"
SGD_FRAG = "class SGD:"
TRAIN_FRAG = "def train_epoch(model, optimizer, loader):"

# ─── Generate notebooks ──────────────────────────────────────────────

out = Path("test_hw2_scratch_submissions")
out.mkdir(exist_ok=True)

notebooks = {
    "perfect": ("PERFECT_201", {
        LINEAR_FRAG: LINEAR_CORRECT,
        RELU_FRAG: RELU_CORRECT,
        CE_FRAG: CE_CORRECT,
        LAYER_FRAG: LINEAR_LAYER_CORRECT,
        MODELS_FRAG: MODELS_CORRECT,
        SGD_FRAG: SGD_CORRECT,
        TRAIN_FRAG: TRAIN_CORRECT,
    }),
    "no_bias": ("NOBIAS_202", {
        LINEAR_FRAG: LINEAR_NO_BIAS,
        RELU_FRAG: RELU_CORRECT,
        CE_FRAG: CE_CORRECT,
        LAYER_FRAG: LINEAR_LAYER_CORRECT,
        MODELS_FRAG: MODELS_CORRECT,
        SGD_FRAG: SGD_CORRECT,
        TRAIN_FRAG: TRAIN_CORRECT,
    }),
    "unstable_ce": ("UNSTABLE_203", {
        LINEAR_FRAG: LINEAR_CORRECT,
        RELU_FRAG: RELU_CORRECT,
        CE_FRAG: CE_NO_STABILITY,
        LAYER_FRAG: LINEAR_LAYER_CORRECT,
        MODELS_FRAG: MODELS_CORRECT,
        SGD_FRAG: SGD_CORRECT,
        TRAIN_FRAG: TRAIN_CORRECT,
    }),
    "partial": ("PARTIAL_204", {
        LINEAR_FRAG: LINEAR_CORRECT,
        RELU_FRAG: RELU_CORRECT,
        # CE, models, SGD, train all left as NotImplementedError
    }),
}

for name, (sid, replacements) in notebooks.items():
    nb = make_submission(sid, replacements)
    path = out / f"student_{name}.ipynb"
    nbformat.write(nb, str(path))
    print(f"  Created: {path.name}")

print(f"\nDone! {len(notebooks)} notebooks in {out}/")
