#!/usr/bin/env python3
"""Generate synthetic student submissions for HW2 applied (digit addition)."""

import copy
import nbformat
from pathlib import Path

TEMPLATE_PATH = Path("hw2_applied.ipynb")
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


# ─── Model definition ────────────────────────────────────────────────

MODEL_MLP = '''class DigitClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = DigitClassifier().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)'''

MODEL_CNN = '''class DigitClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = DigitClassifier().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)'''

# ─── Training loops ──────────────────────────────────────────────────

TRAIN_GOOD = '''train_losses = []
test_accs = []

for epoch in range(10):
    model.train()
    total_loss = 0
    n = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = F.cross_entropy(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n += 1
    avg_loss = total_loss / n
    train_losses.append(avg_loss)

    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / total
    test_accs.append(acc)
    print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, acc={acc:.2%}")'''

TRAIN_SHORT = '''train_losses = []
test_accs = []

for epoch in range(2):  # Only 2 epochs — might not converge
    model.train()
    total_loss = 0; n = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = F.cross_entropy(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item(); n += 1
    train_losses.append(total_loss / n)

    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(1) == labels).sum().item()
            total += labels.size(0)
    test_accs.append(correct / total)
    print(f"Epoch {epoch+1}: loss={train_losses[-1]:.4f}, acc={test_accs[-1]:.2%}")'''

# ─── predict_sum variants ────────────────────────────────────────────

PREDICT_CONVOLUTION = '''def predict_sum(model, img1, img2):
    """Use probability convolution: P(A+B=k) = sum_i P(A=i)*P(B=k-i)."""
    model.eval()
    with torch.no_grad():
        logits1 = model(img1)
        logits2 = model(img2)
        p1 = torch.softmax(logits1, dim=1).cpu()  # (B, 10)
        p2 = torch.softmax(logits2, dim=1).cpu()  # (B, 10)

    B = p1.shape[0]
    # Convolve each pair of distributions
    p_sum = torch.zeros(B, 19)
    for i in range(10):
        for j in range(10):
            p_sum[:, i + j] += p1[:, i] * p2[:, j]

    return p_sum.argmax(dim=1)'''

PREDICT_ARGMAX = '''def predict_sum(model, img1, img2):
    """Simple argmax approach: predict each digit, add them."""
    model.eval()
    with torch.no_grad():
        pred1 = model(img1).argmax(dim=1)
        pred2 = model(img2).argmax(dim=1)
    return (pred1 + pred2).cpu()'''

PREDICT_NUMPY_CONV = '''def predict_sum(model, img1, img2):
    """Use numpy convolve for the probability combination."""
    model.eval()
    with torch.no_grad():
        p1 = torch.softmax(model(img1), dim=1).cpu().numpy()
        p2 = torch.softmax(model(img2), dim=1).cpu().numpy()

    B = p1.shape[0]
    preds = []
    for b in range(B):
        p_sum = np.convolve(p1[b], p2[b])  # length 19
        preds.append(np.argmax(p_sum))
    return torch.tensor(preds)'''

# ─── Approach explanations ────────────────────────────────────────────

EXPLAIN_CONV = '''**Your approach (explain here):**

I use probability convolution. The classifier gives a probability distribution over digits 0-9 for each image.
The probability that the sum equals k is P(A+B=k) = sum over i of P(A=i) * P(B=k-i).
This is exactly a discrete convolution of the two probability vectors, giving a distribution over 0-18.
I take the argmax of this distribution as my prediction.

This is better than just adding the argmax predictions because it properly accounts for uncertainty.
If the classifier is 60% sure digit A is "3" and 40% sure it's "8", and digit B is clearly "5",
the argmax approach always predicts 8 (3+5), but the convolution approach correctly weighs
both possibilities: sum=8 (60%) vs sum=13 (40%).'''

EXPLAIN_ARGMAX = '''**Your approach (explain here):**

I predict each digit using argmax on the classifier output, then add the two predictions.'''

# ─── Fragments ────────────────────────────────────────────────────────

MODEL_FRAG = "# YOUR CODE HERE — define your model"
TRAIN_FRAG = "# YOUR CODE HERE — training loop"
PREDICT_FRAG = "def predict_sum(model, img1, img2):"
EXPLAIN_FRAG = "**Your approach (explain here):**"

# ─── Generate ─────────────────────────────────────────────────────────

out = Path("test_hw2_applied_submissions")
out.mkdir(exist_ok=True)

notebooks = {
    "perfect_conv": ("PERFECT_211", {
        MODEL_FRAG: MODEL_CNN,
        TRAIN_FRAG: TRAIN_GOOD,
        PREDICT_FRAG: PREDICT_CONVOLUTION,
        EXPLAIN_FRAG: EXPLAIN_CONV,
    }),
    "perfect_numpy": ("NUMPY_212", {
        MODEL_FRAG: MODEL_CNN,
        TRAIN_FRAG: TRAIN_GOOD,
        PREDICT_FRAG: PREDICT_NUMPY_CONV,
        EXPLAIN_FRAG: EXPLAIN_CONV,
    }),
    "argmax_only": ("ARGMAX_213", {
        MODEL_FRAG: MODEL_MLP,
        TRAIN_FRAG: TRAIN_GOOD,
        PREDICT_FRAG: PREDICT_ARGMAX,
        EXPLAIN_FRAG: EXPLAIN_ARGMAX,
    }),
    "undertrained": ("UNDER_214", {
        MODEL_FRAG: MODEL_MLP,
        TRAIN_FRAG: TRAIN_SHORT,
        PREDICT_FRAG: PREDICT_CONVOLUTION,
        EXPLAIN_FRAG: EXPLAIN_CONV,
    }),
}

for name, (sid, replacements) in notebooks.items():
    nb = make_submission(sid, replacements)
    path = out / f"student_{name}.ipynb"
    nbformat.write(nb, str(path))
    print(f"  Created: {path.name}")

print(f"\nDone! {len(notebooks)} notebooks in {out}/")
