#!/usr/bin/env python3
"""
Generate synthetic student submissions for the applied (seam carving) notebook.

These are copies of the actual assignment notebook with only the student code cells
filled in — so they have all the markdown instructions, image loading, and visualizations.
"""

import copy
import nbformat
from pathlib import Path

# Load the actual assignment notebook as template
TEMPLATE_PATH = Path("hw1_applied.ipynb")
template_nb = nbformat.read(str(TEMPLATE_PATH), as_version=4)


def find_cell_by_content(nb, content_fragment):
    """Find a cell index containing the given text."""
    for i, cell in enumerate(nb.cells):
        if content_fragment in cell.source:
            return i
    return None


def replace_cell_source(nb, content_fragment, new_source):
    """Replace the source of a cell that contains the given text."""
    idx = find_cell_by_content(nb, content_fragment)
    if idx is not None:
        nb.cells[idx].source = new_source
        return True
    return False


def make_submission(student_id, replacements):
    """
    Create a submission notebook from the template.

    replacements: dict mapping a content fragment (to identify the cell) to new source code.
    """
    nb = copy.deepcopy(template_nb)

    # Set student ID
    replace_cell_source(nb, 'STUDENT_ID = ""', f'STUDENT_ID = "{student_id}"')

    for fragment, new_source in replacements.items():
        if not replace_cell_source(nb, fragment, new_source):
            print(f"  WARNING: Could not find cell containing '{fragment[:50]}...'")

    return nb


# ─── Correct implementations ────────────────────────────────────────────────

ENERGY_CORRECT = """def compute_energy(gray):
    H, W = gray.shape
    sx = torch.tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]])
    sy = torch.tensor([[-1.,-2.,-1.],[0.,0.,0.],[1.,2.,1.]])
    gx = conv2d(gray, sx)
    gy = conv2d(gray, sy)
    e = gx.abs() + gy.abs()
    return F.pad(e.unsqueeze(0), (1, 1, 1, 1), mode='replicate').squeeze(0)"""

SEAM_CORRECT = """def find_seam(energy):
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
    return seam"""

REMOVE_CORRECT = """def remove_seam(img, seam):
    if img.dim() == 2:
        H, W = img.shape
        mask = torch.ones(H, W, dtype=torch.bool)
        for i in range(H):
            mask[i, seam[i]] = False
        return img[mask].reshape(H, W-1)
    else:
        H, W, C = img.shape
        mask = torch.ones(H, W, dtype=torch.bool)
        for i in range(H):
            mask[i, seam[i]] = False
        return img[mask.unsqueeze(-1).expand_as(img)].reshape(H, W-1, C)"""

CARVE_CORRECT = """def seam_carve(img, n_seams, show_progress=True):
    result = img.clone()
    for k in range(n_seams):
        gray = to_grayscale(result)
        energy = compute_energy(gray)
        seam = find_seam(energy)
        result = remove_seam(result, seam)
        if show_progress and (k+1) % 10 == 0:
            print(f'  Removed {k+1}/{n_seams} seams')
    return result"""

INSERT_CORRECT = """def seam_insert(img, n_seams, show_progress=True):
    H, W, C = img.shape
    # Step 1: Find n seams on the original by iteratively finding & removing
    temp = img.clone()
    seams_on_shrunk = []
    for k in range(n_seams):
        gray = to_grayscale(temp)
        energy = compute_energy(gray)
        seam = find_seam(energy)
        seams_on_shrunk.append(seam.clone())
        temp = remove_seam(temp, seam)
        if show_progress and (k+1) % 10 == 0:
            print(f'  Found {k+1}/{n_seams} seams')

    # Step 2: Map seams back to original coordinates
    seams_original = [s.clone() for s in seams_on_shrunk]
    for i in range(len(seams_original)):
        for j in range(i):
            seams_original[i] = torch.where(
                seams_original[i] >= seams_original[j],
                seams_original[i] + 1,
                seams_original[i]
            )

    # Step 3: Insert seams from rightmost to leftmost
    result = img.clone()
    all_seams = sorted(range(len(seams_original)),
                       key=lambda k: -seams_original[k].float().mean())

    for idx in all_seams:
        seam = seams_original[idx]
        H_cur, W_cur = result.shape[0], result.shape[1]
        new_img = torch.zeros(H_cur, W_cur + 1, C)
        for row in range(H_cur):
            col = seam[row].item()
            new_img[row, :col+1] = result[row, :col+1]
            left_val = result[row, col]
            right_val = result[row, min(col+1, W_cur-1)]
            new_img[row, col+1] = (left_val + right_val) / 2
            new_img[row, col+2:] = result[row, col+1:]
        result = new_img
        for other_idx in all_seams:
            if other_idx != idx:
                seams_original[other_idx] = torch.where(
                    seams_original[other_idx] >= seam,
                    seams_original[other_idx] + 1,
                    seams_original[other_idx]
                )

    return result"""

# ─── Buggy variants ──────────────────────────────────────────────────────

ENERGY_NO_PAD = """def compute_energy(gray):
    sx = torch.tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]])
    sy = torch.tensor([[-1.,-2.,-1.],[0.,0.,0.],[1.,2.,1.]])
    gx = conv2d(gray, sx)
    gy = conv2d(gray, sy)
    return gx.abs() + gy.abs()  # forgot to pad!"""

ENERGY_NO_ABS = """def compute_energy(gray):
    H, W = gray.shape
    sx = torch.tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]])
    sy = torch.tensor([[-1.,-2.,-1.],[0.,0.,0.],[1.,2.,1.]])
    gx = conv2d(gray, sx)
    gy = conv2d(gray, sy)
    e = gx + gy  # forgot absolute values!
    return F.pad(e.unsqueeze(0), (1, 1, 1, 1), mode='replicate').squeeze(0)"""

SEAM_GREEDY = """def find_seam(energy):
    H, W = energy.shape
    seam = torch.zeros(H, dtype=torch.long)
    seam[0] = energy[0].argmin()
    for i in range(1, H):
        j = seam[i-1].item()
        lo = max(0, j-1)
        hi = min(W, j+2)
        seam[i] = lo + energy[i, lo:hi].argmin()
    return seam"""

# ─── Generate notebooks ──────────────────────────────────────────────────

out = Path("test_applied_submissions")
out.mkdir(exist_ok=True)

# Identify the cell fragments to replace
ENERGY_FRAGMENT = "def compute_energy(gray):"
SEAM_FRAGMENT = "def find_seam(energy):"
REMOVE_FRAGMENT = "def remove_seam(img, seam):"
CARVE_FRAGMENT = "def seam_carve(img, n_seams"
INSERT_FRAGMENT = "def seam_insert(img, n_seams"

notebooks = {
    "perfect": ("PERFECT_101", {
        ENERGY_FRAGMENT: ENERGY_CORRECT,
        SEAM_FRAGMENT: SEAM_CORRECT,
        REMOVE_FRAGMENT: REMOVE_CORRECT,
        CARVE_FRAGMENT: CARVE_CORRECT,
        INSERT_FRAGMENT: INSERT_CORRECT,
    }),
    "no_pad": ("NOPAD_102", {
        ENERGY_FRAGMENT: ENERGY_NO_PAD,
        SEAM_FRAGMENT: SEAM_CORRECT,
        REMOVE_FRAGMENT: REMOVE_CORRECT,
        CARVE_FRAGMENT: CARVE_CORRECT,
        INSERT_FRAGMENT: INSERT_CORRECT,
    }),
    "no_abs": ("NOABS_103", {
        ENERGY_FRAGMENT: ENERGY_NO_ABS,
        SEAM_FRAGMENT: SEAM_CORRECT,
        REMOVE_FRAGMENT: REMOVE_CORRECT,
        CARVE_FRAGMENT: CARVE_CORRECT,
        INSERT_FRAGMENT: INSERT_CORRECT,
    }),
    "greedy_seam": ("GREEDY_104", {
        ENERGY_FRAGMENT: ENERGY_CORRECT,
        SEAM_FRAGMENT: SEAM_GREEDY,
        REMOVE_FRAGMENT: REMOVE_CORRECT,
        CARVE_FRAGMENT: CARVE_CORRECT,
        INSERT_FRAGMENT: INSERT_CORRECT,
    }),
    "partial": ("PARTIAL_105", {
        ENERGY_FRAGMENT: ENERGY_CORRECT,
        SEAM_FRAGMENT: "def find_seam(energy):\n    raise NotImplementedError",
        REMOVE_FRAGMENT: "def remove_seam(img, seam):\n    raise NotImplementedError",
        CARVE_FRAGMENT: "def seam_carve(img, n_seams, show_progress=True):\n    raise NotImplementedError",
        INSERT_FRAGMENT: "def seam_insert(img, n_seams, show_progress=True):\n    raise NotImplementedError",
    }),
}

for name, (sid, replacements) in notebooks.items():
    nb = make_submission(sid, replacements)
    path = out / f"student_{name}.ipynb"
    nbformat.write(nb, str(path))
    print(f"  Created: {path.name}")

print(f"\nDone! {len(notebooks)} notebooks in {out}/")
