"""
Utility functions for HW3 Applied: Zero-Shot Super Resolution (ZSSR).

DO NOT MODIFY THIS FILE — it will be replaced during grading.

Provides: psnr, resize_bicubic, load_test_images, visualize_sr,
          plot_training_curves, make_flickering_gif
"""

import torch
import torch.nn as nn
import numpy as np
import math
import io
import base64
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from IPython.display import display, HTML
from resize_right import resize as resize_right_resize
from resize_right.interp_methods import cubic as cubic_interp


# ═══════════════════════════════════════════════════════════════════════════════
# Core functions
# ═══════════════════════════════════════════════════════════════════════════════

def psnr(img1, img2):
    """
    Calculate PSNR between two images. Both should be in [0, 1] range.
    Handles torch tensors and numpy arrays.
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().float().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().float().numpy()
    img1 = np.clip(img1.astype(np.float64), 0, 1)
    img2 = np.clip(img2.astype(np.float64), 0, 1)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(1.0 / np.sqrt(mse))


def resize_bicubic(image_tensor, scale_factor=None, size=None):
    """
    Resize using resize_right for correct pixel alignment (no half-pixel shift).

    Handles 2D (H,W), 3D (C,H,W), and 4D (B,C,H,W) inputs.
    Use this instead of F.interpolate for correct downsampling with antialiasing.

    Args:
        image_tensor: input tensor
        scale_factor: float scaling factor (e.g., 0.25 for 4x downscale)
        size: (H, W) target size tuple
    """
    ndim = image_tensor.dim()
    if ndim == 2:
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
    elif ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)

    if scale_factor is not None:
        scale_factors = [1, 1, scale_factor, scale_factor]
        resized = resize_right_resize(image_tensor, scale_factors=scale_factors,
                                       interp_method=cubic_interp,
                                       antialiasing=(scale_factor < 1))
    elif size is not None:
        if isinstance(size, int):
            size = (size, size)
        out_shape = [image_tensor.shape[0], image_tensor.shape[1], size[0], size[1]]
        resized = resize_right_resize(image_tensor, out_shape=out_shape,
                                       interp_method=cubic_interp,
                                       antialiasing=True)
    else:
        raise ValueError("Must provide either scale_factor or size")

    if ndim == 2:
        return resized.squeeze(0).squeeze(0)
    elif ndim == 3:
        return resized.squeeze(0)
    return resized


# ═══════════════════════════════════════════════════════════════════════════════
# Test images
# ═══════════════════════════════════════════════════════════════════════════════

def load_test_images():
    """
    Load RGB test images for ZSSR experiments.
    Returns dict mapping image name -> (C, H, W) float tensor in [0, 1].
    """
    from skimage import data as skdata
    images = {}

    astro = skdata.astronaut().astype(np.float32) / 255.0
    images['Astronaut'] = torch.from_numpy(astro).permute(2, 0, 1)  # (3, 512, 512)

    chelsea = skdata.chelsea().astype(np.float32) / 255.0
    images['Chelsea'] = torch.from_numpy(chelsea).permute(2, 0, 1)  # (3, 300, 451)

    return images


# ═══════════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════════

def _to_display(img):
    """Convert image tensor/array to (H, W, C) numpy for display."""
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().float().numpy()
    img = np.clip(img, 0, 1)
    if img.ndim == 2:
        return np.stack([img] * 3, axis=-1)
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = np.transpose(img, (1, 2, 0))
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    return img


def make_flickering_gif(img_a, img_b, label_a, label_b, duration=800):
    """Create a flickering GIF toggling between two images. Returns HTML string."""
    frames = []
    for arr, label in [(img_a, label_a), (img_b, label_b)]:
        arr_disp = _to_display(arr)
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(arr_disp)
        ax.set_title(label, fontsize=16, fontweight='bold', pad=8)
        ax.axis('off')
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        frames.append(PILImage.open(buf).convert('RGB'))

    gif_buf = io.BytesIO()
    frames[0].save(gif_buf, format='GIF', append_images=frames[1:],
                   save_all=True, duration=duration, loop=0)
    gif_buf.seek(0)
    gif_b64 = base64.b64encode(gif_buf.read()).decode()
    return f'<img src="data:image/gif;base64,{gif_b64}" style="max-width:100%;"/>'


def visualize_sr(sr_image, lr_image, hr_image=None, title="ZSSR Result"):
    """
    Visualize super-resolution results with flickering Bicubic vs ZSSR comparison.

    Args:
        sr_image: Super-resolved image (C, H, W) tensor
        lr_image: Low-resolution input (C, H, W) tensor
        hr_image: Optional ground-truth HR for PSNR computation
        title: Display title

    Returns:
        dict with PSNR values if hr_image is provided, else empty dict.
    """
    sr = torch.clamp(sr_image.detach().cpu().float(), 0, 1)
    lr = lr_image.detach().cpu().float()

    # Bicubic baseline: upsample LR to SR size
    bic = torch.clamp(resize_bicubic(lr, size=(sr.shape[-2], sr.shape[-1])), 0, 1)

    if hr_image is not None:
        hr = hr_image.detach().cpu().float()
        # Resize SR and bicubic to match HR if needed
        if sr.shape[-2:] != hr.shape[-2:]:
            sr = torch.clamp(resize_bicubic(sr, size=(hr.shape[-2], hr.shape[-1])), 0, 1)
            bic = torch.clamp(resize_bicubic(bic, size=(hr.shape[-2], hr.shape[-1])), 0, 1)

        psnr_bic = psnr(bic, hr)
        psnr_sr = psnr(sr, hr)
        label_a = f'Bicubic ({psnr_bic:.2f} dB)'
        label_b = f'ZSSR ({psnr_sr:.2f} dB)'
        improvement = psnr_sr - psnr_bic

        gif_html = make_flickering_gif(bic, sr, label_a, label_b)

        # Ground truth panel
        gt_disp = _to_display(hr)
        fig_gt, ax_gt = plt.subplots(1, 1, figsize=(6, 6))
        ax_gt.imshow(gt_disp)
        ax_gt.set_title('Ground Truth', fontsize=16, fontweight='bold')
        ax_gt.axis('off')
        fig_gt.tight_layout()
        buf_gt = io.BytesIO()
        fig_gt.savefig(buf_gt, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig_gt)
        buf_gt.seek(0)
        gt_b64 = base64.b64encode(buf_gt.read()).decode()

        html = f'''
        <h3>{title} — Improvement: {improvement:+.2f} dB</h3>
        <div style="display:flex; gap:20px; flex-wrap:wrap;">
          <div>{gif_html}</div>
          <div><img src="data:image/png;base64,{gt_b64}" style="max-width:100%;"/></div>
        </div>'''
        display(HTML(html))
        return {'PSNR_Bicubic': psnr_bic, 'PSNR_ZSSR': psnr_sr}
    else:
        gif_html = make_flickering_gif(bic, sr, 'Bicubic', 'ZSSR')
        display(HTML(f'<h3>{title}</h3>{gif_html}'))
        return {}


def plot_training_curves(loss_history, psnr_history=None, figsize=(14, 5)):
    """Plot training loss and optionally PSNR curves."""
    n_plots = 2 if psnr_history else 1
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]

    axes[0].plot(loss_history, linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True, alpha=0.3)

    if psnr_history:
        axes[1].plot(psnr_history, linewidth=2, color='green')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('PSNR (dB)')
        axes[1].set_title('Training PSNR')
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
