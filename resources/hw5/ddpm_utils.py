"""
Provided helpers for HW5 — From Scratch: DDPM (Denoising Diffusion).

DO NOT MODIFY THIS FILE — it will be replaced during grading.

You implement the *diffusion* in the notebook (the noise schedule, the forward
process, the training loss, and the sampler). Everything here is plumbing you
don't need to read: the dataset, a small time-conditioned U-Net (the network
that predicts the noise), a training loop, and image-grid plotting.

Provides:
    device, get_mnist, grab            (device, data, schedule-indexing helper)
    UNet                               (the noise-prediction network)
    train                              (a standard optimisation loop)
    show                               (plot a grid of images)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchvision.utils import make_grid

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ════════════════════════════════════════════════════════════════════════════
#  Data: MNIST, padded to 32x32 and scaled to [-1, 1].
# ════════════════════════════════════════════════════════════════════════════
def get_mnist(batch_size=128, n=None):
    """MNIST DataLoader. Images are 1x32x32 in [-1, 1]. `n` keeps a subset."""
    tf = transforms.Compose([transforms.Pad(2), transforms.ToTensor(),
                             transforms.Normalize((0.5,), (0.5,))])
    ds = datasets.MNIST('./data', train=True, download=True, transform=tf)
    if n is not None:
        ds = torch.utils.data.Subset(ds, range(n))
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

def grab(vals, t):
    """Pick schedule values at timesteps t and shape them to broadcast over
    image batches: vals is (T,), t is (B,) -> returns (B, 1, 1, 1)."""
    return vals[t].view(-1, 1, 1, 1)

# ════════════════════════════════════════════════════════════════════════════
#  The noise-prediction network: a small time-conditioned U-Net (provided).
# ════════════════════════════════════════════════════════════════════════════
class _SinTime(nn.Module):
    def __init__(self, dim): super().__init__(); self.dim = dim
    def forward(self, t):
        half = self.dim // 2
        f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        a = t[:, None].float() * f[None]
        return torch.cat([a.sin(), a.cos()], -1)

class _Res(nn.Module):
    def __init__(self, ci, co, td):
        super().__init__()
        self.n1 = nn.GroupNorm(8, ci); self.c1 = nn.Conv2d(ci, co, 3, padding=1)
        self.temb = nn.Linear(td, co)
        self.n2 = nn.GroupNorm(8, co); self.c2 = nn.Conv2d(co, co, 3, padding=1)
        self.skip = nn.Conv2d(ci, co, 1) if ci != co else nn.Identity()
    def forward(self, x, t):
        h = self.c1(F.silu(self.n1(x))) + self.temb(t)[:, :, None, None]
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)

class UNet(nn.Module):
    """Predicts the noise eps added to an image, given the noisy image x_t and
    the timestep t. Call it as  eps = model(x_t, t)  with x_t:(B,1,32,32),
    t:(B,) integer timesteps."""
    def __init__(self, C=64, td=256):
        super().__init__()
        self.temb = nn.Sequential(_SinTime(td), nn.Linear(td, td), nn.SiLU(), nn.Linear(td, td))
        self.inp = nn.Conv2d(1, C, 3, padding=1)
        self.d1 = _Res(C, C, td); self.d2 = _Res(C, 2 * C, td); self.d3 = _Res(2 * C, 4 * C, td)
        self.mid = _Res(4 * C, 4 * C, td)
        self.u3 = _Res(8 * C, 2 * C, td); self.u2 = _Res(4 * C, C, td); self.u1 = _Res(2 * C, C, td)
        self.down = nn.AvgPool2d(2); self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.out = nn.Sequential(nn.GroupNorm(8, C), nn.SiLU(), nn.Conv2d(C, 1, 3, padding=1))
    def forward(self, x, t):
        t = self.temb(t)
        x0 = self.inp(x)                                  # C  @32
        h1 = self.d1(x0, t)                               # C  @32
        h2 = self.d2(self.down(h1), t)                    # 2C @16
        h3 = self.d3(self.down(h2), t)                    # 4C @8
        m  = self.mid(self.down(h3), t)                   # 4C @4
        h  = self.u3(torch.cat([self.up(m), h3], 1), t)   # 2C @8
        h  = self.u2(torch.cat([self.up(h), h2], 1), t)   # C  @16
        h  = self.u1(torch.cat([self.up(h), h1], 1), t)   # C  @32
        return self.out(h)

# ════════════════════════════════════════════════════════════════════════════
#  A standard training loop (provided). You supply the loss function.
# ════════════════════════════════════════════════════════════════════════════
def train(model, loader, loss_fn, epochs=5, lr=1e-3):
    """Optimise `model` so that `loss_fn(model, x0)` is minimised over the data.
    Returns the list of per-step losses. (AdamW with a one-cycle cosine LR schedule:
    standard, robust defaults for training a diffusion model to clean samples.)"""
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, epochs=epochs, steps_per_epoch=len(loader), pct_start=0.25)
    losses = []
    for ep in range(epochs):
        for x0, _ in loader:
            loss = loss_fn(model, x0.to(device))
            opt.zero_grad(); loss.backward(); opt.step(); lr_sched.step()
            losses.append(loss.item())
        print(f'epoch {ep + 1}/{epochs}   loss {sum(losses[-100:]) / min(len(losses), 100):.4f}')
    return losses

# ════════════════════════════════════════════════════════════════════════════
#  Plotting.
# ════════════════════════════════════════════════════════════════════════════
def show(imgs, title=None, ncol=8):
    """Plot a grid of images (tensor (N,1,H,W) in [-1,1])."""
    x = (imgs.detach().cpu().float().clamp(-1, 1) + 1) / 2
    grid = make_grid(x, nrow=ncol, padding=1, pad_value=0.4).permute(1, 2, 0)
    plt.figure(figsize=(ncol * 0.7, (len(x) / ncol) * 0.7 + 0.3))
    plt.imshow(grid, cmap='gray'); plt.axis('off')
    if title: plt.title(title, fontsize=11)
    plt.show()
