#!/usr/bin/env python3
"""
Generate mock student submissions for HW3 applied (ZSSR).
Creates 4 variants: perfect, no_residual, bad_dataset, minimal
"""

import json
from pathlib import Path
from copy import deepcopy

ZSSR_NET_PERFECT = '''class ZSSRNet(nn.Module):
    """
    Zero-Shot Super Resolution Network with residual connection.
    """
    def __init__(self, in_channels=3, n_channels=64, n_layers=8, kernel_size=3):
        super(ZSSRNet, self).__init__()

        self.in_channels = in_channels
        self.n_channels = n_channels
        self.n_layers = n_layers
        self.kernel_size = kernel_size

        # Build the network
        self.layers = nn.ModuleList()
        padding = kernel_size // 2

        # First layer: in_channels -> n_channels
        self.layers.append(
            nn.Conv2d(in_channels, n_channels, kernel_size, padding=padding)
        )

        # Intermediate layers: n_channels -> n_channels
        for _ in range(n_layers - 2):
            self.layers.append(
                nn.Conv2d(n_channels, n_channels, kernel_size, padding=padding)
            )

        # Last layer: n_channels -> in_channels
        self.layers.append(
            nn.Conv2d(n_channels, in_channels, kernel_size, padding=padding)
        )

        # Activation
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, scale_factor=2, target_size=None):
        # Handle input shape: (C,H,W) -> (1,C,H,W)
        squeeze_out = False
        if x.dim() == 3:
            x = x.unsqueeze(0)
            squeeze_out = True

        # Bicubic upsampling
        if target_size is None:
            target_size = (x.shape[-2] * scale_factor, x.shape[-1] * scale_factor)

        upscaled = resize_bicubic(x, size=target_size)

        # Residual network
        residual = upscaled.clone()

        # Pass through network
        out = upscaled
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if i < len(self.layers) - 1:
                out = self.relu(out)

        # Add residual connection
        out = out + residual

        if squeeze_out:
            out = out.squeeze(0)
        return out'''

ZSSR_NET_NO_RESIDUAL = '''class ZSSRNet(nn.Module):
    """
    Zero-Shot Super Resolution Network without residual connection (BUG).
    """
    def __init__(self, in_channels=3, n_channels=64, n_layers=8, kernel_size=3):
        super(ZSSRNet, self).__init__()

        self.in_channels = in_channels
        self.n_channels = n_channels
        self.n_layers = n_layers
        self.kernel_size = kernel_size

        self.layers = nn.ModuleList()
        padding = kernel_size // 2

        self.layers.append(
            nn.Conv2d(in_channels, n_channels, kernel_size, padding=padding)
        )

        for _ in range(n_layers - 2):
            self.layers.append(
                nn.Conv2d(n_channels, n_channels, kernel_size, padding=padding)
            )

        self.layers.append(
            nn.Conv2d(n_channels, in_channels, kernel_size, padding=padding)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, scale_factor=2, target_size=None):
        squeeze_out = False
        if x.dim() == 3:
            x = x.unsqueeze(0)
            squeeze_out = True

        if target_size is None:
            target_size = (x.shape[-2] * scale_factor, x.shape[-1] * scale_factor)

        upscaled = resize_bicubic(x, size=target_size)

        # BUG: no residual connection
        out = upscaled
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if i < len(self.layers) - 1:
                out = self.relu(out)

        # BUG: missing out = out + residual

        if squeeze_out:
            out = out.squeeze(0)
        return out'''

ZSSR_DATASET_PERFECT = '''class ZSSRDataset(Dataset):
    """
    Dataset for Zero-Shot Super Resolution on a single image.
    Crops from the image first, then downscales on-the-fly to create LR.
    """
    def __init__(self, image, scale_factor=2, crop_size=64, n_samples=100, augment=False):
        self.image = image  # Shape: (C, H, W)
        self.scale_factor = scale_factor
        self.crop_size = crop_size
        self.n_samples = n_samples
        self.augment = augment

    def __len__(self):
        return self.n_samples

    def _augment_8fold(self, img):
        """Generate 8 augmented versions: 4 rotations x 2 flips.
        Input: (C, H, W), Output: list of 8 tensors (C, H, W)"""
        augmented = []
        for k in range(4):
            img_rot = torch.rot90(img, k=k, dims=[-2, -1])
            augmented.append(img_rot)
            augmented.append(torch.flip(img_rot, dims=[-1]))
        return augmented

    def __getitem__(self, idx):
        # Random crop from original image -> this is HR
        h, w = self.image.shape[-2], self.image.shape[-1]
        crop_h = min(self.crop_size, h)
        crop_w = min(self.crop_size, w)

        top = np.random.randint(0, max(1, h - crop_h + 1))
        left = np.random.randint(0, max(1, w - crop_w + 1))

        hr_crop = self.image[:, top:top+crop_h, left:left+crop_w]

        # Downscale the crop on-the-fly to get LR (no coordinate mapping!)
        lr_crop = resize_bicubic(hr_crop, scale_factor=1.0/self.scale_factor)

        if self.augment:
            hr_crops = self._augment_8fold(hr_crop)
            lr_crops = self._augment_8fold(lr_crop)
            return {
                \'HR\': torch.stack(hr_crops),  # (8, C, H, W)
                \'LR\': torch.stack(lr_crops),  # (8, C, H//sf, W//sf)
            }
        else:
            return {
                \'HR\': hr_crop,   # (C, H, W)
                \'LR\': lr_crop,   # (C, H//sf, W//sf)
            }'''

ZSSR_DATASET_BAD = '''class ZSSRDataset(Dataset):
    """
    Dataset for ZSSR (BUG: pre-downscales then tries to match coordinates).
    """
    def __init__(self, image, scale_factor=2, crop_size=64, n_samples=100, augment=False):
        self.image = image  # Shape: (C, H, W)
        self.scale_factor = scale_factor
        self.crop_size = crop_size
        self.n_samples = n_samples
        self.augment = augment

        # BUG: pre-downscaling and trying to match coordinates
        self.lr_image = resize_bicubic(image, scale_factor=0.5)
        self.hr_image = image

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        h, w = self.hr_image.shape[-2], self.hr_image.shape[-1]
        crop_h = min(self.crop_size, h)
        crop_w = min(self.crop_size, w)

        top = np.random.randint(0, max(1, h - crop_h + 1))
        left = np.random.randint(0, max(1, w - crop_w + 1))

        hr_crop = self.hr_image[:, top:top+crop_h, left:left+crop_w]

        # BUG: wrong indexing into LR image (uses HR coordinates)
        lr_crop = self.lr_image[:, top:top+crop_h, left:left+crop_w]

        return {
            \'HR\': hr_crop,
            \'LR\': lr_crop,  # wrong shape and misaligned!
        }'''

TRAIN_ZSSR_PERFECT = '''def train_zssr(
    model,
    dataloader,
    scale_factor=2,
    num_epochs=100,
    learning_rate=0.001,
    device='cpu',
    verbose=True
):
    """
    Train the ZSSR model on a single image.
    """
    model = model.to(device)

    criterion = nn.L1Loss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.5)

    loss_history = []
    psnr_history = []

    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_psnr = 0.0
        num_batches = 0

        for batch in dataloader:
            hr = batch[\'HR\'].to(device)
            lr = batch[\'LR\'].to(device)

            # If augmented (B, 8, C, H, W), merge aug into batch
            if hr.dim() == 5:
                B, aug, C, h, w = hr.shape
                hr = hr.view(B * aug, C, h, w)
                lr = lr.view(B * aug, C, lr.shape[-2], lr.shape[-1])

            # Forward pass
            optimizer.zero_grad()
            target_h, target_w = hr.shape[-2], hr.shape[-1]
            sr = model(lr, scale_factor=scale_factor, target_size=(target_h, target_w))

            # Ensure shapes match
            if sr.shape != hr.shape:
                sr = resize_bicubic(sr, size=(target_h, target_w))

            loss = criterion(sr, hr)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                sr_clipped = torch.clamp(sr, 0, 1)
                batch_psnr = psnr(sr_clipped, hr)

            epoch_loss += loss.item()
            epoch_psnr += batch_psnr
            num_batches += 1

        scheduler.step()

        avg_loss = epoch_loss / num_batches
        avg_psnr = epoch_psnr / num_batches
        loss_history.append(avg_loss)
        psnr_history.append(avg_psnr)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.6f} | PSNR: {avg_psnr:.2f} dB")

    return model, loss_history, psnr_history'''

EVALUATE_ZSSR_PERFECT = '''def evaluate_zssr(
    model,
    image_hr,
    scale_factor=2,
    device='cpu',
    title="ZSSR Evaluation"
):
    """
    Evaluate ZSSR model on a single image.
    """
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        # Create LR from HR
        image_lr = resize_bicubic(image_hr, scale_factor=1.0/scale_factor)

        # Bicubic baseline: upscale LR back to HR size
        image_bicubic = resize_bicubic(image_lr, size=(image_hr.shape[-2], image_hr.shape[-1]))

        # ZSSR prediction
        image_lr_gpu = image_lr.unsqueeze(0).to(device)
        image_sr = model(image_lr_gpu, scale_factor=scale_factor,
                         target_size=(image_hr.shape[-2], image_hr.shape[-1]))
        image_sr = image_sr.squeeze(0)
        image_sr = torch.clamp(image_sr, 0, 1)

        # Clamp bicubic too
        image_bicubic = torch.clamp(image_bicubic, 0, 1)

        # Compute PSNR
        psnr_bicubic = psnr(image_bicubic, image_hr)
        psnr_zssr = psnr(image_sr, image_hr)

        results = {
            \'HR\': image_hr.cpu(),
            \'LR\': image_lr.cpu(),
            \'Bicubic\': image_bicubic.cpu(),
            \'ZSSR\': image_sr.cpu(),
            \'PSNR_Bicubic\': psnr_bicubic,
            \'PSNR_ZSSR\': psnr_zssr,
        }

    return results'''

ENSEMBLE_PREDICT_PERFECT = '''def ensemble_predict_zssr(model, image_lr, scale_factor=2, device='cpu'):
    """
    Predict using geometric self-ensemble: average predictions from 8 augmented versions.
    """
    model.eval()
    device = next(model.parameters()).device

    # Generate 8 augmented versions: 4 rotations x 2 flips
    augmented_images = []
    for k in range(4):
        img_rot = torch.rot90(image_lr, k=k, dims=[-2, -1])
        augmented_images.append(img_rot)
        augmented_images.append(torch.flip(img_rot, dims=[-1]))

    # Predict on all variants
    predictions = []
    with torch.no_grad():
        for aug_img in augmented_images:
            aug_input = aug_img.unsqueeze(0).to(device)
            sr = model(aug_input, scale_factor=scale_factor)
            sr = sr.squeeze(0)
            sr = torch.clamp(sr, 0, 1)

            # Reverse augmentation
            aug_idx = len(predictions)
            if aug_idx % 2 == 1:  # Flip was applied
                sr = torch.flip(sr, dims=[-1])

            k_val = (aug_idx // 2) % 4
            if k_val > 0:
                sr = torch.rot90(sr, k=4-k_val, dims=[-2, -1])

            predictions.append(sr.cpu())

    # Average predictions
    ensemble_sr = torch.stack(predictions).mean(dim=0)
    return ensemble_sr'''

def generate_submissions():
    """Generate 4 variants of student ZSSR submissions."""

    template_path = Path(__file__).resolve().parent.parent / 'applied.ipynb'
    if not template_path.exists():
        # Fallback: look in hw/hw3
        template_path = Path(__file__).resolve().parent.parent.parent.parent / 'hw' / 'hw3' / 'applied.ipynb'
    output_dir = Path(__file__).resolve().parent.parent / 'mock_submissions_applied'
    output_dir.mkdir(exist_ok=True)

    # Read template notebook
    with open(template_path, 'r') as f:
        template = json.load(f)

    # Define variants (cell indices: 7=ZSSRNet, 9=Dataset, 11=train, 13=evaluate, 15=ensemble)
    variants = {
        'student_perfect': {
            7: ZSSR_NET_PERFECT,
            9: ZSSR_DATASET_PERFECT,
            11: TRAIN_ZSSR_PERFECT,
            13: EVALUATE_ZSSR_PERFECT,
            15: ENSEMBLE_PREDICT_PERFECT,
        },
        'student_no_residual': {
            7: ZSSR_NET_NO_RESIDUAL,
            9: ZSSR_DATASET_PERFECT,
            11: TRAIN_ZSSR_PERFECT,
            13: EVALUATE_ZSSR_PERFECT,
            15: 'def ensemble_predict_zssr(model, image_lr, scale_factor=2, device="cpu"):\n    raise NotImplementedError("Bonus not implemented")',
        },
        'student_bad_dataset': {
            7: ZSSR_NET_PERFECT,
            9: ZSSR_DATASET_BAD,
            11: TRAIN_ZSSR_PERFECT,
            13: EVALUATE_ZSSR_PERFECT,
            15: 'def ensemble_predict_zssr(model, image_lr, scale_factor=2, device="cpu"):\n    raise NotImplementedError("Bonus not implemented")',
        },
        'student_minimal': {
            7: ZSSR_NET_PERFECT,
            9: 'class ZSSRDataset(Dataset):\n    def __init__(self, *args, **kwargs):\n        raise NotImplementedError()\n    def __len__(self):\n        raise NotImplementedError()\n    def __getitem__(self, idx):\n        raise NotImplementedError()',
            11: 'def train_zssr(*args, **kwargs):\n    raise NotImplementedError()',
            13: 'def evaluate_zssr(*args, **kwargs):\n    raise NotImplementedError()',
            15: 'def ensemble_predict_zssr(*args, **kwargs):\n    raise NotImplementedError()',
        },
    }

    for variant_name, changes in variants.items():
        nb = deepcopy(template)

        for cell_idx, code in changes.items():
            nb['cells'][cell_idx]['source'] = [code]

        output_path = output_dir / f'{variant_name}.ipynb'
        with open(output_path, 'w') as f:
            json.dump(nb, f, indent=1)

        print(f'Generated {variant_name}.ipynb')

    print(f'Created {output_dir}')

if __name__ == '__main__':
    generate_submissions()
