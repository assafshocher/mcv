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
    def __init__(self, n_channels=64, n_layers=8, kernel_size=3):
        super(ZSSRNet, self).__init__()

        self.n_channels = n_channels
        self.n_layers = n_layers
        self.kernel_size = kernel_size

        # Build the network
        self.layers = nn.ModuleList()
        padding = kernel_size // 2

        # First layer: 1 channel -> n_channels
        self.layers.append(
            nn.Conv2d(1, n_channels, kernel_size, padding=padding)
        )

        # Intermediate layers: n_channels -> n_channels
        for _ in range(n_layers - 2):
            self.layers.append(
                nn.Conv2d(n_channels, n_channels, kernel_size, padding=padding)
            )

        # Last layer: n_channels -> 1 channel
        self.layers.append(
            nn.Conv2d(n_channels, 1, kernel_size, padding=padding)
        )

        # Activation
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, scale_factor=2, target_size=None):
        # Handle input shape
        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        batch_size, channels, height, width = x.shape

        # Bicubic upsampling
        if target_size is None:
            target_size = (height * scale_factor, width * scale_factor)

        upscaled = resize_bicubic(x, size=target_size)

        # Residual network
        residual = upscaled.clone()

        # Pass through network
        out = upscaled
        for i, layer in enumerate(self.layers):
            out = layer(out)
            # Apply ReLU to all layers except the last
            if i < len(self.layers) - 1:
                out = self.relu(out)

        # Add residual connection
        out = out + residual

        return out.squeeze(0)'''

ZSSR_NET_NO_RESIDUAL = '''class ZSSRNet(nn.Module):
    """
    Zero-Shot Super Resolution Network without residual connection (BUG).
    """
    def __init__(self, n_channels=64, n_layers=8, kernel_size=3):
        super(ZSSRNet, self).__init__()

        self.n_channels = n_channels
        self.n_layers = n_layers
        self.kernel_size = kernel_size

        # Build the network
        self.layers = nn.ModuleList()
        padding = kernel_size // 2

        # First layer: 1 channel -> n_channels
        self.layers.append(
            nn.Conv2d(1, n_channels, kernel_size, padding=padding)
        )

        # Intermediate layers: n_channels -> n_channels
        for _ in range(n_layers - 2):
            self.layers.append(
                nn.Conv2d(n_channels, n_channels, kernel_size, padding=padding)
            )

        # Last layer: n_channels -> 1 channel
        self.layers.append(
            nn.Conv2d(n_channels, 1, kernel_size, padding=padding)
        )

        # Activation
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, scale_factor=2, target_size=None):
        # Handle input shape
        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        batch_size, channels, height, width = x.shape

        # Bicubic upsampling
        if target_size is None:
            target_size = (height * scale_factor, width * scale_factor)

        upscaled = resize_bicubic(x, size=target_size)

        # BUG: Missing residual connection
        # residual = upscaled.clone()

        # Pass through network
        out = upscaled
        for i, layer in enumerate(self.layers):
            out = layer(out)
            # Apply ReLU to all layers except the last
            if i < len(self.layers) - 1:
                out = self.relu(out)

        # BUG: No residual connection added
        # out = out + residual

        return out.squeeze(0)'''

ZSSR_DATASET_PERFECT = '''class ZSSRDataset(Dataset):
    """
    Dataset for Zero-Shot Super Resolution on a single image.
    """
    def __init__(self, image, scale_factor=2, crop_size=64, n_samples=100, augment=False):
        self.image = image  # Shape: (H, W)
        self.scale_factor = scale_factor
        self.crop_size = crop_size
        self.n_samples = n_samples
        self.augment = augment

        # Downscale to create LR using resize_bicubic (uses resize_right for correct alignment)
        self.lr_image = resize_bicubic(image, scale_factor=1.0/scale_factor)

        self.hr_image = image

    def __len__(self):
        return self.n_samples

    def _augment_8fold(self, img):
        """Generate 8 augmented versions: 4 rotations x 2 flips"""
        augmented = []
        for k in range(4):
            img_rot = torch.rot90(img, k=k)
            augmented.append(img_rot)
            augmented.append(torch.flip(img_rot, dims=[1]))
        return augmented

    def __getitem__(self, idx):
        # Random crop from HR image
        h, w = self.hr_image.shape
        crop_h, crop_w = min(self.crop_size, h), min(self.crop_size, w)

        top = np.random.randint(0, max(1, h - crop_h + 1))
        left = np.random.randint(0, max(1, w - crop_w + 1))

        hr_crop = self.hr_image[top:top+crop_h, left:left+crop_w]

        # Corresponding LR crop (accounting for scale factor)
        lr_top = top // self.scale_factor
        lr_left = left // self.scale_factor
        lr_crop_h = crop_h // self.scale_factor
        lr_crop_w = crop_w // self.scale_factor

        lr_crop = self.lr_image[
            lr_top:lr_top+lr_crop_h,
            lr_left:lr_left+lr_crop_w
        ]

        if self.augment:
            # Apply 8-fold augmentation
            hr_crops = self._augment_8fold(hr_crop)
            lr_crops = self._augment_8fold(lr_crop)
            return {
                'HR': torch.stack(hr_crops),  # (8, H, W)
                'LR': torch.stack(lr_crops),  # (8, H, W)
            }
        else:
            return {
                'HR': hr_crop.unsqueeze(0),  # (1, H, W)
                'LR': lr_crop.unsqueeze(0),  # (1, H, W)
            }'''

ZSSR_DATASET_BAD = '''class ZSSRDataset(Dataset):
    """
    Dataset for Zero-Shot Super Resolution on a single image (BUG: Wrong downsampling).
    """
    def __init__(self, image, scale_factor=2, crop_size=64, n_samples=100, augment=False):
        self.image = image  # Shape: (H, W)
        self.scale_factor = scale_factor
        self.crop_size = crop_size
        self.n_samples = n_samples
        self.augment = augment

        # BUG: Not properly downscaling — hard-coded division by 2
        self.lr_image = resize_bicubic(image, scale_factor=0.5)

        self.hr_image = image

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Random crop from HR image
        h, w = self.hr_image.shape
        crop_h, crop_w = min(self.crop_size, h), min(self.crop_size, w)

        top = np.random.randint(0, max(1, h - crop_h + 1))
        left = np.random.randint(0, max(1, w - crop_w + 1))

        hr_crop = self.hr_image[top:top+crop_h, left:left+crop_w]

        # BUG: Wrong indexing into LR image
        # This doesn't properly map HR crop to LR crop
        lr_crop = self.lr_image[top:top+crop_h, left:left+crop_w]

        return {
            'HR': hr_crop.unsqueeze(0),  # (1, H, W)
            'LR': lr_crop.unsqueeze(0),  # (1, H, W) - wrong shape!
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

    # Loss and optimizer
    criterion = nn.L1Loss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.5)

    loss_history = []
    psnr_history = []

    model.train()
    total_iterations = 0

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_psnr = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            total_iterations += 1

            hr = batch['HR'].to(device)
            lr = batch['LR'].to(device)

            # Ensure 4D: (B, 1, H, W)
            if hr.dim() == 3:
                hr = hr.unsqueeze(1)
            if lr.dim() == 3:
                lr = lr.unsqueeze(1)

            # If augmented (B, 8, H, W), merge aug into batch
            if hr.shape[1] > 1:
                B, aug, h, w = hr.shape
                hr = hr.view(B * aug, 1, h, w)
                lr = lr.view(B * aug, 1, lr.shape[-2], lr.shape[-1])

            # Forward pass
            optimizer.zero_grad()
            target_h, target_w = hr.shape[-2], hr.shape[-1]
            sr = model(lr, scale_factor=scale_factor, target_size=(target_h, target_w))

            # Ensure 4D output
            if sr.dim() == 2:
                sr = sr.unsqueeze(0).unsqueeze(0)
            elif sr.dim() == 3:
                sr = sr.unsqueeze(1)

            # Ensure shapes match
            if sr.shape != hr.shape:
                sr = resize_bicubic(sr, size=(target_h, target_w))

            # Compute loss
            loss = criterion(sr, hr)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Compute PSNR
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
            print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.6f} | PSNR: {avg_psnr:.3f} dB")

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
        # Create LR from HR using resize_bicubic (resize_right for correct alignment)
        image_lr = resize_bicubic(image_hr, scale_factor=1.0/scale_factor)

        # Bicubic baseline
        image_bicubic = resize_bicubic(image_lr, scale_factor=scale_factor)

        # ZSSR prediction
        image_lr_gpu = image_lr.unsqueeze(0).unsqueeze(0).to(device)
        image_sr = model(image_lr_gpu, scale_factor=scale_factor)
        # Squeeze to 2D (H, W) to match image_hr
        while image_sr.dim() > 2:
            image_sr = image_sr.squeeze(0)
        image_sr = torch.clamp(image_sr, 0, 1)

        # Ensure bicubic matches HR size
        if image_bicubic.shape != image_hr.shape:
            image_bicubic = resize_bicubic(image_bicubic, size=(image_hr.shape[0], image_hr.shape[1]))

        # Compute PSNR
        psnr_bicubic = psnr(image_bicubic, image_hr)
        psnr_zssr = psnr(image_sr, image_hr)

        results = {
            'HR': image_hr.cpu().numpy(),
            'LR': image_lr.cpu().numpy(),
            'Bicubic': image_bicubic.cpu().numpy(),
            'ZSSR': image_sr.cpu().numpy(),
            'PSNR_Bicubic': psnr_bicubic,
            'PSNR_ZSSR': psnr_zssr,
        }

    return results'''

ENSEMBLE_PREDICT_PERFECT = '''def ensemble_predict_zssr(model, image_lr, scale_factor=2, device='cpu'):
    """
    Predict using geometric self-ensemble: average predictions from 8 augmented versions.
    """
    model.eval()
    device = next(model.parameters()).device

    # Generate 8 augmented versions
    augmented_images = []
    for k in range(4):
        img_rot = torch.rot90(image_lr, k=k)
        augmented_images.append(img_rot)
        augmented_images.append(torch.flip(img_rot, dims=[1]))

    # Predict on all variants
    predictions = []
    with torch.no_grad():
        for aug_img in augmented_images:
            aug_img_gpu = aug_img.unsqueeze(0).unsqueeze(0).to(device)
            sr = model(aug_img_gpu, scale_factor=scale_factor)
            while sr.dim() > 2:
                sr = sr.squeeze(0)
            sr = torch.clamp(sr, 0, 1)

            # Reverse augmentation
            aug_idx = len(predictions)
            if aug_idx % 2 == 1:  # Flip was applied
                sr = torch.flip(sr, dims=[-1])

            k_val = (aug_idx // 2) % 4
            for _ in range(k_val):
                sr = torch.rot90(sr, k=3)  # Rotate back

            predictions.append(sr.cpu())

    # Average predictions
    ensemble_sr = torch.stack(predictions).mean(dim=0)
    return ensemble_sr'''

def generate_submissions():
    """Generate 4 variants of student ZSSR submissions."""

    template_path = Path('/sessions/kind-amazing-carson/hw3_applied.ipynb')
    output_dir = Path('/sessions/kind-amazing-carson/mock_submissions_applied')
    output_dir.mkdir(exist_ok=True)

    # Read template notebook
    with open(template_path, 'r') as f:
        template = json.load(f)

    # Define variants
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

    # Generate each variant
    for variant_name, changes in variants.items():
        nb = deepcopy(template)

        # Apply changes
        for cell_idx, code in changes.items():
            nb['cells'][cell_idx]['source'] = [code]

        # Write variant notebook
        output_path = output_dir / f'{variant_name}.ipynb'
        with open(output_path, 'w') as f:
            json.dump(nb, f, indent=1)

        print(f'Generated {variant_name}.ipynb')

    print(f'Created {output_dir}')

if __name__ == '__main__':
    generate_submissions()
