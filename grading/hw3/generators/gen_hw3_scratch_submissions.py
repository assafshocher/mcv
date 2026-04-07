#!/usr/bin/env python3
"""
Generate mock student submissions for HW3 from-scratch.
Creates 4 variants: perfect, xcorr_bug, no_pool_backward, partial
"""

import json
import shutil
from pathlib import Path
from copy import deepcopy

# Complete correct implementations
CONV2D_FORWARD_CORRECT = '''def conv2d_forward(x, w, b=None, padding=0, stride=1, dilation=1):
    """
    Forward pass of 2D convolution using unfold.

    Args:
        x: input tensor of shape (B, C_in, H, W)
        w: weight tensor of shape (C_out, C_in, K_h, K_w)
        b: optional bias of shape (C_out,)
        padding: padding size (same for all sides)
        stride: stride size (same for H and W)
        dilation: dilation factor for the kernel

    Returns:
        output tensor of shape (B, C_out, H_out, W_out)
    """
    B, C_in, H, W = x.shape
    C_out, _, K_h, K_w = w.shape

    # Unfold the input
    x_unfolded = F.unfold(
        x,
        kernel_size=(K_h, K_w),
        padding=padding,
        stride=stride,
        dilation=dilation
    )  # Shape: (B, C_in*K_h*K_w, L)

    # Reshape weights for matrix multiplication
    w_reshaped = w.view(C_out, -1)  # Shape: (C_out, C_in*K_h*K_w)

    # Perform convolution as matrix multiplication
    # (C_out, C_in*K_h*K_w) @ (B, C_in*K_h*K_w, L) = (B, C_out, L)
    output = w_reshaped @ x_unfolded  # Shape: (B, C_out, L)

    # Add bias if provided
    if b is not None:
        output = output + b.view(1, -1, 1)

    # Compute output spatial dimensions
    H_out = (H + 2 * padding - dilation * (K_h - 1) - 1) // stride + 1
    W_out = (W + 2 * padding - dilation * (K_w - 1) - 1) // stride + 1

    # Reshape to (B, C_out, H_out, W_out)
    output = output.view(B, C_out, H_out, W_out)

    return output'''

CONV2D_FORWARD_XCORR_BUG = '''def conv2d_forward(x, w, b=None, padding=0, stride=1, dilation=1):
    """
    Forward pass of 2D convolution using unfold.

    Args:
        x: input tensor of shape (B, C_in, H, W)
        w: weight tensor of shape (C_out, C_in, K_h, K_w)
        b: optional bias of shape (C_out,)
        padding: padding size (same for all sides)
        stride: stride size (same for H and W)
        dilation: dilation factor for the kernel

    Returns:
        output tensor of shape (B, C_out, H_out, W_out)
    """
    B, C_in, H, W = x.shape
    C_out, _, K_h, K_w = w.shape

    # Unfold the input
    x_unfolded = F.unfold(
        x,
        kernel_size=(K_h, K_w),
        padding=padding,
        stride=stride,
        dilation=dilation
    )  # Shape: (B, C_in*K_h*K_w, L)

    # Reshape weights for matrix multiplication
    w_reshaped = w.view(C_out, -1)  # Shape: (C_out, C_in*K_h*K_w)

    # BUG: Wrong output size calculation (off by one)
    output = w_reshaped @ x_unfolded  # Shape: (B, C_out, L)

    # Add bias if provided
    if b is not None:
        output = output + b.view(1, -1, 1)

    # BUG: Incorrect formula - missing the +1
    H_out = (H + 2 * padding - dilation * (K_h - 1)) // stride
    W_out = (W + 2 * padding - dilation * (K_w - 1)) // stride

    # Reshape to (B, C_out, H_out, W_out)
    output = output.view(B, C_out, H_out, W_out)

    return output'''

CONV2D_BACKWARD_CORRECT = '''def conv2d_backward(dy, x, w, b, padding, stride, dilation):
    """
    Backward pass for 2D convolution.

    Args:
        dy: gradient w.r.t. output of shape (B, C_out, H_out, W_out)
        x: input tensor of shape (B, C_in, H, W)
        w: weight tensor of shape (C_out, C_in, K_h, K_w)
        b: bias tensor of shape (C_out,) or None
        padding, stride, dilation: same as forward pass

    Returns:
        dx, dw, db: gradients w.r.t. x, w, b
    """
    B, C_in, H, W = x.shape
    C_out, _, K_h, K_w = w.shape
    B, C_out, H_out, W_out = dy.shape

    # Flatten dy for computation
    dy_flat = dy.view(B, C_out, -1)  # (B, C_out, H_out*W_out)

    # Unfold x
    x_unfolded = F.unfold(
        x,
        kernel_size=(K_h, K_w),
        padding=padding,
        stride=stride,
        dilation=dilation
    )  # (B, C_in*K_h*K_w, H_out*W_out)

    # Compute dw: (C_out, C_in*K_h*K_w) from (B, C_out, H_out*W_out) @ (B, C_in*K_h*K_w, H_out*W_out).T
    w_reshaped = w.view(C_out, -1)
    dw = (dy_flat @ x_unfolded.transpose(1, 2)).sum(0)  # Sum over batch
    dw = dw.view(w.shape)

    # Compute db: sum of dy over spatial dimensions
    if b is not None:
        db = dy_flat.sum(dim=(0, 2))  # Sum over batch and spatial
    else:
        db = None

    # Compute dx: use col2im/fold
    w_t = w.view(C_out, -1).t()  # (C_in*K_h*K_w, C_out)
    dcol = w_t @ dy_flat  # (B, C_in*K_h*K_w, H_out*W_out)

    dx = F.fold(
        dcol,
        output_size=(H, W),
        kernel_size=(K_h, K_w),
        padding=padding,
        stride=stride,
        dilation=dilation
    )  # (B, C_in, H, W)

    return dx, dw, db'''

MAX_POOL2D_FORWARD_CORRECT = '''def max_pool2d_forward(x, kernel_size, padding=0, stride=None):
    """
    Forward pass of 2D max pooling.

    Args:
        x: input tensor of shape (B, C, H, W)
        kernel_size: size of pooling kernel
        padding: padding size
        stride: stride size (defaults to kernel_size if None)

    Returns:
        output: pooled output tensor
        argmax_indices: indices of max values (for backward pass)
    """
    if stride is None:
        stride = kernel_size

    output = F.max_pool2d(x, kernel_size=kernel_size, padding=padding, stride=stride)

    # For backward, we need to track max indices
    # Store metadata for backward pass
    return output, None'''

MAX_POOL2D_BACKWARD_CORRECT = '''def max_pool2d_backward(dy, x, kernel_size, padding=0, stride=None, argmax_indices=None):
    """
    Backward pass for 2D max pooling.

    Args:
        dy: gradient w.r.t. output
        x: original input tensor
        kernel_size: size of pooling kernel
        padding: padding size
        stride: stride size
        argmax_indices: (unused) for compatibility

    Returns:
        dx: gradient w.r.t. input
    """
    if stride is None:
        stride = kernel_size

    # Use PyTorch's max_pool2d with return_indices to compute backward
    x_expand = x.unsqueeze(1)
    _, indices = F.max_pool2d_with_indices(x.unsqueeze(1), kernel_size=kernel_size, padding=padding, stride=stride)

    # Initialize dx
    B, C, H, W = x.shape
    dx = torch.zeros_like(x)

    # Distribute gradients to max positions
    H_out = (H + 2 * padding - (kernel_size - 1) - 1) // stride + 1
    W_out = (W + 2 * padding - (kernel_size - 1) - 1) // stride + 1

    for b in range(B):
        for c in range(C):
            for h_out in range(H_out):
                for w_out in range(W_out):
                    max_idx = indices[b, c, h_out, w_out].item()
                    h_in = (h_out * stride + max_idx // kernel_size) - padding
                    w_in = (h_out * stride + max_idx % kernel_size) - padding
                    if 0 <= h_in < H and 0 <= w_in < W:
                        dx[b, c, h_in, w_in] += dy[b, c, h_out, w_out]

    return dx'''

MAX_POOL2D_BACKWARD_NO_POOL = '''def max_pool2d_backward(dy, x, kernel_size, padding=0, stride=None, argmax_indices=None):
    """
    Backward pass for 2D max pooling.
    BUG: Scatters gradient everywhere instead of only to max positions.
    """
    if stride is None:
        stride = kernel_size

    # BUG: This just returns the gradient expanded everywhere
    B, C, H, W = x.shape
    H_out = (H + 2 * padding - (kernel_size - 1) - 1) // stride + 1
    W_out = (W + 2 * padding - (kernel_size - 1) - 1) // stride + 1

    # Expand dy to match input shape (WRONG - should only scatter to max indices)
    dx = torch.zeros(B, C, H, W, device=dy.device, dtype=dy.dtype)

    # BUG: Just repeat the gradient everywhere in the receptive field
    for h_out in range(H_out):
        for w_out in range(W_out):
            h_start = h_out * stride - padding
            h_end = h_start + kernel_size
            w_start = w_out * stride - padding
            w_end = w_start + kernel_size

            # Just add gradient to all positions in receptive field (WRONG!)
            h_start_clamped = max(0, h_start)
            h_end_clamped = min(H, h_end)
            w_start_clamped = max(0, w_start)
            w_end_clamped = min(W, w_end)

            dx[:, :, h_start_clamped:h_end_clamped, w_start_clamped:w_end_clamped] += dy[:, :, h_out:h_out+1, w_out:w_out+1]

    return dx'''

MOMENTUM_SGD_CORRECT = '''class MomentumSGD:
    """
    SGD with momentum optimizer.
    """
    def __init__(self, params, lr=0.01, momentum=0.9):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.velocity = [torch.zeros_like(p) for p in self.params]

    def step(self):
        for param, vel in zip(self.params, self.velocity):
            if param.grad is not None:
                vel.mul_(self.momentum).add_(param.grad, alpha=self.lr)
                param.data.sub_(vel)

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()'''

CONVNET_CORRECT = '''class ConvNet(Module):
    """
    Simple convolutional network for CIFAR-10.
    """
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = Linear(64 * 8 * 8, 128)
        self.fc2 = Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = relu_forward(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = relu_forward(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = relu_forward(x)
        x = self.fc2(x)
        return x'''

def generate_submissions():
    """Generate 4 variants of student submissions."""

    template_path = Path('/sessions/kind-amazing-carson/hw3_from_scratch.ipynb')
    output_dir = Path('/sessions/kind-amazing-carson/mock_submissions_scratch')
    output_dir.mkdir(exist_ok=True)

    # Read template notebook
    with open(template_path, 'r') as f:
        template = json.load(f)

    # Define what changes to make for each variant
    changes = {
        'student_perfect': {
            'conv2d_forward': CONV2D_FORWARD_CORRECT,
            'conv2d_backward': CONV2D_BACKWARD_CORRECT,
            'max_pool2d_forward': MAX_POOL2D_FORWARD_CORRECT,
            'max_pool2d_backward': MAX_POOL2D_BACKWARD_CORRECT,
            'MomentumSGD': MOMENTUM_SGD_CORRECT,
            'ConvNet': CONVNET_CORRECT,
        },
        'student_xcorr': {
            'conv2d_forward': CONV2D_FORWARD_XCORR_BUG,
            'conv2d_backward': CONV2D_BACKWARD_CORRECT,
            'max_pool2d_forward': MAX_POOL2D_FORWARD_CORRECT,
            'max_pool2d_backward': MAX_POOL2D_BACKWARD_CORRECT,
            'MomentumSGD': MOMENTUM_SGD_CORRECT,
            'ConvNet': CONVNET_CORRECT,
        },
        'student_no_pool_backward': {
            'conv2d_forward': CONV2D_FORWARD_CORRECT,
            'conv2d_backward': CONV2D_BACKWARD_CORRECT,
            'max_pool2d_forward': MAX_POOL2D_FORWARD_CORRECT,
            'max_pool2d_backward': MAX_POOL2D_BACKWARD_NO_POOL,
            'MomentumSGD': MOMENTUM_SGD_CORRECT,
            'ConvNet': CONVNET_CORRECT,
        },
        'student_partial': {
            'conv2d_forward': CONV2D_FORWARD_CORRECT,
            'conv2d_backward': 'def conv2d_backward(dy, x, w, b, padding, stride, dilation):\n    raise NotImplementedError("Student did not complete this")',
            'max_pool2d_forward': 'def max_pool2d_forward(x, kernel_size, padding=0, stride=None):\n    raise NotImplementedError("Student did not complete this")',
            'max_pool2d_backward': 'def max_pool2d_backward(dy, x, kernel_size, padding=0, stride=None, argmax_indices=None):\n    raise NotImplementedError("Student did not complete this")',
            'MomentumSGD': 'class MomentumSGD:\n    raise NotImplementedError("Student did not complete this")',
            'ConvNet': 'class ConvNet(Module):\n    raise NotImplementedError("Student did not complete this")',
        }
    }

    # Generate each variant
    for variant_name, change_map in changes.items():
        nb = deepcopy(template)

        # Find and replace code cells by function name
        for cell_idx, cell in enumerate(nb['cells']):
            if cell['cell_type'] != 'code':
                continue

            source = ''.join(cell['source'])

            # Check which function this cell defines
            for func_name, new_code in change_map.items():
                if f'def {func_name}' in source or f'class {func_name}' in source:
                    nb['cells'][cell_idx]['source'] = [new_code]
                    break

        # Write variant notebook
        output_path = output_dir / f'{variant_name}.ipynb'
        with open(output_path, 'w') as f:
            json.dump(nb, f, indent=1)

        print(f'Generated {variant_name}.ipynb')

    print(f'Created {output_dir}')

if __name__ == '__main__':
    generate_submissions()
