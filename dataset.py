import numpy as np
from numpy.fft import fftn, ifftn
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from scipy.signal import correlate2d

def GRF(alpha, beta, gamma, N, num_samples=10):
    xi = np.random.randn(num_samples, N, N)
    K1, K2 = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    freq_sq = (K1**2 + K2**2)
    coef = alpha**0.5 * (4*np.pi**2 * freq_sq + beta)**(-gamma/2)
    L = N * coef * xi
    L[:, 0, 0] = 0  # enforce mean 0
    f_spatials = np.real(ifftn(L, norm='forward', axes=(-2, -1)))
    return f_spatials

def np_conv(kernel, img):
    kh, kw = kernel.shape
    H, W = img.shape

    # Compute asymmetric 'same' padding for even-sized kernels
    ph_top = kh // 2
    ph_bottom = kh - ph_top - 1
    pw_left = kw // 2
    pw_right = kw - pw_left - 1

    # Circular (wrap-around) padding
    padded = np.pad(
        img,
        ((ph_top, ph_bottom), (pw_left, pw_right)),
        mode='wrap'
    )

    # Perform cross-correlation (equivalent to conv2d with no flipping in PyTorch)
    # If you need strict convolution, flip the kernel:
    kernel_flipped = np.flip(kernel, axis=(0, 1))

    # Use 'valid' to get result of original size due to pre-padding
    result = correlate2d(padded, kernel_flipped, mode='valid')

    return result


def make_random_operator_dataset(kx, ky, num_samples=25, num_bases=5, H=64, W=64, seed=42):
    np.random.seed(seed)

    alpha, beta, gamma, N = 1.0, 1.0, 4.0, H

    # Sample basis functions
    basis_fs = GRF(alpha, beta, gamma, N, num_bases)
    basis_gs = GRF(alpha, beta, gamma, N, num_bases)

    # Precompute convolutions (k_y * g_i)
    g_convs = np.array([np_conv(ky, g) for g in basis_gs])

    # Sample random weights λ_i ~ N(0,1)
    lambdas = np.random.randn(num_bases)

    # Sample new input functions
    fs = GRF(alpha, beta, gamma, N, num_samples)

    # Apply operator
    Ofs = np.zeros((num_samples, H, W))
    for j in range(num_samples):
        for i in range(num_bases):
            inner = kx(fs[j], basis_fs[i])
            Ofs[j] += lambdas[i] * inner * g_convs[i]

    return fs, Ofs


def solve_poisson_periodic(f):
    """
    Solve Δu = f on a periodic domain using an FFT-based Poisson solver.
    Sets the mean of u to zero by zeroing the DC component.
    """
    H, W = f.shape
    f_hat = fftn(f)

    k1 = np.fft.fftfreq(H) * 2 * np.pi
    k2 = np.fft.fftfreq(W) * 2 * np.pi
    K1, K2 = np.meshgrid(k1, k2, indexing="ij")
    denom = -(K1 ** 2 + K2 ** 2)

    # Avoid division by zero at the zero frequency; set mean of u to 0
    denom[0, 0] = np.inf
    u_hat = f_hat / denom
    u_hat[0, 0] = 0.0

    u = np.real(ifftn(u_hat))
    return u


def make_heat_dataset(num_samples=25, H=64, W=64, seed=42, alpha=1.0, beta=1.0, gamma=4.0):
    """
    Sample forcing functions f from a GRF and generate solutions u to the
    2D Poisson problem Δu = f on a periodic grid.
    """
    np.random.seed(seed)
    fs = GRF(alpha, beta, gamma, H, num_samples=num_samples)
    us = np.stack([solve_poisson_periodic(f) for f in fs], axis=0)
    return fs, us


def construct_Z(f_test, Of, f, im_size, device="cuda"):
    f_full = np.concatenate([f[:-1], np.expand_dims(f_test, axis=0)], axis=0)
    Z = np.expand_dims(np.concatenate([f_full, Of], axis=1), axis=0)
    Z_pt = torch.from_numpy(Z).to(device).to(torch.float32)
    Z_pt[:,-1,im_size[0]:] = 0
    return Z_pt
