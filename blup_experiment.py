import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftn, ifftn
from scipy.signal import correlate2d

from opformer import TransformerOperator
import kernels

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


def make_simple_learnable_operator_dataset(kx, ky, num_samples=25, H=64, W=64, sigma=1.0, seed=42):
    # Sample 5 random basis pairs (f_i, g_i)
    np.random.seed(seed)

    alpha, beta, gamma, N = 1.0, 1.0, 4.0, H
    num_bases = 1

    basis_fs = GRF(alpha, beta, gamma, N, num_bases)
    basis_gs = GRF(alpha, beta, gamma, N, num_bases)

    # Precompute convolutions (k * g_i)
    g_convs = np.array([np_conv(ky, g) for g in basis_gs])

    # Sample new input functions
    fs = GRF(alpha, beta, gamma, N, num_samples)

    # Compute operator outputs:
    # O(f) = sum_{i=1}^5 <f, f_i> * (k * g_i)
    Ofs = np.zeros((num_samples, H, W))
    for j in range(num_samples):
        for i in range(num_bases):
            inner = kx(fs[j], basis_fs[i])
            Ofs[j] += inner * g_convs[i]

    return fs, Ofs

def construct_Z(f_test, Of, f, im_size, device="cuda"):
    f_full = np.concatenate([f[:-1], np.expand_dims(f_test, axis=0)], axis=0)
    Z = np.expand_dims(np.concatenate([f_full, Of], axis=1), axis=0)
    Z_pt = torch.from_numpy(Z).to(device).to(torch.float32)
    Z_pt[:,-1,im_size[0]:] = 0
    return Z_pt

def viz_errors(kx_name_true, kernel_to_errors):
    plt.clf()
    plt.title(f"True Kernel: {kx_name_true}")
    plt.xlabel("Layers")
    plt.ylabel("|| u - Of ||^2")

    for kernel_name in kernel_to_errors:
        if np.isfinite(kernel_to_errors[kernel_name][-1]):
            plt.semilogy(kernel_to_errors[kernel_name], label=kernel_name)
        
    plt.legend()

    os.makedirs("results", exist_ok=True)
    plt.savefig(os.path.join("results", f"{kx_name_true}.png"))

def main(kx_name_true):
    H, W = 64, 64
    kernel_maps = kernels.Kernels(H, W)

    kx_sigma = 1.0
    kx_names = ['linear', 'laplacian', 'gradient_rbf', 'energy']
    kx_true = kernels.get_kx_kernel(kx_name_true, sigma=kx_sigma)
    ky_true = kernel_maps.get_kernel("gaussian")

    num_samples = 25
    f, Of = make_simple_learnable_operator_dataset(
        kx_true, ky_true, num_samples=num_samples
    )

    f_test   = f[-1]
    Of_test  = Of[-1]

    im_size = (64, 64)
    device = "cuda"
    Z_test = construct_Z(f_test, Of, f, im_size, device)

    kernel_to_preds, kernel_to_errors = {}, {}
    for kx_name in kx_names:
        r = .01
        num_layers = 250
        opformer = TransformerOperator(
            num_layers=num_layers, 
            im_size=im_size, 
            ky_kernel=ky_true, 
            kx_name=kx_name, 
            kx_sigma=kx_sigma, 
            icl_lr=-r, 
            icl_init=True).to(device)

        _, preds = opformer(Z_test)
        test_preds = np.array([pred[0,-1,64:,:,0] for pred in preds]) # just extract bottom right for test prediction
        errors = np.array([np.linalg.norm(test_pred + Of_test) for test_pred in test_preds])

        kernel_to_preds[kx_name]  = test_preds
        kernel_to_errors[kx_name] = errors

    viz_errors(kx_name_true, kernel_to_errors)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kx")
    args = parser.parse_args()
    
    main(args.kx)