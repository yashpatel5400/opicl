import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftn, ifftn
from scipy.signal import correlate2d

from opformer import TransformerOperator
from kernels import Kernels

device = "cuda" if torch.cuda.is_available() else "cpu"

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

def make_simple_learnable_operator_dataset(kernel, num_samples=25, H=64, W=64, sigma=1.0, seed=42):
    # Sample 5 random basis pairs (f_i, g_i)
    np.random.seed(seed)

    alpha, beta, gamma, N = 1.0, 1.0, 4.0, H
    num_bases = 5

    basis_fs = GRF(alpha, beta, gamma, N, num_bases)
    basis_gs = GRF(alpha, beta, gamma, N, num_bases)

    # Precompute convolutions (k * g_i)
    g_convs = np.array([np_conv(kernel, g) for g in basis_gs])

    # Sample new input functions
    fs = GRF(alpha, beta, gamma, N, num_samples)

    # Compute operator outputs:
    # O(f) = sum_{i=1}^5 <f, f_i> * (k * g_i)
    Ofs = np.zeros((num_samples, H, W))
    for j in range(num_samples):
        for i in range(num_bases):
            inner = np.sum(fs[j] * basis_fs[i])
            Ofs[j] += inner * g_convs[i]

    return fs, Ofs

def construct_Z(f, Of, f_test, im_size):
    f_full = np.concatenate([f[:-1], np.expand_dims(f_test, axis=0)], axis=0)
    Z = np.expand_dims(np.concatenate([f_full, Of], axis=1), axis=0)
    Z_pt = torch.from_numpy(Z).to(device).to(torch.float32)
    Z_pt[:,-1, im_size[0]:] = 0
    return Z_pt

def viz_errors(true_kernel_name, kernel_to_errors):
    plt.clf()
    plt.title(f"True Kernel: {true_kernel_name}")
    plt.xlabel("Layers")
    plt.ylabel("|| u - Of ||^2")

    for kernel_name in kernel_to_errors:
        plt.semilogy(kernel_to_errors[kernel_name], label=kernel_name)
        
    plt.legend()
    plt.savefig(os.path.join("results", f"{true_kernel_name}_errors.png"))

def viz_fields(true_kernel_name, truth, kernel_to_preds):
    # Plot comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0,0].imshow(truth, cmap='viridis')
    axes[0,0].set_title("Ground Truth: O f_test")
    axes[0,0].axis('off')

    for i, kernel_name in enumerate(kernel_to_preds):
        row, col = divmod(i+1, 3)

        axes[row, col].imshow(-kernel_to_preds[kernel_name][-1], cmap='viridis')
        axes[row, col].set_title(f"{kernel_name} Prediction")
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join("results", f"{true_kernel_name}_fields.png"))

def main(true_kernel_name):
    im_size = (64, 64)
    kernel_maps = Kernels(im_size[0], im_size[1])

    kernels = {
        "gaussian"    : kernel_maps.get_kernel("gaussian", sigma=5.0),
        "laplace"     : kernel_maps.get_kernel("laplace", sigma=1.0),
        "exponential" : kernel_maps.get_kernel("exponential", sigma=10.0),
        "mexican_hat" : kernel_maps.get_kernel("mexican_hat", sigma=3.0),
        "cauchy"      : kernel_maps.get_kernel("cauchy", sigma=8.0),
    }

    # Generate data
    icl_num_samples = 50
    f, Of = make_simple_learnable_operator_dataset(
        kernels[true_kernel_name], 
        num_samples=icl_num_samples
    )

    r = 2 # 5e-1
    num_layers = 250

    kernel_to_preds, kernel_to_errors = {}, {}
    for kernel_name in kernels:
        opformer = TransformerOperator(num_layers=num_layers, im_size=im_size, kernel=kernels[kernel_name], icl_lr=-r, icl_init=True).to(device)

        y_idx  = -1
        Z_test = construct_Z(f, Of, f[y_idx], im_size)

        _, preds = opformer(Z_test)
        test_preds = np.array([pred[0,-1,64:,:,0] for pred in preds]) # just extract bottom right for test prediction
        errors = np.array([np.linalg.norm(test_pred + Of[y_idx]) for test_pred in test_preds])

        kernel_to_preds[kernel_name]  = test_preds
        kernel_to_errors[kernel_name] = errors

    os.makedirs("results", exist_ok=True)
    viz_errors(true_kernel_name, kernel_to_errors)
    viz_fields(true_kernel_name, Of[y_idx], kernel_to_preds)

if __name__ == "__main__":
    for true_kernel_name in ["gaussian", "laplace", "exponential", "mexican_hat", "cauchy"]:
        print(f"Running {true_kernel_name} experiment...")
        main(true_kernel_name)