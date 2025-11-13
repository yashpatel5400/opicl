import pickle
import argparse
import os
import seaborn as sns
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftn, ifftn
from scipy.signal import correlate2d

from opformer import TransformerOperator
import kernels
import dataset

def compute_blup_prediction(f, Of, kx_true, ridge=1e-6):
    num_samples = f.shape[0]
    n_train = num_samples - 1

    f_train = f[:n_train]       # f_1..f_n
    f_test  = f[-1]             # f_{n+1}
    u_train = Of[:n_train]      # u_1..u_n
    u_test  = Of[-1]            # u_{n+1} (ground truth)

    # Build K_x on training inputs
    Kx = np.empty((n_train, n_train), dtype=np.float64)
    for i in range(n_train):
        for j in range(n_train):
            Kx[i, j] = kx_true(f_train[i], f_train[j])

    # Cross-covariance vector between test input and training inputs
    k_vec = np.empty(n_train, dtype=np.float64)
    for i in range(n_train):
        k_vec[i] = kx_true(f_test, f_train[i])

    # Solve K_x w^T = k^T  ->  w = K_x^{-1} k
    # (we add a tiny ridge term for numerical stability)
    Kx_reg = Kx + ridge * np.eye(n_train)
    w = np.linalg.solve(Kx_reg, k_vec)   # shape (n_train,)

    # BLUP prediction: u_blup = sum_i w_i * u_i
    # tensordot over training index
    u_blup = np.tensordot(w, u_train, axes=(0, 0))  # (H, W)

    # BLUP error: ||u_blup - u_test||_2
    blup_error = np.linalg.norm(u_blup - u_test)

    return u_blup, blup_error

def main(ax, kx_name_true, show_xlabel=False, show_ylabel=False, seed=0):
    H, W = 64, 64
    kernel_maps = kernels.Kernels(H, W)

    kx_sigma = 1.0
    kx_names = ['linear', 'laplacian', 'gradient_rbf', 'energy']
    kx_true = kernels.get_kx_kernel(kx_name_true, sigma=kx_sigma)
    ky_true = kernel_maps.get_kernel("gaussian")

    num_samples = 25
    f, Of = dataset.make_random_operator_dataset(
        kx_true, ky_true, num_samples=num_samples, num_bases=10, seed=seed,
    )
    
    f_test = f[-1]
    Of_test = Of[-1]
    
    _, blup_error = compute_blup_prediction(f, Of, kx_true)
    
    im_size = (64, 64)
    device = "cuda"
    Z_test = dataset.construct_Z(f_test, Of, f, im_size, device)

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

        _, preds, _ = opformer(Z_test)
        test_preds = np.array([pred[0,-1,64:,:,0] for pred in preds])
        errors = np.array([np.linalg.norm(test_pred + Of_test) for test_pred in test_preds])

        kernel_to_preds[kx_name]  = test_preds
        kernel_to_errors[kx_name] = errors
    kernel_to_errors["blup"] = np.repeat(blup_error, (num_layers,))

    # Save the final prediction and target
    kernel_to_final_preds = {kx_name: preds[-1] for kx_name, preds in kernel_to_preds.items()}
    true_field = Of_test

    return kernel_to_preds, kernel_to_errors, kernel_to_final_preds, true_field

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    kx_names = ['linear', 'laplacian', 'gradient_rbf', 'energy']

    true_kx_to_preds, true_kx_to_errors = {}, {}
    true_kx_to_final_preds = {}
    true_kx_to_targets = {}

    for i, kx_name in enumerate(kx_names):
        row, col = divmod(i, 2)
        show_ylabel = (col == 0)  # Only left column
        show_xlabel = (row == 1)  # Only bottom row

        kernel_to_preds, kernel_to_errors, kernel_to_final_preds, Of_true = main(axs[row, col], kx_name, show_xlabel=show_xlabel, show_ylabel=show_ylabel, seed=args.seed)

        true_kx_to_preds[kx_name]  = kernel_to_preds
        true_kx_to_errors[kx_name] = kernel_to_errors
        true_kx_to_final_preds[kx_name] = kernel_to_final_preds
        true_kx_to_targets[kx_name] = Of_true

    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", f"preds_trial={args.seed}.pkl"), "wb") as f:
        pickle.dump(true_kx_to_preds, f)

    with open(os.path.join("results", f"errors_trial={args.seed}.pkl"), "wb") as f:
        pickle.dump(true_kx_to_errors, f)

    with open(os.path.join("results", f"final_preds_trial={args.seed}.pkl"), "wb") as f:
        pickle.dump(true_kx_to_final_preds, f)

    with open(os.path.join("results", f"targets_trial={args.seed}.pkl"), "wb") as f:
        pickle.dump(true_kx_to_targets, f)
