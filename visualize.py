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

def format_title(kx_name_true):
    return " ".join([word.capitalize() if word != "rbf" else "RBF" for word in kx_name_true.split("_")])

def viz_errors(ax, kx_name_true, kernel_to_full_errors, show_xlabel=False, show_ylabel=False):
    ax.set_title(format_title(kx_name_true), fontsize=16, fontweight='bold')

    if show_xlabel:
        ax.set_xlabel("Layers", fontsize=14)
    if show_ylabel:
        ax.set_ylabel(r"$\| u - \mathcal{O}f \|^2$", fontsize=14)

    colors = sns.color_palette("colorblind", n_colors=4)

    for idx, kernel_name in enumerate(kernel_to_full_errors):
        mean = np.mean(kernel_to_full_errors[kernel_name], axis=0)
        std  = np.std(kernel_to_full_errors[kernel_name], axis=0)
        
        x     = np.arange(mean.shape[0])
        lower = mean - std
        upper = mean + std

        ax.plot(x, mean, label=kernel_name, linewidth=2.5, color=colors[idx])
        ax.fill_between(x, lower, upper, alpha=0.3)

    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    ax.legend(fontsize=10, loc='upper right', frameon=True)
    ax.tick_params(axis='both', which='major', labelsize=12)

def main():
    full_errors = []

    i = 1
    while True:
        result_fn = os.path.join("results", f"errors_trial={i}.pkl")
        if not os.path.exists(result_fn):
            break
        with open(result_fn, "rb") as f:
            full_errors.append(pickle.load(f))
        i += 1

    kx_names = list(full_errors[0].keys())
    kx_name_to_full_errors = {}
    for kx_name_true in kx_names:
        kx_name_to_full_errors[kx_name_true] = {}
        for kernel_name in kx_names:
            kx_name_to_full_errors[kx_name_true][kernel_name] = np.array([full_error[kx_name_true][kernel_name] for full_error in full_errors])

    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    for i, kx_name in enumerate(kx_names):
        row, col = divmod(i, 2)
        show_ylabel = (col == 0)  # Only left column
        show_xlabel = (row == 1)  # Only bottom row
        viz_errors(axs[row, col], kx_name, kx_name_to_full_errors[kx_name], show_xlabel, show_ylabel)

    os.makedirs("results", exist_ok=True)
    plt.savefig(os.path.join("results", "blup_final.png"), dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    main()