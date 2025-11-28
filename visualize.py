import pickle
import argparse
import os
import seaborn as sns
import torch
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftn, ifftn
from scipy.signal import correlate2d

def format_title(kx_name_true):
    return " ".join([word.capitalize() if word != "rbf" else "RBF" for word in kx_name_true.split("_")])

def viz_errors(ax, kx_name_true, kernel_to_full_errors, show_xlabel=False, show_ylabel=False, title_override=None):
    ax.set_title(title_override or format_title(kx_name_true), fontsize=16, fontweight='bold')

    if show_xlabel:
        ax.set_xlabel("Layers", fontsize=14)
    if show_ylabel:
        ax.set_ylabel(r"$\| u - \mathcal{O}f \|^2$", fontsize=14)

    colors = sns.color_palette("colorblind", n_colors=5)

    for idx, kernel_name in enumerate(kernel_to_full_errors):
        mean = np.mean(kernel_to_full_errors[kernel_name], axis=0)
        std  = np.std(kernel_to_full_errors[kernel_name], axis=0)
        
        x     = np.arange(mean.shape[0])
        lower = mean - std
        upper = mean + std

        kernel_name_label = kernel_name if kernel_name != "blup" else "BLUP"
        ax.plot(x, mean, label=kernel_name_label, linewidth=2.5, color=colors[idx])
        ax.fill_between(x, lower, upper, alpha=0.3)

    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    ax.legend(fontsize=10, loc='upper right', frameon=True)
    ax.tick_params(axis='both', which='major', labelsize=12)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="random_operator",
        choices=["random_operator", "heat"],
        help="Visualization layout: original four-panel operator plots or single heat-equation plot.",
    )
    parser.add_argument(
        "--trial_idx",
        type=int,
        default=0,
        help="Which trial index to use for field visualizations (0-based).",
    )
    parser.add_argument(
        "--num_fu_pairs",
        type=int,
        default=4,
        help="Number of (f,u) pairs to visualize.",
    )
    args = parser.parse_args()

    full_errors = []

    i = 1
    while True:
        result_fn = os.path.join("results", f"errors_trial={i}.pkl")
        if not os.path.exists(result_fn):
            break
        with open(result_fn, "rb") as f:
            full_errors.append(pickle.load(f))
        i += 1

    if not full_errors:
        raise RuntimeError("No results found in 'results/'. Run blup_experiment.py first.")

    kx_names = list(full_errors[0].keys())

    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    if args.dataset_type == "heat":
        # Use a single panel; curves show all kx choices on the heat-equation data
        reference_true = kx_names[0]
        kernel_names = kx_names + ["blup"]
        kernel_to_full_errors = {
            kernel_name: np.array([full_error[reference_true][kernel_name] for full_error in full_errors])
            for kernel_name in kernel_names
        }

        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        viz_errors(ax, "heat_equation", kernel_to_full_errors, show_xlabel=True, show_ylabel=True, title_override="Heat Equation")
    else:
        kx_name_to_full_errors = {}
        for kx_name_true in kx_names:
            kx_name_to_full_errors[kx_name_true] = {}
            for kernel_name in (kx_names + ["blup"]):
                kx_name_to_full_errors[kx_name_true][kernel_name] = np.array([full_error[kx_name_true][kernel_name] for full_error in full_errors])

        fig, axs = plt.subplots(2, 2, figsize=(14, 12))
        for i, kx_name in enumerate(kx_names):
            row, col = divmod(i, 2)
            show_ylabel = (col == 0)  # Only left column
            show_xlabel = (row == 1)  # Only bottom row
            viz_errors(axs[row, col], kx_name, kx_name_to_full_errors[kx_name], show_xlabel, show_ylabel)

    os.makedirs("results", exist_ok=True)
    plt.savefig(os.path.join("results", "blup_final.png"), dpi=300, bbox_inches='tight')

    # Visualization of predicted fields and true target
    trial_idx = args.trial_idx  # choose which trial to show (0-based)

    with open(os.path.join("results", f"final_preds_trial={trial_idx + 1}.pkl"), "rb") as f:
        all_preds = pickle.load(f)

    with open(os.path.join("results", f"targets_trial={trial_idx + 1}.pkl"), "rb") as f:
        all_targets = pickle.load(f)

    with open(os.path.join("results", f"fu_pairs_trial={trial_idx + 1}.pkl"), "rb") as f:
        fu_pairs = pickle.load(f)

    sns.set_theme(style="whitegrid", palette="colorblind", font_scale=1.2)

    if args.dataset_type == "heat":
        reference_true = kx_names[0]
        fig = plt.figure(figsize=(10, 4.2))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.8], wspace=0.1)

        ax_true = fig.add_subplot(gs[0])
        ax_true.imshow(all_targets[reference_true], cmap='viridis')
        ax_true.set_title("Heat Equation (Truth)", fontsize=16, fontweight='bold')
        ax_true.axis("off")

        grid = gs[1].subgridspec(2, 2, hspace=0.25, wspace=0.2)
        for idx, kx_model in enumerate(kx_names):
            ax = fig.add_subplot(grid[idx])
            pred_img = -all_preds[reference_true][kx_model]
            ax.imshow(pred_img, cmap='viridis')
            ax.set_title(format_title(kx_model), fontsize=13)
            ax.axis("off")

        fig.subplots_adjust(left=0.03, right=0.97, top=0.90, bottom=0.08)
        save_path = os.path.join("results", "preds_heat.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        # (f,u) pairs visualization
        f_all = fu_pairs[reference_true]["f"]
        u_all = fu_pairs[reference_true]["u"]
        num_pairs = min(args.num_fu_pairs, f_all.shape[0])
        fig, axs = plt.subplots(2, num_pairs, figsize=(3 * num_pairs, 6))
        for idx in range(num_pairs):
            axs[0, idx].imshow(f_all[idx], cmap="viridis")
            axs[0, idx].set_title(f"f_{idx+1}")
            axs[0, idx].axis("off")

            axs[1, idx].imshow(u_all[idx], cmap="viridis")
            axs[1, idx].set_title(f"u_{idx+1}")
            axs[1, idx].axis("off")
        fig.suptitle("Sample (f, u) pairs - Heat Equation", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join("results", "fu_pairs_heat.png"), dpi=300, bbox_inches="tight")
        plt.close()
    else:
        for kx_true in kx_names:
            fig = plt.figure(figsize=(10, 4.2))  # smaller overall width
            gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.8], wspace=0.1)

            # Left: Truth
            ax_true = fig.add_subplot(gs[0])
            ax_true.imshow(all_targets[kx_true], cmap='viridis')
            ax_true.set_title(f"{format_title(kx_true)} (Truth)", fontsize=16, fontweight='bold')
            ax_true.axis("off")

            # Right: 2×2 grid
            grid = gs[1].subgridspec(2, 2, hspace=0.25, wspace=0.2)  # increase vertical spacing
            for idx, kx_model in enumerate(kx_names):
                ax = fig.add_subplot(grid[idx])
                pred_img = -all_preds[kx_true][kx_model]
                ax.imshow(pred_img, cmap='viridis')
                ax.set_title(format_title(kx_model), fontsize=13)
                ax.axis("off")

            fig.subplots_adjust(left=0.03, right=0.97, top=0.90, bottom=0.08)  # tighter layout
            save_path = os.path.join("results", f"preds_{kx_true}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            # (f,u) pairs visualization per kx_true
            f_all = fu_pairs[kx_true]["f"]
            u_all = fu_pairs[kx_true]["u"]
            num_pairs = min(args.num_fu_pairs, f_all.shape[0])
            fig, axs = plt.subplots(2, num_pairs, figsize=(3 * num_pairs, 6))
            for idx in range(num_pairs):
                axs[0, idx].imshow(f_all[idx], cmap="viridis")
                axs[0, idx].set_title(f"f_{idx+1}")
                axs[0, idx].axis("off")

                axs[1, idx].imshow(u_all[idx], cmap="viridis")
                axs[1, idx].set_title(f"u_{idx+1}")
                axs[1, idx].axis("off")
            fig.suptitle(f"Sample (f, u) pairs - {format_title(kx_true)}", fontsize=16, fontweight="bold")
            plt.tight_layout()
            plt.savefig(os.path.join("results", f"fu_pairs_{kx_true}.png"), dpi=300, bbox_inches="tight")
            plt.close()


if __name__ == "__main__":
    main()
