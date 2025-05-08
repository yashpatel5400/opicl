import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt

from opformer import TransformerOperator
import kernels
import dataset


def load_latest_checkpoint(model, checkpoint_dir="checkpoints"):
    checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, "opformer_epoch_*.pth")))
    if not checkpoint_files:
        raise FileNotFoundError("No checkpoint files found in directory.")
    latest_ckpt = checkpoint_files[-1]
    print(f"✅ Loading checkpoint: {latest_ckpt}")
    model.load_state_dict(torch.load(latest_ckpt))
    model.eval()
    return model


def test_and_visualize(model, kx, ky, im_size=(64, 64), device="cuda", seed=123, out_dir="results_train"):
    os.makedirs(out_dir, exist_ok=True)

    # === Generate a new test sample ===
    f, Of = dataset.make_random_operator_dataset(
        kx, ky, num_samples=25, num_bases=10, H=im_size[0], W=im_size[1], seed=seed
    )
    f_test = f[-1]
    Of_test = Of[-1]
    Z_test = dataset.construct_Z(f_test, Of, f, im_size, device=device)  # shape: (1, 25, 2H, W)

    # === Run prediction ===
    with torch.no_grad():
        preds, _ = model(Z_test)  # shape: (1, H, W)
        preds = preds.squeeze(0).cpu().numpy()
        Of_test = Of_test  # shape: (H, W)
        residual = preds + Of_test

    # === Plot and save ===
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(Of_test, cmap='viridis')
    axs[0].set_title("True Output")
    axs[0].axis("off")

    axs[1].imshow(-preds, cmap='viridis')
    axs[1].set_title("Model Prediction")
    axs[1].axis("off")

    axs[2].imshow(residual, cmap='viridis')
    axs[2].set_title("Residual")
    axs[2].axis("off")

    plt.tight_layout()
    save_path = os.path.join(out_dir, f"visualization_seed_{seed}.png")
    plt.savefig(save_path)
    print(f"✅ Visualization saved to {save_path}")
    plt.close()

def get_normalized_kernels(checkpoint_dir, model, layers_to_check=None):
    """
    Analyze whether W_k,l ~ b_l * Sigma^{-1} for learned Fourier weights across layers.

    Args:
        checkpoint_dir (str): Directory with model checkpoints (*.pth)
        model (nn.Module): TransformerOperator instance (unloaded)
        layers_to_check (list[int], optional): Which layers to evaluate. If None, checks all.
    """
    ckpt_files = sorted(
        [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")],
        key=lambda f: int("".join(filter(str.isdigit, f)))
    )

    last_ckpt = os.path.join(checkpoint_dir, ckpt_files[-1])
    print(f"Loading weights from: {last_ckpt}")
    state_dict = torch.load(last_ckpt, map_location="cpu")
    model.load_state_dict(state_dict)

    num_layers = len(model.layers)
    if layers_to_check is None:
        layers_to_check = list(range(num_layers))

    # Extract key operator Fourier weights
    R_k_list = []
    for l in layers_to_check:
        weight = model.layers[l].self_attn.key_operator.weights1.data
        R_k = weight.squeeze()  # shape: (modes1, modes2)
        R_k_list.append(R_k)

    # Normalize all R_k to unit norm
    R_k_normed = [(R / torch.norm(R)).detach().cpu().numpy() for R in R_k_list]
    return R_k_normed

def compute_residual_norms(normalized_kernels):
    """
    Compute residuals from the first normalized kernel as reference (i.e., Σ^{-1}).
    Returns list of L2 residual norms per layer.
    """
    ref = normalized_kernels[0]
    residuals = []
    for k in normalized_kernels:
        diff = k - ref
        residual_norm = np.linalg.norm(diff)
        residuals.append(residual_norm)
    return residuals

def plot_residuals(residuals, save_path=None):
    plt.figure(figsize=(6, 4))
    plt.plot(residuals, marker='o')
    plt.title("Residuals of $\\tilde{R}_{k,\\ell}$ from $\\Sigma^{-1}$")
    plt.xlabel("Layer $\\ell$")
    plt.ylabel("Residual Norm $\\| \\tilde{R}_{k,\\ell} - \\Sigma^{-1} \\|$")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()

def main():
    # === Config ===
    H, W = 64, 64
    im_size = (H, W)
    device = "cuda"

    kernel_maps = kernels.Kernels(H, W)
    ky_kernel = kernel_maps.get_kernel("gaussian")
    kx_name = "linear"
    kx = kernels.get_kx_kernel(kx_name, sigma=1.0)

    # === Model definition (must match training config) ===
    model = TransformerOperator(
        num_layers=250,
        im_size=im_size,
        ky_kernel=ky_kernel,
        kx_name=kx_name,
        kx_sigma=1.0,
        icl_lr=-0.01,
        icl_init=False
    ).to(device)

    normalized_kernels = get_normalized_kernels("checkpoints", model)
    residuals = compute_residual_norms(normalized_kernels)
    plot_residuals(residuals, save_path="kernel_key_operator_residuals.png")

if __name__ == "__main__":
    main()
