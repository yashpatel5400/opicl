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

    axs[1].imshow(preds, cmap='viridis')
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
        icl_init=True
    ).to(device)

    model = load_latest_checkpoint(model)
    test_and_visualize(model, kx, ky_kernel, im_size, device=device)


if __name__ == "__main__":
    main()
