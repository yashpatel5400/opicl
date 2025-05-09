import numpy as np
import os
import einops
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

from opformer import TransformerOperator, apply_spatial_conv, compute_kx
import kernels
import dataset


def construct_Z_batch(f_test, Of, f, device="cuda"):
    """
    Constructs Z for a batch of samples.

    Inputs:
        f_test: (B, H, W)
        Of:     (B, S, H, W)
        f:      (B, S, H, W)
        im_size: (H, W)

    Returns:
        Z: (B, S, 2H, W)
    """
    B, S, H, W = Of.shape
    f_test_exp = f_test.unsqueeze(1)  # (B, 1, H, W)
    f_full = torch.cat([f[:, :-1], f_test_exp], dim=1)  # (B, S, H, W)
    Z = torch.cat([f_full, Of], dim=2)  # (B, S, 2H, W)
    Z[:, -1, H:] = 0.0  # zero out the last output
    return Z.to(device).to(torch.float32)


class MetaOperatorDataset(Dataset):
    def __init__(self, num_meta_samples, kx_name, ky, num_incontext=25, num_bases=5, im_size=(64, 64), device="cuda", batch_size=32):
        self.num_meta_samples = num_meta_samples
        self.kx_name = kx_name
        self.ky = ky
        self.num_incontext = num_incontext
        self.num_bases = num_bases
        self.H, self.W = im_size
        self.device = device
        self.batch_size = batch_size

        # Operator generation constants
        self.alpha = 1.0
        self.beta = 1.0
        self.gamma = 4.0
        self.N = self.H
        self.lambda_std = 5.0

    def __len__(self):
        return self.num_meta_samples

    def __getitem__(self, idx):
        # Sample function values
        basis_fs = einops.rearrange(
            torch.from_numpy(dataset.GRF(self.alpha, self.beta, self.gamma, self.N, self.batch_size * self.num_bases)).to(torch.float32),
            "(n b) h w -> n b h w", n=self.batch_size
        ).unsqueeze(-1)  # (B, B', H, W, 1)

        basis_gs = einops.rearrange(
            torch.from_numpy(dataset.GRF(self.alpha, self.beta, self.gamma, self.N, self.batch_size * self.num_bases)).to(torch.float32),
            "(n b) h w -> n b h w", n=self.batch_size
        ).unsqueeze(-1)

        fs = einops.rearrange(
            torch.from_numpy(dataset.GRF(self.alpha, self.beta, self.gamma, self.N, self.batch_size * self.num_incontext)).to(torch.float32),
            "(n s) h w -> n s h w", n=self.batch_size
        ).unsqueeze(-1)  # (B, S, H, W, 1)

        # Spatial convolution: (B, B', H, W, 1)
        ky_kernel_torch = torch.from_numpy(self.ky).to(torch.float32).view(1, 1, *self.ky.shape)
        g_convs_torch = apply_spatial_conv(basis_gs, ky_kernel_torch)  # (B, B', H, W, 1)
        g_convs = g_convs_torch.squeeze(-1)  # (B, B', H, W)

        # Sample λs and apply to g_convs
        lambdas = torch.from_numpy(self.lambda_std * np.random.randn(self.batch_size, self.num_bases)).to(torch.float32)  # (B, B')
        lambdas = lambdas.unsqueeze(-1).unsqueeze(-1)  # (B, B', 1, 1)
        scaled_g_convs = lambdas * g_convs  # (B, B', H, W)

        # Compute inner products: (B, S, B')
        f_kx = compute_kx(fs, basis_fs, kx_name=self.kx_name, kx_sigma=1.0)

        # Combine via batched sum: (B, S, H, W)
        Ofs = torch.einsum("nbhw,nsb->nshw", scaled_g_convs, f_kx)
        
        # Extract test function and output (B, H, W)
        f_test = fs[:, -1].squeeze(-1)    # (B, H, W)
        Of_test = Ofs[:, -1]              # (B, H, W)
        fs_unstacked = fs.squeeze(-1)     # (B, S, H, W)

        # Construct Z using batched constructor
        Z = construct_Z_batch(f_test, Ofs, fs_unstacked, device=self.device)  # (B, S, 2H, W)

        return Z, Of_test.to(torch.float32).to(self.device)
        

def get_dataloader(num_meta_samples, batch_size, kx_name, ky, im_size=(64, 64), device="cuda"):
    dataset = MetaOperatorDataset(
        num_meta_samples=num_meta_samples,
        kx_name=kx_name,
        ky=ky,
        num_incontext=25,
        num_bases=50,
        im_size=im_size,
        device=device,
        batch_size=batch_size,
    )
    return DataLoader(dataset, batch_size=None)


def plot_loss_curve(losses, save_path=None):
    plt.figure(figsize=(6, 4))
    plt.plot(losses, label="Train Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("OpFormer Training Loss")
    plt.grid(True)
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def log_predictions(Z, Of_true, model, epoch, out_dir="logs", max_samples=2):
    model.eval()
    with torch.no_grad():
        preds, _ = model(Z)
        preds_np = preds.cpu().numpy()
        targets_np = Of_true.cpu().numpy()

        for i in range(min(max_samples, Z.shape[0])):
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.imshow(targets_np[i], cmap='viridis')
            plt.title("True Output")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.imshow(-preds_np[i], cmap='viridis')
            plt.title("Predicted Output")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.imshow(preds_np[i] + targets_np[i], cmap='viridis')
            plt.title("Residual")
            plt.axis("off")

            os.makedirs(out_dir, exist_ok=True)
            plt.savefig(f"{out_dir}/epoch_{epoch}_sample_{i}.png")
            plt.close()


def train():
    H, W = 64, 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    im_size = (H, W)

    kernel_maps = kernels.Kernels(H, W)
    ky_true = kernel_maps.get_kernel("gaussian")
    kx_name = 'linear'
    
    model = TransformerOperator(
        num_layers=25, 
        im_size=im_size, 
        ky_kernel=ky_true, 
        kx_name=kx_name, 
        kx_sigma=1.0, 
        icl_lr=-0.01, 
        icl_init=False,
    )
    model = model.to(device)

    dataset_len = 512
    dataloader = get_dataloader(
        num_meta_samples=dataset_len,
        batch_size=16,
        kx_name=kx_name,
        ky=ky_true,
        im_size=im_size,
        device=device
    )

    epochs = 1_000
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []

    results_dir = "checkpoints"
    os.makedirs(results_dir, exist_ok=True)
    model_path = os.path.join(results_dir, "opformer_epoch_0.pth")
    torch.save(model.state_dict(), model_path)
    
    for epoch in range(epochs):
        print(f"Starting epoch {epoch}...")
        total_loss = 0.0
        for batch in dataloader:
            Z_batch, Of_batch = batch
            preds, _ = model(Z_batch)
            loss = F.mse_loss(preds + Of_batch, torch.zeros_like(preds))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch {epoch} | Loss: {avg_loss:.6f}")

        if epoch % 1 == 0:
            log_predictions(Z_batch, Of_batch, model, epoch)
            model_path = os.path.join(results_dir, f"opformer_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")


if __name__ == "__main__":
    train()


