import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

from opformer import TransformerOperator
import kernels
import dataset

class MetaOperatorDataset(Dataset):
    def __init__(self, num_meta_samples, kx, ky, num_incontext=25, num_bases=5, im_size=(64, 64), device="cuda"):
        self.num_meta_samples = num_meta_samples
        self.kx = kx
        self.ky = ky
        self.num_incontext = num_incontext
        self.num_bases = num_bases
        self.H, self.W = im_size
        self.device = device

    def __len__(self):
        return self.num_meta_samples

    def __getitem__(self, idx):
        # Sample new basis functions and weights for this operator
        alpha, beta, gamma, N = 1.0, 1.0, 4.0, self.H
        basis_fs = dataset.GRF(alpha, beta, gamma, N, self.num_bases)
        basis_gs = dataset.GRF(alpha, beta, gamma, N, self.num_bases)
        lambdas = np.random.randn(self.num_bases)
        g_convs = np.array([dataset.np_conv(self.ky, g) for g in basis_gs])

        # Sample new input functions
        fs = dataset.GRF(alpha, beta, gamma, N, self.num_incontext)

        # Apply operator
        Ofs = np.zeros((self.num_incontext, self.H, self.W))
        for j in range(self.num_incontext):
            for i in range(self.num_bases):
                inner = self.kx(fs[j], basis_fs[i])
                Ofs[j] += lambdas[i] * inner * g_convs[i]

        f_test = fs[-1]
        Of_test = Ofs[-1]

        Z_test = dataset.construct_Z(f_test, Ofs, fs, (self.H, self.W), device=self.device)
        return Z_test.squeeze(0), torch.from_numpy(Of_test).to(torch.float32).to(self.device)


def get_dataloader(num_meta_samples, batch_size, kx, ky, im_size=(64, 64), device="cuda"):
    dataset = MetaOperatorDataset(
        num_meta_samples=num_meta_samples,
        kx=kx,
        ky=ky,
        num_incontext=25,
        num_bases=10,
        im_size=im_size,
        device=device
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


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
    device = "cuda"
    im_size = (H, W)

    kernel_maps = kernels.Kernels(H, W)
    ky_true = kernel_maps.get_kernel("gaussian")
    kx_name = 'linear'
    kx = kernels.get_kx_kernel(kx_name, sigma=1.0)

    opformer = TransformerOperator(
        num_layers=250, 
        im_size=im_size, 
        ky_kernel=ky_true, 
        kx_name=kx_name, 
        kx_sigma=1.0, 
        icl_lr=-0.01, 
        icl_init=False,
    ).to(device)

    dataset_len = 100
    dataloader = get_dataloader(
        num_meta_samples=dataset_len,
        batch_size=8,
        kx=kx,
        ky=ky_true,
        im_size=im_size,
        device=device
    )

    optimizer = torch.optim.Adam(opformer.parameters(), lr=1e-4)
    losses = []

    for epoch in range(100):
        print(f"Starting epoch {epoch}...")
        total_loss = 0.0
        for Z_batch, Of_batch in dataloader:
            Z_batch = Z_batch.to(device)
            Of_batch = Of_batch.to(device)

            preds, _ = opformer(Z_batch)
            loss = F.mse_loss(preds + Of_batch, torch.zeros_like(preds))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch {epoch} | Loss: {avg_loss:.6f}")

        if epoch % 2 == 0:
            log_predictions(Z_batch, Of_batch, opformer, epoch)

            # Save the model every 5 epochs
            os.makedirs("checkpoints", exist_ok=True)
            model_path = f"checkpoints/opformer_epoch_{epoch}.pth"
            torch.save(opformer.state_dict(), model_path)
            print(f"Model saved to {model_path}")


if __name__ == "__main__":
    train()


