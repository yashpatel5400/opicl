import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftn, ifftn

# GRF generator for random input functions
def GRF(alpha, beta, gamma, N, num_samples=10):
    xi = np.random.randn(num_samples, N, N)
    K1, K2 = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    freq_sq = (K1**2 + K2**2)
    coef = alpha**0.5 * (4*np.pi**2 * freq_sq + beta)**(-gamma/2)
    L = N * coef * xi
    L[:, 0, 0] = 0  # enforce mean 0
    f_spatials = np.real(ifftn(L, norm='forward', axes=(-2, -1)))
    return f_spatials

# Dataset generation
def make_simple_learnable_operator_dataset(num_samples=25, H=64, W=64, sigma=1.0, seed=0):
    np.random.seed(seed)
    y = np.arange(H)
    x = np.arange(W)
    Y, X = np.meshgrid(y, x, indexing='ij')
    center_y = (H - 1) / 2
    center_x = (W - 1) / 2
    dist_sq = (X - center_x)**2 + (Y - center_y)**2
    f0 = np.exp(-dist_sq / (2 * (H/8)**2))
    f0 /= np.linalg.norm(f0)

    kernel_spatial = np.exp(-dist_sq / (2 * sigma**2))
    kernel_spatial /= np.sum(kernel_spatial)
    kernel_hat = fft2(kernel_spatial)
    f0_hat = fft2(f0)
    conv_f0 = ifft2(kernel_hat * f0_hat).real

    alpha, beta, gamma, N = 1.0, 1.0, 4.0, H
    fs = GRF(alpha, beta, gamma, N, num_samples)

    Ofs = np.array([
        np.sum(f_i * f0) * conv_f0
        for f_i in fs
    ])

    return fs, Ofs, f0

# Generate data
icl_num_samples = 100
f, Of, f0 = make_simple_learnable_operator_dataset(num_samples=icl_num_samples)

# Split into train and test
f_train  = f[:-1]
Of_train = Of[:-1]
f_test   = f[-1]
Of_test  = Of[-1]

# Linear regression in operator RKHS
# Model: O_hat(f) = sum_i alpha_i * <f, f_i> * conv_f0

# Precompute inner products <f_train[i], f_train[j]>
T, H, W = f_train.shape
gram = np.array([
    [np.sum(f_train[i] * f_train[j]) for j in range(T)]
    for i in range(T)
])

# Flatten training outputs
Y = Of_train.reshape(T, -1)  # shape (T, H*W)

# Solve (G + lambda I) alpha = Y for each pixel
G_reg = gram
alpha = np.linalg.solve(G_reg, Y)  # shape (T, H*W)

# Predict Of_test
test_inner_prods = np.array([np.sum(f_test * f_train[i]) for i in range(T)])  # shape (T,)
Of_test_pred_flat = test_inner_prods @ alpha  # shape (H*W,)
Of_test_pred = Of_test_pred_flat.reshape(H, W)

# Retry with fewer steps and more efficient updates
alpha_gd = np.zeros_like(alpha)
lr = 1e-5
num_steps = 500
max_grad_norm = 1e2

losses = []

for step in range(num_steps):
    residual = gram @ alpha_gd - Y
    grad = gram.T @ residual

    # Gradient clipping
    grad_norm = np.linalg.norm(grad)
    if grad_norm > max_grad_norm:
        grad = grad * (max_grad_norm / grad_norm)

    alpha_gd -= lr * grad

    # Track loss
    if step % 50 == 0 or step == num_steps - 1:
        loss = 0.5 * np.linalg.norm(residual)**2
        losses.append((step, loss))

# Predict using GD solution
Of_test_pred_gd_flat = test_inner_prods @ alpha_gd
Of_test_pred_gd = Of_test_pred_gd_flat.reshape(H, W)

# Plot comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(Of_test, cmap='viridis')
axes[0].set_title("Ground Truth: O f_test")
axes[0].axis('off')

axes[1].imshow(Of_test_pred, cmap='viridis')
axes[1].set_title("Closed-Form RKHS Prediction")
axes[1].axis('off')

axes[2].imshow(Of_test_pred_gd, cmap='viridis')
axes[2].set_title("Gradient Descent Prediction")
axes[2].axis('off')

plt.tight_layout()
plt.savefig("debug.png")