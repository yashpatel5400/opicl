import numpy as np
from numpy.fft import fftn, ifftn
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

################################################
# 1) Build the diagonal covariance matrix Sigma
#    in a vectorized manner
################################################
def build_operator_covariance_matrix(K, sigma_gauss):
    """
    Build the big diagonal matrix Sigma_{(j,i),(j,i)} of size (side^2 * side^2).
    In the Hilbert-Schmidt setting with:
      k_x = linear kernel => diagonal on X's Fourier basis
      k_y = Gaussian kernel => diagonal on Y's Fourier basis
    the final matrix is diagonal with
      diag((j,i)) = gauss_factor[j],
    ignoring i because <varphi_i, varphi_k> = delta_{i,k}.

    We'll store this as a dense diagonal matrix, but
    in practice you might keep only the diag vector.
    """

    side = 2*K + 1
    big_size = side**2 * side**2

    # -- We want an array of all (j,i) indices in [-K..K]^2
    rangev = np.arange(-K, K+1)
    J1, J2 = np.meshgrid(rangev, rangev, indexing='ij')  # shape=(side, side)
    # Flatten them
    j1a = J1.ravel()  # shape=(side^2,)
    j2a = J2.ravel()

    # We'll create a diagonal vector diagVals of length big_size.
    # diagVals[row] = gauss_factor[j],  ignoring i in the sense that
    # row = t_j*(side^2) + t_i, so j depends only on t_j.
    diagVals = np.zeros(big_size, dtype=np.float64)

    # Precompute frequency squares for each possible j.
    # freq_j^2 = (2*pi*j1)^2 + (2*pi*j2)^2
    freqsq_j = (2*np.pi*j1a)**2 + (2*np.pi*j2a)**2
    # Then gauss_factor[j] = exp(-0.5*sigma_gauss^2 * freqsq_j)
    gauss_factors = np.exp(-0.5 * sigma_gauss**2 * freqsq_j)

    # Now for each row in [0..big_size), we decode:
    #   row = t_j*(side^2) + t_i,
    # so t_j = row // (side^2).
    # diagVals[row] = gauss_factors[t_j].
    rows = np.arange(big_size)
    t_j = rows // (side**2)  # integer division
    diagVals = gauss_factors[t_j]

    # Build Sigma as a diagonal matrix
    Sigma = np.diag(diagVals)
    return Sigma


def sample_random_operator(N=64, K=8, num_samples=1000, sigma_gauss=1.0, alpha=1.0, beta=1.0, gamma=4.0):
    side = 2*K + 1

    # --- Step 1: Sample a single operator ---
    Sigma = build_operator_covariance_matrix(K, sigma_gauss)
    big_size = side**2 * side**2
    z = np.random.randn(big_size)
    alpha_vec = np.sqrt(np.diag(Sigma)) * z
    alpha2D = alpha_vec.reshape(side**2, side**2)

    # --- Step 2: Sample multiple input functions ---
    xi = np.random.randn(num_samples, N, N)
    K1, K2 = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    freq_sq = (K1**2 + K2**2)
    coef = alpha**0.5 * (4*np.pi**2 * freq_sq + beta)**(-gamma/2)
    L = N * coef * xi
    L[:, 0, 0] = 0  # mean-zero
    f_real = np.real(ifftn(L, norm='forward', axes=(-2, -1)))

    # --- Step 3: Get truncated Fourier coefficients ---
    F = fftn(f_real, norm='forward', axes=(-2, -1))
    rowvals = np.arange(-K, K+1)
    colvals = np.arange(-K, K+1)
    R, C = np.meshgrid(rowvals, colvals, indexing='ij')
    Rmod = R % N
    Cmod = C % N
    fHat2Ds = F[:, Rmod, Cmod]  # (num_samples, side, side)

    # --- Step 4: Apply the same operator to all f's ---
    f_vecs = fHat2Ds.reshape(num_samples, side**2)
    Of_vecs = f_vecs @ alpha2D.T  # (num_samples, side^2)
    Of2Ds = Of_vecs.reshape(num_samples, side, side)

    # --- Step 5: Convert back to real space ---
    O_F_full = np.zeros((num_samples, N, N), dtype=complex)
    f_full   = np.zeros((num_samples, N, N), dtype=complex)
    O_F_full[:, Rmod, Cmod] = Of2Ds
    f_full[:, Rmod, Cmod] = fHat2Ds

    Of_spatials = np.real(ifftn(O_F_full, norm='forward', axes=(-2, -1)))
    f_spatials  = np.real(ifftn(f_full,    norm='forward', axes=(-2, -1)))

    return f_spatials, Of_spatials, alpha2D


def get_spatial_coordinates(N):
    """
    Generate x,y coordinates for spatial discretization over [0,2π)².
    
    Args:
        N (int): Number of grid points in each dimension
        
    Returns:
        tuple: (x_coords, y_coords) where each is a 2D array of shape (N,N)
               containing the x and y coordinates respectively
    """
    # Create evenly spaced points from 0 to 2π (exclusive)
    points = np.linspace(0, 2*np.pi, N, endpoint=False)
    
    # Create 2D coordinate grids
    x_coords, y_coords = np.meshgrid(points, points, indexing='ij')
    
    return x_coords, y_coords


def l2_norm(u, v):
    return np.sqrt(np.sum((u - v) ** 2))


def GRF(alpha, beta, gamma, N, num_samples=10):
    xi = np.random.randn(num_samples, N, N)
    K1, K2 = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    freq_sq = (K1**2 + K2**2)
    coef = alpha**0.5 * (4*np.pi**2 * freq_sq + beta)**(-gamma/2)
    L = N * coef * xi
    L[:, 0, 0] = 0  # enforce mean 0
    f_spatials = np.real(ifftn(L, norm='forward', axes=(-2, -1)))
    return f_spatials


def plot_operator_smoothness_check(N=64, K=8, num_samples=50, sigma_gauss=1.0, alpha=1.0, beta=1.0, gamma=4.0):
    side = 2*K + 1

    # Sample a single operator
    j1a, j2a = np.meshgrid(np.arange(-K, K+1), np.arange(-K, K+1), indexing='ij')
    j1a, j2a = j1a.ravel(), j2a.ravel()
    freqsq_j = (2*np.pi*j1a)**2 + (2*np.pi*j2a)**2
    gauss_factors = np.exp(-0.5 * sigma_gauss**2 * freqsq_j)
    diagVals = np.repeat(gauss_factors, side**2)
    alpha_vec = np.sqrt(diagVals) * np.random.randn(side**2 * side**2)
    alpha2D = alpha_vec.reshape(side**2, side**2)

    # Sample functions
    f_spatials = GRF(alpha, beta, gamma, N, num_samples)

    # Get Fourier coefficients and apply operator
    F = fftn(f_spatials, norm='forward', axes=(-2, -1))
    rowvals = np.arange(-K, K+1)
    colvals = np.arange(-K, K+1)
    R, C = np.meshgrid(rowvals, colvals, indexing='ij')
    Rmod = R % N
    Cmod = C % N
    fHat2Ds = F[:, Rmod, Cmod]
    f_vecs = fHat2Ds.reshape(num_samples, side**2)
    Of_vecs = f_vecs @ alpha2D.T
    Of2Ds = Of_vecs.reshape(num_samples, side, side)

    # Convert to real space
    O_F_full = np.zeros((num_samples, N, N), dtype=complex)
    f_full   = np.zeros((num_samples, N, N), dtype=complex)
    O_F_full[:, Rmod, Cmod] = Of2Ds
    f_full[:, Rmod, Cmod] = fHat2Ds
    Of_spatials = np.real(ifftn(O_F_full, norm='forward', axes=(-2, -1)))

    # Compute pairwise distances
    input_dists = []
    output_dists = []
    for i in range(num_samples):
        for j in range(i+1, num_samples):
            input_dists.append(l2_norm(f_spatials[i], f_spatials[j]))
            output_dists.append(l2_norm(Of_spatials[i], Of_spatials[j]))

    # Plot
    plt.figure(figsize=(6,5))
    plt.scatter(input_dists, output_dists, alpha=0.6, s=12)
    plt.xlabel(r"$\|f^{(i)} - f^{(j)}\|_{L^2}$")
    plt.ylabel(r"$\|\mathcal{O}f^{(i)} - \mathcal{O}f^{(j)}\|_{L^2}$")
    plt.title("Smoothness of Sampled Operator")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("check_smooth.png")

if __name__ == "__main__":
    plot_operator_smoothness_check() # Sanity check