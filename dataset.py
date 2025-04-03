import numpy as np
from numpy.fft import fftn, ifftn
import torch
from torch.utils.data import Dataset

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


################################################
# 2) Sample operator coefficients alpha_{(j,i)}
#    from N(0, Sigma). Sigma is diagonal => easy
################################################
def sample_operator_coeffs(Sigma, K):
    """
    Sigma is diagonal => sampling is alpha_vec = sqrt(diag(Sigma)) * z
    where z ~ N(0,I).
    We'll store alpha_{(j,i)} in an array alpha2D of shape (side^2, side^2),
    i.e. alpha2D[t_j, t_i].
    Then j,i can be recovered from the flattened indices.
    """
    side = 2*K + 1
    big_size = side**2 * side**2

    diagVals = np.diag(Sigma)  # shape (big_size,)
    z = np.random.randn(big_size)
    alpha_vec = np.sqrt(diagVals) * z  # elementwise multiplication

    # Reshape into alpha2D[t_j, t_i], each in [0..side^2-1].
    alpha2D = alpha_vec.reshape(side**2, side**2)
    return alpha2D


################################################
# 3) Build a random function f using GRF
#    and get its truncated Fourier coefficients
################################################
def build_random_function_hat(alpha, beta, gamma, N, K):
    """
    Use the provided GRF(...) to get a real-space field.
    Then compute its truncated Fourier coefficients f_hat[i].
    We'll store these in an array fHat2D of shape (side, side),
    with side=2K+1, centered on frequencies -K..K.

    NOTE: There's more than one way to handle indexing & normalization;
          be consistent with your main code assumptions.
    """
    # 3a) Use the user-provided GRF(...) directly
    L = GRF(alpha, beta, gamma, N)  # freq-domain amplitude * random
    f_real = ifftn(L, norm='forward')  # shape=(N,N) => real-space field approx

    # 3b) Compute forward FFT to get the full freq domain
    F = fftn(f_real, norm='forward')  # shape=(N,N)

    # 3c) Extract the block of frequencies [-K..K]^2.
    #     Suppose we treat freq indices in 0..N-1, and (j1,j2) = j1-K if we want the "centered" approach.
    #     We'll do a simple wrap-around indexing function here:
    side = 2*K+1
    fHat2D = np.zeros((side, side), dtype=complex)

    def wrap_index(idx):
        # Convert [-K..K] to the corresponding [0..N-1], mod N
        return idx % N

    rowvals = np.arange(-K, K+1)
    colvals = np.arange(-K, K+1)
    # Vectorize mesh
    R, C = np.meshgrid(rowvals, colvals, indexing='ij')  # shape=(side, side)
    # wrap them
    Rmod = wrap_index(R)
    Cmod = wrap_index(C)

    # gather
    fHat2D = F[Rmod, Cmod]

    return fHat2D


################################################
# 4) Apply the operator in the freq domain
#    in a vectorized manner:
#      (Of_hat)[t_j] = sum_{t_i} alpha2D[t_j, t_i] * f_hat[t_i]
################################################
def apply_operator(alpha2D, fHat2D):
    """
    alpha2D has shape (side^2, side^2).
    fHat2D has shape (side, side).
    We flatten fHat2D => vector of length side^2, multiply by alpha2D => new freq vector => reshape => O(f) freq block.
    """
    side = fHat2D.shape[0]
    # flatten
    fvec = fHat2D.ravel()  # shape (side^2,)
    Ofvec = alpha2D.dot(fvec)  # shape (side^2,)

    # reshape back to (side, side)
    Of2D = Ofvec.reshape(side, side)
    return Of2D


################################################
# 5) The user-provided GRF function
################################################
def GRF(alpha, beta, gamma, N):
    """
    Returns the random field in FREQUENCY space scaled by the
    factor N and the (eigenvalue)^(-gamma/2).
    The output shape is (N, N). If you then do ifftn(...),
    you get real-space samples.
    """
    xi = np.random.randn(N, N)

    K1, K2 = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    freq_sq = (K1**2 + K2**2)
    coef = alpha**0.5 * (4*np.pi**2 * freq_sq + beta)**(-gamma/2)

    L = N * coef * xi
    # enforce mean 0
    L[0, 0] = 0
    return L


################################################
# 6) Main demonstration
################################################
def sample_random_operator(N=64, K=8, num_samples=1000):
    """
    Generate num_samples random operator samples in a vectorized manner.
    Returns a list of tuples (f_spatial, Of_spatial, alpha2D).
    N:         # spatial discretization of [0, 2pi)
    K:         # truncated frequencies from -2..2
    """
    # Parameters
    sigma_gauss = 0.5
    
    alpha = 1.0
    beta  = 1.0
    gamma = 4.0

    # --------------------------------
    # (A) Build diagonal covariance matrix, Sigma
    # --------------------------------
    Sigma = build_operator_covariance_matrix(K, sigma_gauss)
    side = 2*K + 1

    # --------------------------------
    # (B) Sample operator coefficients alpha_{(j,i)} for all samples at once
    # --------------------------------
    # Generate random numbers for all samples at once
    big_size = side**2 * side**2
    z = np.random.randn(num_samples, big_size)
    diagVals = np.diag(Sigma)
    alpha_vecs = np.sqrt(diagVals) * z  # shape (num_samples, big_size)
    alpha2Ds = alpha_vecs.reshape(num_samples, side**2, side**2)

    # --------------------------------
    # (C) Generate random functions f using GRF for all samples at once
    # --------------------------------
    # Generate random fields in frequency space for all samples
    xi = np.random.randn(num_samples, N, N)
    K1, K2 = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    freq_sq = (K1**2 + K2**2)
    coef = alpha**0.5 * (4*np.pi**2 * freq_sq + beta)**(-gamma/2)
    L = N * coef * xi
    # enforce mean 0 for all samples
    L[:, 0, 0] = 0
    
    # Convert to real space for all samples
    f_real = np.real(ifftn(L, norm='forward', axes=(-2, -1)))
    
    # Get truncated Fourier coefficients for all samples
    F = fftn(f_real, norm='forward', axes=(-2, -1))
    
    # Create frequency indices for all samples
    rowvals = np.arange(-K, K+1)
    colvals = np.arange(-K, K+1)
    R, C = np.meshgrid(rowvals, colvals, indexing='ij')
    Rmod = R % N
    Cmod = C % N
    
    # Extract the block of frequencies for all samples
    fHat2Ds = F[:, Rmod, Cmod]  # shape (num_samples, side, side)

    # --------------------------------
    # (D) Apply operators to all samples at once
    # --------------------------------
    # Reshape fHat2Ds for matrix multiplication
    f_vecs = fHat2Ds.reshape(num_samples, side**2)
    
    # Apply operators using einsum for efficient batch matrix multiplication
    Of_vecs = np.einsum('nij,nj->ni', alpha2Ds, f_vecs)
    Of2Ds = Of_vecs.reshape(num_samples, side, side)

    # --------------------------------
    # (E) Convert to real space for all samples
    # --------------------------------
    # Create full frequency arrays for all samples
    O_F_full = np.zeros((num_samples, N, N), dtype=complex)
    O_F_full[:, Rmod, Cmod] = Of2Ds
    
    # Convert to real space
    Of_spatials = np.real(ifftn(O_F_full, norm='forward', axes=(-2, -1)))
    
    # Get f in real space for all samples
    f_full = np.zeros((num_samples, N, N), dtype=complex)
    f_full[:, Rmod, Cmod] = fHat2Ds
    f_spatials = np.real(ifftn(f_full, norm='forward', axes=(-2, -1)))
    
    # Return list of tuples
    return f_spatials, Of_spatials


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


if __name__ == "__main__":
    sample_random_operator()
