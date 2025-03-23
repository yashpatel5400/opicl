import numpy as np
from numpy.fft import fftn, ifftn

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
def main():
    # Parameters
    K = 2         # truncated frequencies from -2..2
    sigma_gauss = 0.5
    N = 16        # grid for the GRF

    alpha = 1.0
    beta  = 1.0
    gamma = 4.0

    # --------------------------------
    # (A) Build diagonal covariance matrix, Sigma
    # --------------------------------
    Sigma = build_operator_covariance_matrix(K, sigma_gauss)
    print("Sigma shape:", Sigma.shape, "(should be ( (2K+1)^2*(2K+1)^2, ...))")

    # --------------------------------
    # (B) Sample operator coefficients alpha_{(j,i)}
    # --------------------------------
    alpha2D = sample_operator_coeffs(Sigma, K)
    # alpha2D: shape (side^2, side^2).

    # --------------------------------
    # (C) Generate a random function f using the GRF function
    #     Then get its truncated Fourier coefficients fHat2D
    # --------------------------------
    fHat2D = build_random_function_hat(alpha, beta, gamma, N, K)
    side = 2*K + 1
    print("fHat2D shape:", fHat2D.shape, "(should be (side, side))")

    # --------------------------------
    # (D) Apply O to f in freq domain
    # --------------------------------
    Of2D = apply_operator(alpha2D, fHat2D)

    # Inverse FFT to get O(f)(y) in real space
    # We'll put Of2D back onto an NxN grid. We do the same
    # "wrap-around" indexing as build_random_function_hat:
    O_F_full = np.zeros((N,N), dtype=complex)

    rowvals = np.arange(-K, K+1)
    colvals = np.arange(-K, K+1)
    # building a small mesh
    R, C = np.meshgrid(rowvals, colvals, indexing='ij')

    def wrap_index(idx):
        return idx % N

    Rmod = wrap_index(R)
    Cmod = wrap_index(C)
    # place Of2D into the big freq array
    O_F_full[Rmod, Cmod] = Of2D

    # iFFT to get real space
    Of_spatial = ifftn(O_F_full, norm='forward')
    print("Of_spatial shape:", Of_spatial.shape)
    print("Of_spatial (real part) sample:\n", np.real(Of_spatial)[:5,:5])


if __name__ == "__main__":
    main()
