import numpy as np
from typing import Callable

from numpy.fft import fft2
import numpy as np
from typing import Callable

class Kernels:
    def __init__(self, H, W):
        self.H = H
        self.W = W
        self.coords = self.get_coords(H, W)
        self.kernel_maps = {
            "gaussian"    : self.get_gaussian_kernel,
            "laplace"     : self.get_laplace_kernel,
            "exponential" : self.get_exponential_kernel,
            "mexican_hat" : self.get_mexican_hat_kernel,
            "cauchy"      : self.get_cauchy_kernel,
        }

    def available_kernels(self):
        return list(self.kernel_maps.keys())

    def get_kernel(self, kernel_name, sigma=1.0):
        return self.kernel_maps[kernel_name](sigma)

    def get_coords(self, H, W):
        y = np.arange(H)
        x = np.arange(W)
        Y, X = np.meshgrid(y, x, indexing='ij')
        center_y = (H - 1) / 2
        center_x = (W - 1) / 2
        Xc = X - center_x
        Yc = Y - center_y
        return Xc, Yc

    def get_gaussian_kernel(self, sigma=1.0):
        Xc, Yc = self.coords
        dist_sq = Xc**2 + Yc**2
        kernel = np.exp(-dist_sq / (2 * sigma**2))
        kernel /= np.sum(kernel)
        return kernel

    def get_laplace_kernel(self, sigma=1.0):
        Xc, Yc = self.coords
        dist_sq = Xc**2 + Yc**2
        kernel = np.exp(-np.sqrt(dist_sq) / (2 * sigma**2))
        kernel /= np.sum(kernel)
        return kernel

    def get_exponential_kernel(self, sigma=1.0):
        Xc, Yc = self.coords
        dist = np.sqrt(Xc**2 + Yc**2)
        kernel = np.exp(-dist / sigma)
        kernel /= np.sum(kernel)
        return kernel

    def get_mexican_hat_kernel(self, sigma=1.0):
        Xc, Yc = self.coords
        r2 = (Xc**2 + Yc**2) / sigma**2
        kernel = (1 - r2) * np.exp(-r2 / 2)
        kernel -= np.mean(kernel)
        kernel /= np.sum(np.abs(kernel))
        return kernel

    def get_cauchy_kernel(self, sigma=1.0):
        Xc, Yc = self.coords
        dist_sq = Xc**2 + Yc**2
        kernel = 1 / (1 + dist_sq / sigma**2)
        kernel /= np.sum(kernel)
        return kernel

def get_kx_kernel(kernel_type, sigma=1.0):
    """
    Returns a kernel function k_x(f1, f2) depending on the kernel_type.
    Each f1, f2 is a 2D array (e.g., shape H x W).

    Supported types: 'linear', 'rbf', 'laplacian', 'gradient_rbf', 'energy'
    """
    if kernel_type == 'linear':
        return lambda f1, f2: np.sum(f1 * f2)

    elif kernel_type == 'laplacian':
        return lambda f1, f2: np.exp(-np.sum(np.abs(f1 - f2)) / sigma)

    elif kernel_type == 'gradient_rbf':
        def grad_rbf(f1, f2):
            f1x, f1y = np.gradient(f1)
            f2x, f2y = np.gradient(f2)
            grad_diff = (f1x - f2x)**2 + (f1y - f2y)**2
            return np.exp(-np.sum(grad_diff) / (2 * sigma**2))
        return grad_rbf

    elif kernel_type == 'energy':
        def energy_k(f1, f2):
            diff = np.linalg.norm(f1)**2 - np.linalg.norm(f2)**2
            return np.exp(-diff**2 / (2 * sigma**2))
        return energy_k

    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")