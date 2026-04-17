"""Grid creation and smoothing functions."""

import numpy as np
from scipy.ndimage import gaussian_filter


def make_height_grid(X, Y, Z, num_bins: int = 50):
    """Create a binned 2D grid of average Z values over projected coordinates."""
    xi = np.linspace(X.min(), X.max(), num_bins)
    yi = np.linspace(Y.min(), Y.max(), num_bins)

    xi_grid, yi_grid = np.meshgrid(xi, yi)
    grid = np.zeros_like(xi_grid)

    dx = xi[1] - xi[0]
    dy = yi[1] - yi[0]

    for i in range(num_bins):
        for j in range(num_bins):
            x_min = xi[i] - dx / 2
            x_max = xi[i] + dx / 2
            y_min = yi[j] - dy / 2
            y_max = yi[j] + dy / 2

            mask = (X >= x_min) & (X < x_max) & (Y >= y_min) & (Y < y_max)
            if mask.any():
                grid[j, i] = Z[mask].mean()

    return xi_grid, yi_grid, grid


def smooth_grid(grid, sigma: float = 1.5):
    """Apply Gaussian smoothing to a 2D grid."""
    return gaussian_filter(grid, sigma=sigma)
