"""
Created on Tue Jul 26 08:38:10 2022

@author: pablo

Statistical utilities.

"""

import numpy as np


# =============================================================================
# Cumulative distribution
# =============================================================================
def get_cmf(x, weights=None, norm=False, axis=0):
    """Compute the cumulative distribution function from an array x.

    Returns
    -------
    x_sorted, cmf
    """
    if weights is not None:
        pass
    else:
        weights = np.ones_like(x)
    sort_pos = np.argsort(x, axis=axis)
    cmf = np.cumsum(weights[sort_pos], axis=axis)
    if norm:
        cmf /= cmf[-1]
    return x[sort_pos], cmf

# =============================================================================
# Compute contour containing a fraction
# =============================================================================
def compute_fraction_from_map(H, xedges=None, yedges=None, **kwargs):
    """..."""
    if xedges is None:
        H_mass = H
    else:
        H_mass = (H * np.diff(xedges)[:, np.newaxis]
                  * np.diff(yedges)[np.newaxis, :])
    sorted_flat = np.argsort(H, axis=None)
    sorted_2D = np.unravel_index(sorted_flat, H.shape)
    density_sorted = H.flatten()[sorted_flat]
    cumulative_mass = np.cumsum(H_mass[sorted_2D])
    fraction_sorted = cumulative_mass/cumulative_mass[-1]
    fraction = np.interp(H, density_sorted, fraction_sorted)
    return fraction
# ==========================================================================
# Statistical distances
# =============================================================================
def kullback_leibler_divergence(p, q):
    """Kullback-Leibler divergence."""
    return np.sum(p * np.log(p/q))


def hellinger_dist(p, q):
    """Hellinger distance."""
    return 1 - np.sum(np.sqrt(p*q))


def total_variation_dist(p, q):
    """...."""
    return np.max(np.abs(p - q))


def jensen_shannon_divergence(p, q):
    """Jensen-Shannon divergence."""
    m = (p + q)/2
    js = (kullback_leibler_divergence(p, m)/2
          + kullback_leibler_divergence(q, m)/2)
    return js

# Mr Krtxo \(ﾟ▽ﾟ)/
