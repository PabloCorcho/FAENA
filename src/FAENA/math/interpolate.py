#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 17:42:58 2022

@author: pablo
"""

import numpy as np

__all__ = ['regular_bilin_interp']


def regular_bilin_interp(x, y, z, new_x, new_y):
    """Apply a bilinear interpolation at f(x_new, y_new).

    Params
    ------
    - x: (1D-array) Original x values.
    - y: (1D-array) Original y values.
    - z: (2D-array) Original z values.
    - new_x: (1D- or 2D-array) New x values to interpolate z.
    - new_y: (1D- or 2D-array) New y values to interpolate z.
    """
    shape = new_x.shape
    if len(new_x.shape) > 1 and len(new_y.shape) > 1:
        new_x = new_x.flatten()
        new_y = new_y.flatten()

    x_idx = np.searchsorted(x, new_x, side='right')
    x_idx = np.clip(x_idx, a_min=0, a_max=x.size - 1)
    y_idx = np.searchsorted(y, new_y, side='right')
    y_idx = np.clip(y_idx, a_min=0, a_max=y.size - 1)

    x_1 = x[x_idx - 1]
    y_1 = y[y_idx - 1]
    x_2 = x[x_idx]
    y_2 = y[y_idx]
    dx = x_2 - x_1
    dy = y_2 - y_1
    w_22, w_21, w_12, w_11 = ((new_x - x_1) / dx * (new_y - y_1) / dy,
                              (new_x - x_1) / dx * (y_2 - new_y) / dy,
                              (x_2 - new_x) / dx * (new_y - y_1) / dy,
                              (x_2 - new_x) / dx * (y_2 - new_y) / dy)
    q_11, q_12, q_21, q_22 = (z[x_idx - 1, y_idx - 1], z[x_idx - 1, y_idx],
                              z[x_idx, y_idx - 1], z[x_idx, y_idx])
    z_new = q_11 * w_11 + q_12 * w_12 + q_21 * w_21 + q_22 * w_22
    return z_new.reshape(shape)

# Mr Krtxo \(ﾟ▽ﾟ)/
