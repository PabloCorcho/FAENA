#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 10:41:18 2022

This script comprises functions typically used in astronomy

@author: pablo
"""

import numpy as np

def schechter(x, phi, x_0, alpha, log=False):
    """Return a Schechter function.

    Params
    ------
    - x: (array)
    - phi: (float) Normalization constant.
    - x_0: (float) Transition value of x where the function decays
    exponentially.
    - alpha: Power-law slope for x << x_0.
    - log: (bool) If True, returns log10(schechter)
    Returns
    -------
    - schechter: (array) Schechter function evaluated at x.
    """
    s = phi * (x/x_0)**alpha * np.exp(-x/x_0)
    if log:
        s = np.log10(s)
    return s


def double_schechter(x, phi1, phi2, x_01, x_02, alpha1, alpha2):
    """Double schechter function."""
    ds = schechter(x, phi1, x_01, alpha1) + schechter(x, phi2, x_02, alpha2)
    return ds


def double_power_law(x, phi, x0, alpha, beta):
    """..."""
    xx = x/x0
    f = phi * xx**alpha / (1 + xx**beta)
    return f

# Mr Krtxo \(ﾟ▽ﾟ)/
