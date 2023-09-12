#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 08:56:58 2023

@author: pablo

Install extinction packages

```
pip install extinction
```

"""

import numpy as np
from extinction import ccm89, calzetti00, fitzpatrick99


def get_balmer_decrement(halpha, hbeta, bd_ref=2.86):
    """Compute ratio of Balmer Decrement."""
    bd = halpha / hbeta
    return bd / bd_ref

def get_extinction(wavelength, halpha, hbeta, ext_curve=ccm89, r_v=3.1,
                   **kwargs):
    """Compute the extinction in magnitudes from Balmer Decrement."""
    bd = get_balmer_decrement(halpha, hbeta, **kwargs)
    # Avoid negative extinction
    bd = np.clip(np.array(bd), a_min=1., a_max=None)
    # Adopted extinction curve for input wavelength
    wavelenght = np.hstack(
        (np.array([4861., 6563.]),
         np.array([wavelength], dtype=float).squeeze())
        )
    # Set extinction cuver
    k_hb, k_ha, *k_lambda = ext_curve(wavelenght, a_v=1.0, r_v=r_v)
    # Compute the extinction
    a_lambda = (2.5 * np.array(k_lambda)) / (k_hb - k_ha) * np.log10(bd)

    return a_lambda

def get_intrinsic_flux(wavelength, fluxes, halpha, hbeta, **kwargs):
    """Get the instrinsic flux.
    
    Parameters
    ----------
    - wavelength: sequence
        Wavelength array of input flux values.
    - flux: sequence
        Reddened fluxes.
    - halpha: float
        H-alpha line flux
    - hbeta: float
        H-beta line flux
    
    Additional parameters
    ---------------------
    - r_v: float, default=3.1
        Default value for R_V = A_V / E(B-V)
    - ext_curve: function, default=ccm89
        Reference extinction curve function. Default is Cardelli+98 extinction
        curve. It must accept wavelength (AA) as first argument, and a_v
        and r_v as optional keyword arguments.
    - bd_red: float, default=2.86
        Intrinsic ratio between Halpha and Hbeta.
    Returns
    -------
    - intrinsic_flux: np.ndarray
        Array containing the extinction-corrected flux values.
    """
    extinction = get_extinction(wavelength, halpha, hbeta, **kwargs)
    intrinsic_flux = fluxes * 10**(0.4 * extinction)
    return intrinsic_flux

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    
    
    wavelength = np.linspace(4000, 9000, 100)
    # example SED (all ones to see the correction factor after de-reddening)
    fluxes = np.ones_like(wavelength)
    # Fake extinction
    halpha, hbeta = 4, 1
    
    plt.figure()
    plt.plot(wavelength, fluxes, c='k', label='Reddened')
    for c, curve in zip(['r', 'g', 'b'], [ccm89, calzetti00, fitzpatrick99]):
        
        int_fluxes = get_intrinsic_flux(wavelength, fluxes, halpha, hbeta,
                                        ext_curve=curve)
        plt.plot(wavelength, int_fluxes, c=c, label=str(curve).split(" ")[-1])
    
    plt.legend()