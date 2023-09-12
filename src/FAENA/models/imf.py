#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 17:24:11 2023

@author: pablo
"""

import numpy as np
from astropy import units as u


class IMF(object):
    """..."""
    m_low = 0.1
    m_up = 100
    def __init__(self, name):
        self.name = name
    
    def density():
        pass


class Salp55_IMF(IMF):
    def __init__(self, alpha=-2.35):
        super().__init__(name='Salpeter55')
        self.alpha = alpha
        self.density_norm = self.norm()

    def norm(self):
        """Normalised to 1 Msun."""
        c = (self.alpha + 2) / (
            self.m_up**(self.alpha + 2) - self.m_low**(self.alpha + 2))
        return c

    def pdf_n(self, m):
        return self.density_norm * m**self.alpha
    
    def pdf_mass(self, m):
        return self.density_norm * m**(self.alpha + 1)
    
    def cdf_n(self, m):
        cdf = self.density_norm / (self.alpha + 1) * (
            m**(self.alpha + 1) - self.m_low**(self.alpha + 1))
        return cdf
    
    def cdf_mass(self, m):
        cdf = self.density_norm / (self.alpha + 2) * (
            m**(self.alpha + 2) - self.m_low**(self.alpha + 2))
        return cdf
        
    
if __name__ == '__main__':
    imf = Salp55_IMF()
    
    masses = np.logspace(-1, 2)
    
    from matplotlib import pyplot as plt
    from matplotlib.colors import LogNorm

    plt.loglog(masses, imf.pdf_mass(masses))
    
    plt.figure()
    plt.loglog(masses, imf.cdf_mass(masses))
    
    mass_sn = 1 - imf.cdf_mass(8.0)
    n_sn = imf.cdf_n(imf.m_up) - imf.cdf_n(8.0)
    
    dummy_sfr = np.logspace(-2, 2)
    
    plt.figure()
    plt.loglog(dummy_sfr, 1 / (n_sn * dummy_sfr))
    plt.axhspan(100/3, 100, color='lime', alpha=0.5)
    plt.axvspan(1.3, 2.7, color='cyan', alpha=0.5)
    plt.ylabel('yr/SNe')
    plt.grid(visible=True, which='both')
    plt.xlabel(r"$SFR~(\rm M_\odot/yr)$")
    
    plt.figure()
    plt.loglog(dummy_sfr, n_sn * dummy_sfr)
    
    plt.figure()
    plt.loglog(dummy_sfr, 1 / (n_sn * dummy_sfr))
    plt.axhspan(100/3, 100, color='lime', alpha=0.5, label='MW SNR')
    plt.axvspan(1.3, 2.7, color='cyan', alpha=0.5, label='MW SFR')
    plt.ylabel('yr/SNe')
    plt.grid(visible=True, which='both')
    plt.xlabel(r"$SFR~(\rm M_\odot/yr)$")
    plt.legend()
    
    