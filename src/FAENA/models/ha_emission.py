#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 09:46:37 2022

@author: pablo
"""

import h5py
import numpy as np
from matplotlib import pyplot as plt
from astropy import units as u
from scipy.interpolate import interp1d

# ----------------------------------------------------------------------------
default_lookbacktime = np.array([
    1.00000e-03, 3.00000e-03, 3.98110e-03, 5.62300e-03, 8.91300e-03,
    1.00000e-02, 1.25900e-02, 1.41300e-02, 1.77800e-02, 1.99500e-02,
    2.51190e-02, 3.16200e-02, 3.98110e-02, 5.62300e-02, 6.30000e-02,
    6.31000e-02, 7.08000e-02, 1.00000e-01, 1.12200e-01, 1.25900e-01,
    1.58500e-01, 1.99500e-01, 2.81800e-01, 3.54800e-01, 5.01200e-01,
    7.07900e-01, 8.91300e-01, 1.12200e+00, 1.25890e+00, 1.41250e+00,
    1.99530e+00, 2.51190e+00, 3.54810e+00, 4.46680e+00, 6.30960e+00,
    7.94330e+00, 1.00000e+01, 1.25893e+01, 1.41254e+01])
default_lookbacktime_edges = np.hstack(
            (default_lookbacktime[0] -
             (default_lookbacktime[1]-default_lookbacktime[0])/2,
             default_lookbacktime[1:] - np.diff(default_lookbacktime)/2,
             default_lookbacktime[-1] +
             (default_lookbacktime[-1]-default_lookbacktime[-2])/2))
ddefault_time = np.diff(default_lookbacktime_edges)


class LuminosityModel(object):
    """..."""

    def luminosity(self, masses, tau_v):
        """..."""
        if len(masses.shape) > 1:
            lum = (np.exp(- self.eta * tau_v)
                   * (np.nansum(self.coeffs[:, np.newaxis] * masses, axis=0))
                   )
        else:
            lum = (np.exp(- self.eta * tau_v)
                   * (np.nansum(self.coeffs * masses, axis=0))
                   )
        return lum

    def interp_coeffs(self, newlookbacktime, plot=False):
        """..."""
        interplookbacktime = np.hstack(
            (newlookbacktime[0] -
             (newlookbacktime[1]-newlookbacktime[0])/2,
             newlookbacktime[1:] - np.diff(newlookbacktime)/2,
             newlookbacktime[-1] +
             (newlookbacktime[-1]-newlookbacktime[-2])/2))
        cumcoeffs = np.interp(interplookbacktime,
                              default_lookbacktime,
                              np.cumsum(self.coeffs*ddefault_time))
        coeffs = np.diff(cumcoeffs) / np.diff(interplookbacktime)
        print('Coefficients interpolated')
        if plot:
            fig = plt.figure()
            plt.plot(default_lookbacktime, self.coeffs, 'k')
            plt.plot(newlookbacktime, coeffs, 'r')
            plt.xscale('log')
            plt.yscale('log')
            plt.ylim(self.coeffs.max()*1e-4, self.coeffs.max()*10)
            plt.show()
            plt.close(fig)
        self.coeffs = coeffs


class SF_model(LuminosityModel):
    """Step function model for predicting Ha emission.

    C(t) = Lambda / tau  for t < tau

    Attributes
    ----------
    - eta: (float)
        Proportionality extinction coefficient (A_v-gas = eta * A_v-stars)
    - normalization: (float)
        Energy budget in log(erg/Msun)
    - timescale: (float)
        Characteristica age for the ionising population in Gyr.
    """

    def __init__(self, **kwargs):
        """..."""
        self.eta = kwargs.get('eta', None)
        self.normalization = kwargs.get('normalization', None)
        self.timescale = kwargs.get('timescale', None)

    def get_coeffs(self, normalization, timescale,
                   lookbacktime=default_lookbacktime):
        """...

        - timescale: maximum age for a stellar population to contribute to Ha
        emission
        """
        coeffs = np.zeros_like(lookbacktime)
        coeff_value = 10**normalization / (timescale * 1e9)
        coeffs[:] = coeff_value
        coeffs[lookbacktime > timescale] = 0
        return coeffs


class SF_PL_model(LuminosityModel):
    """..."""

    def __init__(self):
        """..."""
        self.eta = None
        self.normalization = None
        # omega = L_old / L_tot
        self.omega = None
        self.tau_young = None

    def get_coeffs(self, normalization, tau_young, omega,
                   lookbacktime=default_lookbacktime):
        """..."""
        tau_young_s = tau_young * u.Gyr.to('s')
        lambda_young = normalization/tau_young_s

        coeffs = np.empty_like(lookbacktime)
        young = lookbacktime <= tau_young
        coeffs[young] = lambda_young
        old = lookbacktime > tau_young
        coeffs[old] = lambda_young * (
            lookbacktime[old]/tau_young)**(-1/omega)

        return coeffs


def to_gaussian_profile(luminosity, wave, c_wave=6563., sigma=3,
                        vel=False):
    """..."""
    if vel:
        sigma = sigma / 3e5 * c_wave

    g = (luminosity * 1/np.sqrt(2 * np.pi * sigma**2)
         * np.exp(- 0.5 * (wave - c_wave)**2 / sigma**2))
    return g


# =============================================================================
# Ha emission model based on Mollá+09
# =============================================================================

class HaModel_Molla(object):
    """Balmer lines emission model."""

    def __init__(self):
        """..."""

    @staticmethod
    def electronic_temp(z):
        """Interpolate metallicity to electronic temperature.

        Data selected from Table 6 Molla+09.
        """
        te_Z = np.array([[0.0001, 19950],
                         [0.0004, 15850],
                         [0.004, 10000],
                         [0.008, 7940],
                         [0.02, 6310],
                         [0.05, 3160]])
        return np.interp(z, te_Z[:, 0], te_Z[:, 1])

    @staticmethod
    def recomb_coeff(e_temp):
        """Interpolate Case-B recombination coefficient (cm^3/s).

        Data selected from Ferland 1980
        """
        if e_temp <= 2.6e4:
            alpha_b = 2.90e-10 * e_temp**-0.77
        else:
            alpha_b = 1.31e-8 * e_temp**-1.13
        return alpha_b

    @staticmethod
    def effective_emission_coeff(e_temp):
        """Interpolate effective emission coefficient (erg/s/cm^3).

        Data selected from Ferland 1980"""
        if e_temp <= 2.6e4:
            j_hb = 2.53e-22 * e_temp**-0.833
        else:
            j_hb = 1.12e-20 * e_temp**-1.20
        return j_hb / 4 / np.pi

    @staticmethod
    def balmer_ratios(e_temp):
        """Compute the Balmer ratios for alpha, gamma, delta and epsilon lines.
        Data selected from Osterbrock 1989."""
        # T_e, N_e, j_alpha/j_beta, j_gamma/j_beta, j_delta/j_beta, j_eps/j_bet

        # case_b_table = np.array([
        #         [5000, 1e2, 3.04, 0.458, 0.251, 0.154],
        #         [5000, 1e4, 3.00, 0.460, 0.253, 0.155],
        #         [10000, 1e2, 2.86, 0.468, 0.259, 0.159],
        #         [10000, 1e4, 2.85, 0.469, 0.260, 0.159],
        #         [20000, 1e2, 2.75, 0.475, 0.264, 0.163],
        #         [20000, 1e4, 2.74, 0.476, 0.264, 0.163]
        #         ])
        t = np.array([5000., 10000., 20000.])
        case_b_table = np.array([
                [3.04, 0.458, 0.251, 0.154],
                [2.86, 0.468, 0.259, 0.159],
                [2.75, 0.475, 0.264, 0.163],
                ])
        interpolator = interp1d(x=t, y=case_b_table, axis=0,
                                bounds_error=False,
                                fill_value=([3.04, 0.458, 0.251, 0.154],
                                            [2.75, 0.475, 0.264, 0.163]))
        return interpolator(e_temp)

    def luminosity(self, q_ion, met):
        """..."""
        e_t = self.electronic_temp(met)
        alpha_b = self.recomb_coeff(e_t)
        j_hb = self.effective_emission_coeff(e_t)
        balmer_rat = self.balmer_ratios(e_t)
        l_hb = q_ion * j_hb / alpha_b
        l_balmer = l_hb * balmer_rat
        return l_hb, l_balmer
