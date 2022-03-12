#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 13:26:41 2021

@author: pablo
"""

from .GSSP import Pipe3Dssp as SSP_model
import numpy as np
from astropy import units as u
from photutils.centroids import centroid_com
from matplotlib import pyplot as plt


class SSP(SSP_model):
    """Recieves as input the SSP fits extension (output from Pipe3D)."""

    flux_norm = u.def_unit('10^{-16} erg/s/cm^2', 1e-16 * u.erg/u.s/u.cm**2)

    def __init__(self, SSP_extension, **kwargs):
        SSP_model.__init__(self)
        self.verbose = kwargs['verbose']
        self.SSP_data = SSP_extension
        self.read_ssp_data()

    def read_ssp_data(self):
        """todo."""
        if self.verbose:
            print('· [SSP MODULE] READING --> Header extensions')
        hdr = self.SSP_data.header
        hdr_keys = list(hdr.keys())
        self.SSP_variables = {}
        for ii, key in enumerate(hdr_keys):
            npos = key.find('_')
            if npos > -1:
                number = int(key[npos+1:])
                if number not in self.SSP_variables.keys():
                    self.SSP_variables[number] = {}
                if key.find('TYPE') == 0:
                    self.SSP_variables[number]['TYPE'] = hdr[key]
                elif key.find('UNITS') == 0:
                    self.SSP_variables[number]['UNITS'] = hdr[key]
                elif key.find('FILE') == 0:
                    self.SSP_variables[number]['FILE'] = hdr[key]
                elif key.find('DESC') == 0:
                    self.SSP_variables[number]['DESC'] = hdr[key]

    def get_normalization_flux(self):
        """todo."""
        self.median_flux = self.get_variable(3) * self.flux_norm
        self.median_flux_error = self.get_variable(4) * self.flux_norm
        return self.median_flux, self.median_flux_error

    def get_pseudo_v_band_map(self):
        """todo."""
        self.pseudo_v_band_map = self.get_variable(0) * self.flux_norm
        return self.pseudo_v_band_map

    def get_dezonification(self):
        """todo."""
        self.dezonification = self.get_variable(2)
        return self.dezonification

    def get_binning(self):
        """todo."""
        self.get_dezonification()
        self.binning = self.get_variable(1)
        # Bin == 0 accounts for the non-binned regions
        self.nbins = len(np.unique(self.binning)) - 1
        self.nspax_bin = np.zeros(self.nbins, dtype=int)
        self.bins_centroids = np.zeros((self.nbins, 2))
        for i in range(self.nbins):
            self.nspax_bin[i] = self.binning[self.binning == i+1].size
            xpos, ypos = np.where(self.binning == i+1)
            xmean = np.sum(self.dezonification[xpos, ypos] * xpos
                           )/self.dezonification[xpos, ypos].sum()
            ymean = np.sum(self.dezonification[xpos, ypos] * ypos
                           )/self.dezonification[xpos, ypos].sum()
            self.bins_centroids[i, :] = xmean, ymean
        return self.binning

    def get_Av_map(self):
        """todo."""
        self.av_map = self.get_variable(11)
        self.av_map_err = self.get_variable(12)
        return self.av_map, self.av_map_err

    def get_luminosity_w_age_map(self):
        """todo."""
        self.luminisity_w_age_map = self.get_variable(5)
        return self.luminisity_w_age_map

    def get_mass_w_age_map(self):
        """todo."""
        self.mass_w_age_map = self.get_variable(6)
        return self.mass_w_age_map

    def get_luminosity_w_met_map(self):
        """todo."""
        self.luminisity_w_age_map = self.get_variable(8)
        return self.luminisity_w_age_map

    def get_mass_w_met_map(self):
        """todo."""
        self.mass_w_age_map = self.get_variable(9)
        return self.mass_w_age_map

    def get_mass_to_light(self):
        """todo."""
        self.mass_to_light = self.get_variable(17)
        return self.mass_to_light

    def get_mass_dens_dust_corr(self):
        """todo."""
        self.mass_dens_dust_corr = self.get_variable(19)
        return self.mass_dens_dust_corr

    def get_mass_dens(self):
        """todo."""
        self.mass_dens_dust_corr = self.get_variable(18)
        return self.mass_dens_dust_corr

    def get_variable(self, var_number):
        """todo."""
        if self.verbose:
            print('· [SSP MODULE] READING')
            print('         FILE --> {}'.format(
                self.SSP_variables[var_number]['FILE']))
            print('         DESC --> {}'.format(
                self.SSP_variables[var_number]['DESC']))
            print('         UNITS --> {}'.format(
                self.SSP_variables[var_number]['UNITS']))
        return self.SSP_data.data[var_number]

    def get_redshift(self, redshift):
        """todo."""
        # FIXME
        self.redshift = redshift

    def bin_map(self, _map, mode='sum', weight=False):
        """todo."""
        try:
            self.binning
        except Exception:  # FIXME: Choose appropiate error exception
            self.get_binning()
        try:
            _map.unit
            unit = _map.unit
        except Exception:  # FIXME: Choose appropiate error exception
            unit = 1
        if mode == 'sum':
            func = np.sum
        elif mode == 'mean':
            func = np.mean
        elif mode == 'std':
            func = np.std
        binned_map = np.zeros(self.nbins) * unit
        for ii in np.arange(1, self.nbins+1):
            bin_ii = self.binning == ii
            if weight:
                beta = self.coadded_spectral_empirical_correlation(
                    bin_ii[bin_ii].size)
            else:
                beta = 1
            binned_map[ii-1] = func(_map[bin_ii]) * beta**2
        return binned_map

    def coadded_spectral_empirical_correlation(self, n_spaxels):
        """todo."""
        # alpha = {'V500': 1.10, 'V1200': 1.08, 'COMB': 1.08}
        beta = 1 + 1.10 * np.log10(n_spaxels)
        return beta

    def compute_com(self):
        """todo."""
        try:
            self.median_flux
        except Exception:
            self.get_normalization_flux()
        com = centroid_com(self.median_flux)
        self.center_of_mass = com
        return com

    def plot_binned_map(self, _binned_map, **kwargs):
        """todo."""
        try:
            unit = _binned_map.unit
        except Exception:
            unit = 1 * u.m/u.m
        _map = np.zeros_like(self.binning) * unit
        _map[:, :] = np.nan
        for i in range(self.nbins):
            mask = self.binning == i+1
            _map[mask] = _binned_map[i]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        mappable = ax.imshow(_map.value, interpolation='none', origin='lower',
                             aspect='auto',
                             vmin=np.nanpercentile(_map.value, 5),
                             vmax=np.nanpercentile(_map.value, 95),
                             **kwargs)
        cbar = plt.colorbar(mappable, ax=ax)
        plt.show()
        plt.close()
        return fig, cbar

    def binned_to_map(self, _binned_map):
        """todo."""
        try:
            unit = _binned_map.unit
        except Exception:
            unit = 1 * u.m/u.m
        _map = np.zeros_like(self.binning) * unit
        _map[:, :] = np.nan
        for i in range(self.nbins):
            mask = self.binning == i+1
            _map[mask] = _binned_map[i]
        return _map


if __name__ == '__main__':

    pass

#  Mr. Krtxo
