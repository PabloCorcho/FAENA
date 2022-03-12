#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 13:22:55 2021

@author: pablo
"""

import numpy as np
from astropy import units as u
from specutils import Spectrum1D
from specutils.manipulation import FluxConservingResampler
from .utils import Cosmology, extinction_law


class SFH(object):

    def __init__(self, SFH_extension, **kwargs):
        self.verbose = kwargs['verbose']
        self.SFH_data = SFH_extension
        if 'SSP' in kwargs.keys():
            self.SSP = kwargs['SSP']
        self.read_sfh_data()
# =============================================================================
#   LOAD DATA
# =============================================================================

    def read_sfh_data(self):
        if self.verbose:
            print('· [SFH MODULE] READING --> Header extensions')
        hdr = self.SFH_data.header
        hdr_keys = list(hdr.keys())
        self.SFH_variables = {}

        self.age_met_w = []
        self.hdr_age_met = []
        self.age_met_we = []
        self.age_w = []
        self.age_we = []
        self.met_w = []
        self.met_we = []

        for ii, key in enumerate(hdr_keys):
            npos = key.find('_')
            if npos > -1:
                number = int(key[npos+1:])
                if number not in self.SFH_variables.keys():
                    self.SFH_variables[number] = {}
                if key.find('TYPE') == 0:
                    self.SFH_variables[number]['TYPE'] = hdr[key]
                elif key.find('UNITS') == 0:
                    self.SFH_variables[number]['UNITS'] = hdr[key]
                elif key.find('FILE') == 0:
                    self.SFH_variables[number]['FILE'] = hdr[key]
                elif key.find('DESC') == 0:
                    self.SFH_variables[number]['DESC'] = hdr[key]
                    if ('Luminosity' in hdr[key]):
                        if 'age-met' in hdr[key]:
                            self.age_met_w.append(number)
                            age_met = hdr[key][hdr[key].find(
                                'age-met ')+8:hdr[key].find(' SSP')]
                            self.hdr_age_met.append(age_met.split('-'))
                        elif ' age ' in hdr[key]:
                            self.age_w.append(number)
                        elif ' met ' in hdr[key]:
                            self.met_w.append(number)
                    elif ('Error' in hdr[key]):
                        if 'age-met' in hdr[key]:
                            self.age_met_we.append(number)
                        elif ' age ' in hdr[key]:
                            self.age_we.append(number)
                        elif ' met ' in hdr[key]:
                            self.met_we.append(number)

    def load_SSP_weights(self, mode='individual'):

        ssp_weights = self.SFH_data.data
        if mode == 'individual':
            # This weights are not well sorted in age
            return (ssp_weights[self.age_met_w, :, :],
                    ssp_weights[self.age_met_we, :, :])
        if mode == 'age':
            # This weights are sorted in age
            return (ssp_weights[self.age_w, :, :],
                    ssp_weights[self.age_we, :, :])
        if mode == 'met':
            return (ssp_weights[self.met_w, :, :],
                    ssp_weights[self.met_we, :, :])

# =============================================================================
#   COMPUTATIONAL METHODS
# =============================================================================
    def compute_ssp_masses(self, **kwargs):
        """
        This method computes the stellar mass corresponding to each SSP used
        in the fit.
        ----------------------------------------------------------------------
        Input
            - stellar_death (bool) (default=True): If True, the SSP stellar
            masses are corrected for stellar mass loss.
        Output
            - self.ssp_masses (np.array(156)): Individual stellar masses in
            solar units.
            - self.ssp_masses_err (np.array(156)): Errors.
        """
        dezonification = kwargs.get('dezonification', False)
        stellar_death = kwargs.get('stellar_death', False)
        # This weights are sorted in age
        if not hasattr(self.SSP, 'median_flux'):
            self.SSP.get_normalization_flux()
        flux, flux_err = self.SSP.median_flux, self.SSP.median_flux_error
        if dezonification:
            if not hasattr(self.SSP, 'dezonification'):
                self.SSP.get_dezonification()
            dezon = self.SSP.dezonification
        else:
            dezon = np.ones_like(flux.value)
        if not hasattr(self.SSP, 'ext_norm_5500'):
            self.compute_extinction_normalization()
        ext_norm_5500 = self.ext_norm_5500
        av_map_err = self.SSP.av_map_err

        # Mass to luminosity ratio at 5500 AA and stellar mass loss correction
        ssp_mass_to_lum = self.SSP.ssp_present_mass_lum_ratio()
        if stellar_death:
            ssp_alive_stellar_mass = self.SSP.ssp_alive_stellar_mass()
        else:
            ssp_alive_stellar_mass = np.ones_like(ssp_mass_to_lum.value)
        # Individual weights for each SSP
        ssp_weights, ssp_weights_err = self.load_SSP_weights(mode='individual')
        # Conversion to luminosity
        self.lum_distance = Cosmology.luminosity_distance(self.SSP.redshift
                                                          ).to('cm')
        median_luminosity = 4 * np.pi * self.lum_distance**2 * flux
        median_luminosity_err = 4 * np.pi * self.lum_distance**2 * flux_err
        # SSP_MASS = weights * L_V * e^(tau_V) * M/L_V * M_ALIVE/M_TOT * DEZON
        self.ssp_masses = (
                ssp_weights
                * median_luminosity[np.newaxis, :, :]  # L_5500
                * 1/ext_norm_5500[np.newaxis, :, :]  # e^tau_5500
                * ssp_mass_to_lum[:, np.newaxis, np.newaxis]  # Upsilon
                * ssp_alive_stellar_mass[:, np.newaxis, np.newaxis]
                * dezon[np.newaxis, :, :]
                            )
        if ssp_weights_err.shape != ssp_weights.shape:
            print('[SFH] · Warning!  SSP Weights errors are not provided')
            self.ssp_masses_err = np.full_like(self.ssp_masses,
                                               fill_value=np.nan)
        else:
            self.ssp_masses_err = np.sqrt(
                ssp_weights_err**2 * median_luminosity[np.newaxis, :, :]**2
                + (0.4 * np.log(10) * av_map_err[np.newaxis, :, :])**2
                * ssp_weights**2 * median_luminosity[np.newaxis, :, :]**2
                + median_luminosity_err[np.newaxis, :, :]**2 * ssp_weights**2
                            ) * (
                ssp_mass_to_lum[:, np.newaxis, np.newaxis]
                * ssp_alive_stellar_mass[:, np.newaxis, np.newaxis]
                * dezon[np.newaxis, :, :]
                * 1/ext_norm_5500[np.newaxis, :, :])

        self.ssp_masses = self.ssp_masses.to('Msun')
        self.ssp_masses_err = self.ssp_masses_err.to('Msun')

    def compute_ssp_stellar_mass_surface_density(self, stellar_death=False,
                                                 dezonification=False):
        """
        This method computes the stellar mass corresponding to each SSP used
        in the fit.
        ----------------------------------------------------------------------
        Input
            - stellar_death (bool) (default=True): If True, the SSP stellar
            masses are corrected for stellar mass loss.
        Output
            - self.ssp_masses (np.array(156)): Individual stellar masses in
            solar units.
            - self.ssp_masses_err (np.array(156)): Errors.
        """
        # Observed flux used for normalization
        if not hasattr(self.SSP, 'median_flux'):
            self.SSP.get_normalization_flux()
        flux, flux_err = self.SSP.median_flux, self.SSP.median_flux_error
        if dezonification:
            if not hasattr(self.SSP, 'dezonification'):
                self.SSP.get_dezonification()
            dezon = self.SSP.dezonification
        else:
            dezon = np.ones_like(flux)
        if not hasattr(self.SSP, 'ext_norm_5500'):
            self.compute_extinction_normalization()
        ext_norm_5500 = self.ext_norm_5500
        av_map_err = self.SSP.av_map_err

        # Mass to luminosity ratio at 5500 AA and stellar mass loss correction
        ssp_mass_to_lum = self.SSP.ssp_present_mass_lum_ratio()
        if stellar_death:
            ssp_alive_stellar_mass = self.SSP.ssp_alive_stellar_mass()
        else:
            ssp_alive_stellar_mass = np.ones_like(ssp_mass_to_lum)
        # Individual weights for each SSP
        ssp_weights, ssp_weights_err = self.load_SSP_weights()
        # surface brightness
        # FIXME: Only valid for CALIFA
        surface_brightness = flux / (np.deg2rad(1/3600))**2 
        # SSP_MASS = weights * L_V * e^(tau_V) * M/L_V * M_ALIVE/M_TOT * DEZON
        self.ssp_surf_density = (
                ssp_weights
                * surface_brightness[np.newaxis, :, :]  # F_5500
                * 1/ext_norm_5500[np.newaxis, :, :]  # e^tau_5500
                * ssp_mass_to_lum[:, np.newaxis, np.newaxis]  # Upsilon
                * ssp_alive_stellar_mass[:, np.newaxis, np.newaxis]
                * dezon[np.newaxis, :, :]
                            )
        self.ssp_surf_density = self.ssp_surf_density.to(u.Msun/u.pc**2)

    def compute_SFH(self, **kwargs):
        """
        This method reconstructs the star formation history for each spaxel of
        the galaxy
        -----------------------------------------------------------------------
        Input
            - stellar_death (bool) (default=True): If True, the SSP stellar
            masses are corrected for stellar mass loss.
        Output
            - self.mass_formed_at_t: Stellar mass formed at each lookback-time
            per spaxel
                np.array(39, x_spax, y_spax)
            - self.star_formation_history: M_spax(t) = int( SFH_spax * dt )
                np.array(39, x_spax, y_spax)
            - self.integrated_star_formation_history: M(t)
                np.array(39)
        """
        try:
            self.ssp_masses
        except:
            self.compute_ssp_masses(**kwargs)
        # The SSP models are unsorted in age
        ages = self.SSP.ssp_ages(mode='individual')
        ages = ages.reshape((39, 4))
        sort_ages = np.argsort(ages[:, 0])
        sort_ages = np.argsort(ages)
        # (age, metallicity, spax_y, spax_x) grid
        sorted_ssp_masses = self.ssp_masses.reshape(
            39, 4,
            self.ssp_masses.shape[1],
            self.ssp_masses.shape[2])[sort_ages, :, :, :]
        sorted_ssp_masses_err = self.ssp_masses_err.reshape(
            39, 4,
            self.ssp_masses_err.shape[1],
            self.ssp_masses_err.shape[2])[sort_ages, :, :, :]
        # Summation over all metallicities --> (age, spax_y, spax_x)
        self.total_ssp_mass = np.sum(sorted_ssp_masses, axis=1)
        self.total_ssp_mass_err = np.sqrt(np.sum(sorted_ssp_masses**2, axis=1))
        # The grid now is arranged from old to young ssp's
        self.total_ssp_mass = self.total_ssp_mass[::-1, :, :]
        self.total_ssp_mass_err = self.total_ssp_mass_err[::-1, :, :]

        self.star_formation_history = np.cumsum(
            self.total_ssp_mass, axis=0)
        self.star_formation_history_err = np.sqrt(np.cumsum(
            self.total_ssp_mass**2, axis=0))

        self.integrated_star_formation_history = np.nansum(
            self.total_ssp_mass, axis=(1, 2))
        self.integrated_star_formation_history = np.cumsum(
            self.integrated_star_formation_history)
        # sorted from present to origin
        self.integrated_star_formation_history =\
            self.integrated_star_formation_history[::-1]
        self.star_formation_history = self.star_formation_history[::-1, :, :]
        self.total_ssp_mass = self.total_ssp_mass[::-1, :, :]
        self.total_ssp_mass_err = self.total_ssp_mass_err[::-1, :, :]
        self.lookbacktime = np.sort(np.unique(ages))

    def compute_binned_SFH(self, **kwargs):
        """
        This method reconstructs the star formation history for each bin of
        the galaxy
        """
        try:
            self.ssp_masses
        except Exception:  # FIXME
            self.compute_ssp_masses(**kwargs)
        if not hasattr(self.SSP, 'binning'):
            self.SSP.get_binning()
        binning = self.SSP.binning

        # The SSP models are unsorted in age
        ages = self.SSP.ssp_ages(mode='individual')
        ages = ages.reshape((39, 4))
        sort_ages = np.argsort(ages[:, 0])
        mets = np.unique(self.SSP.ssp_met)
        # (age, metallicity, spax_y, spax_x) grid
        sorted_ssp_masses = self.ssp_masses.reshape(
            (39, 4,
            self.ssp_masses.shape[1],
            self.ssp_masses.shape[2]))[sort_ages, :, :, :]
        sorted_ssp_masses_err = self.ssp_masses_err.reshape(
            (39, 4,
            self.ssp_masses_err.shape[1],
            self.ssp_masses_err.shape[2]))[sort_ages, :, :, :]
        # Mass-weighted Metallicity history
        self.metallicity_history = np.sum(
            sorted_ssp_masses * mets[np.newaxis, :, np.newaxis, np.newaxis],
            axis=1)/np.sum(sorted_ssp_masses, axis=1)
        # Summation over all metallicities --> (age, spax_y, spax_x)

        self.total_ssp_mass = np.sum(sorted_ssp_masses, axis=1)
        self.total_ssp_mass_err = np.sqrt(np.sum(sorted_ssp_masses**2, axis=1))
        # The grid now is arranged from old to young ssp's
        self.total_ssp_mass = self.total_ssp_mass[::-1, :, :]
        self.total_ssp_mass_err = self.total_ssp_mass_err[::-1, :, :]
        self.metallicity_history = self.metallicity_history[::-1, :, :]
        # Binning masses
        self.total_ssp_mass_binned = np.zeros((self.total_ssp_mass.shape[0],
                                               self.SSP.nbins)
                                              ) * self.total_ssp_mass.unit
        self.total_ssp_mass_binned_err = np.zeros(
            (self.total_ssp_mass.shape[0],
             self.SSP.nbins)
            ) * self.total_ssp_mass.unit
        self.metallicity_history_binned = np.zeros_like(
            self.total_ssp_mass_binned.value)

        for ii in np.arange(1, self.SSP.nbins+1):
            bin_ii = binning == ii
            self.total_ssp_mass_binned[:, ii-1] = np.sum(
                self.total_ssp_mass[:, bin_ii], axis=(1))
            self.total_ssp_mass_binned_err[:, ii-1] = np.sqrt(np.sum(
                self.total_ssp_mass_err[:, bin_ii]**2, axis=(1)
                )
                # * self.SSP.coadded_spectral_empirical_correlation(
                # bin_ii[bin_ii].size)
                )
            self.metallicity_history_binned[:, ii-1] = np.nanmean(
                self.metallicity_history[:, bin_ii], axis=1)
        # Cumulative history
        self.binned_star_formation_history = np.cumsum(
            self.total_ssp_mass_binned, axis=0)
        self.binned_star_formation_history_err = np.sqrt(np.cumsum(
            self.total_ssp_mass_binned_err**2, axis=0))
        # sorted from present to origin
        self.binned_star_formation_history =\
            self.binned_star_formation_history[::-1, :]
        self.metallicity_history_binned = self.metallicity_history_binned[::-1,
                                                                          :]
        self.lookbacktime = np.sort(np.unique(ages))

    def compute_synthetic_cube(self):
        """
        This method builds the synthetic spectrum for each spaxel after the
        dezonification of each bin.
        Namely:
        F_lambda_spaxel = F_5500_spax * dezon_spax * exp(-(tau_lambda-tau_5500)
        * sum( weight_i * F_lambda_SSP_i )
        """
        if self.verbose:
            print('· [SFH MODULE] COMPUTING --> SYNTHETIC CUBE')
        if not hasattr(self.SSP, 'median_flux'):
            self.SSP.get_normalization_flux()
        flux, flux_err = self.SSP.median_flux, self.SSP.median_flux_error
        if not hasattr(self.SSP, 'dezonification'):
            self.SSP.get_dezonification()
        dezon = self.SSP.dezonification
        if not hasattr(self.SSP, 'av_map'):
            self.SSP.get_Av_map()
        av = self.SSP.av_map

        ssp_weights, _ = self.load_SSP_weights()

        # SSP luminosity normalization and dezonification
        self.ssp_norm = (flux[np.newaxis, :, :] *
                         dezon[np.newaxis, :, :] * ssp_weights)

        self.synth_cube = np.zeros((self.SSP.ssp_SED.shape[1],
                                   self.ssp_norm.shape[1],
                                   self.ssp_norm.shape[2])
                                   ) * self.ssp_norm.unit / u.angstrom
        self.extinction_cube = np.zeros((self.SSP.ssp_wl.size, av.shape[0],
                                         av.shape[1]))
        for ith in range(self.ssp_norm.shape[1]):
            for jth in range(self.ssp_norm.shape[2]):
                self.extinction_cube[:, ith, jth] = 10 ** (
                    -0.4 * extinction_law(self.SSP.ssp_wl,
                                          av[ith, jth],
                                          r_v=3.1)
                                                  )
                self.extinction_cube[:, ith, jth] /= np.interp(
                    5500*u.angstrom,
                    self.SSP.ssp_wl,
                    self.extinction_cube[:, ith, jth])

                synth_spectra = np.sum(
                    self.ssp_norm[:, ith, jth, np.newaxis]
                    * self.SSP.ssp_SED[:, :], axis=0
                    ) * self.extinction_cube[:, ith, jth] / u.angstrom
                self.synth_cube[:, ith, jth] = synth_spectra

    def compute_synthetic_binned_spectra(self):
        """
        This method builds the synthetic spectrum for each bin used during the
        Pipe3D fit.
        Namely:
        F_lambda = F_5500 * exp(-(tau_lambda-tau_5500))
        * sum( weight_i * F_lambda_SSP_i )
        """
        if self.verbose:
            print('· [SFH MODULE] COMPUTING --> SYNTHETIC BINNED SPECTRA')
        # Bin map
        if not hasattr(self.SSP, 'binning'):
            self.SSP.get_binning()
        binning = self.SSP.binning
        # Normalizing flux at 5500 AA
        if not hasattr(self.SSP, 'median_flux'):
            self.SSP.get_normalization_flux()
        flux, flux_err = self.SSP.median_flux, self.SSP.median_flux_error
        # Extinction (A_v) map
        if not hasattr(self.SSP, 'av_map'):
            self.SSP.get_Av_map()
        av = self.SSP.av_map
        ssp_weights, ssp_weights_err = self.load_SSP_weights()
        # Empty set
        self.synth_binned_spectra = np.zeros((self.SSP.ssp_SED.shape[1],
                                              self.SSP.nbins)
                                             ) * flux.unit / u.angstrom
        # Array for different extinctions
        self.extinction = np.zeros((self.SSP.ssp_wl.size, self.SSP.nbins))
        # Loop over each bin
        for ii in np.arange(1, self.SSP.nbins+1):
            bin_ii = binning == ii
            weights_bin_ii = np.mean(ssp_weights[:, bin_ii], axis=1)
            flux_bin_ii = np.sum(flux[bin_ii])
            av_bin_ii = np.mean(av[bin_ii])
            self.extinction[:, ii-1] = 10**(-0.4*extinction_law(
                                            self.SSP.ssp_wl, av_bin_ii,
                                            r_v=3.1))
            # The attenuation curve is normalized to the V band
            self.extinction[:, ii-1] /= np.interp(5500*u.angstrom,
                                                  self.SSP.ssp_wl,
                                                  self.extinction[:, ii-1])
            # Synthetic spectra for bin ii
            synth_spectra = np.sum(flux_bin_ii * weights_bin_ii[:, np.newaxis]
                                   * self.SSP.ssp_SED[:, :], axis=0
                                   ) * self.extinction[:, ii-1] / u.angstrom
            self.synth_binned_spectra[:, ii-1] = synth_spectra

    def compute_extinction_normalization(self):
        """
        This method computes: e^{tau_5500}, or equivalently 10**{0.4 A_5500},
        accounting for the normalization of the dust attenuation curve for
        each spaxel.
        """
        if not hasattr(self.SSP, 'av_map'):
            self.SSP.get_Av_map()
        av = self.SSP.av_map
        # av_err = self.SSP.av_map
        self.ext_norm_5500 = np.zeros_like(av)
        for ith in range(self.ext_norm_5500.shape[0]):
            for jth in range(self.ext_norm_5500.shape[1]):
                self.extinction = 10 ** (-0.4 * extinction_law(self.SSP.ssp_wl,
                                                               av[ith, jth],
                                                               r_v=3.1)
                                         )
                self.ext_norm_5500[ith, jth] = np.interp(5500*u.angstrom,
                                                         self.SSP.ssp_wl,
                                                         self.extinction)

    def compute_luminosity_weighted_age(self):
        ages = self.SSP.ssp_ages(mode='individual')
        ssp_weights, _ = self.load_SSP_weights(mode='individual')
        lum_weighted_age = np.sum(
            np.log10(ages[:, np.newaxis, np.newaxis]/ages.unit)
            * ssp_weights, axis=0)
        # TODO: Implement uncertainties
        return lum_weighted_age

    def compute_mass_weighted_age(self):
        ages = self.SSP.ssp_ages(mode='individual')
        ssp_mass_to_lum = self.SSP.ssp_present_mass_lum_ratio(
            mode='individual')
        ssp_weights, _ = self.load_SSP_weights()
        all_weights = np.sum(ssp_weights, axis=0)
        mask = all_weights == 0
        mass_weighted_age = np.sum(
            np.log10(ages[:, np.newaxis, np.newaxis]/ages.unit)
            * ssp_weights*ssp_mass_to_lum[:, np.newaxis, np.newaxis], axis=0
            ) / np.sum(
                ssp_weights*ssp_mass_to_lum[:, np.newaxis, np.newaxis], axis=0)
        mass_weighted_age[mask] = 0
        # TODO: Implement uncertainties
        return mass_weighted_age

    def compute_luminosity_weighted_met(self):
        met = self.SSP.ssp_met
        ssp_weights, _ = self.load_SSP_weights(mode='individual')
        lum_weighted_met = np.sum(
            np.log10(met[:, np.newaxis, np.newaxis]/0.02) * ssp_weights,
            axis=0)
        # TODO: Implement uncertainties
        return lum_weighted_met

    def compute_mass_weighted_met(self):
        met = self.SSP.ssp_met
        ssp_mass_to_lum = self.SSP.ssp_present_mass_lum_ratio(
            mode='individual')
        ssp_weights, _ = self.load_SSP_weights()
        all_weights = np.sum(ssp_weights, axis=0)
        mask = all_weights == 0
        mass_weighted_met = np.sum(
            np.log10(met[:, np.newaxis, np.newaxis]/0.02)
            * ssp_weights*ssp_mass_to_lum[:, np.newaxis, np.newaxis], axis=0
            )/np.sum(ssp_weights*ssp_mass_to_lum[:, np.newaxis, np.newaxis],
                     axis=0)
        mass_weighted_met[mask] = 0
        # TODO: Implement uncertainties
        return mass_weighted_met

    def compute_mass_to_light_ratio(self):
        if not hasattr(self.SSP, 'ext_norm_5500'):
            self.compute_extinction_normalization()
        ext_norm_5500 = self.ext_norm_5500
        ssp_mass_to_lum = self.SSP.ssp_present_mass_lum_ratio().to(
            u.Msun/u.Lsun)
        ssp_mass_to_lum /= self.SSP.wavenorm
        # ssp_alive_stellar_mass = self.SSP.ssp_alive_stellar_mass()
        # ssp_mass_to_lum /= ssp_alive_stellar_mass
        ssp_weights, _ = self.load_SSP_weights(mode='individual')
        mass_to_lum = 1/ext_norm_5500 * np.sum(
            ssp_mass_to_lum[:, np.newaxis, np.newaxis] * ssp_weights, axis=0)
        # TODO: Implement uncertainties
        return np.log10(mass_to_lum.value)

    def interpolate_spec(self, new_wave, wave, flux):
        input_spectra = Spectrum1D(flux=flux, spectral_axis=wave)
        fluxc_resample = FluxConservingResampler(
            extrapolation_treatment='nan_fill')
        output_spectrum1D = fluxc_resample(input_spectra, new_wave)
        return output_spectrum1D

# =============================================================================
# Mr. Krtxo
# =============================================================================
