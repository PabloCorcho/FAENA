#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 16:37:32 2022

@author: pablo
"""

from astropy.io import fits
from astropy.wcs import WCS

import numpy as np

class Spectra(object):
    """..."""

    def __init__(self, wavelength, flux, flux_error, **kwargs):
        self.wavelength, self.flux, self.flux_error = (
            wavelength, flux, flux_error)
        # Redshift and rest frame
        self.redshift = kwargs.get('redshift', None)
        self.rest_frame = False

    def to_rest_frame(self, verbose=False):
        """Set wavelength array to rest frame."""
        if self.redshift is not None:
            self.wavelength /= 1 + self.redshift
            if verbose:
                print("[Spectra] Wavelength array to rest-frame (z={:.4})"
                      .format(self.redshift))
            self.rest_frame = True
        else:
            if verbose:
                print("[Spectra] WARNING: Redshift not provided.")


class VipersSpectra(Spectra):
    """VIMOS VLT Spectra."""

    def __init__(self, path_to_file):
        hdul = fits.open(path_to_file)
        super().__init__(
            wavelength=hdul[1].data["waves"],
            flux=hdul[1].data["fluxes"],
            flux_error=hdul[1].data["noise"],
            redshift=hdul[1].header['redshift'])


class GAMASpectra(Spectra):
    """GAMA (AAT+SDSS+UKIRT) spectra."""

    def __init__(self, path_to_file):
        hdul = fits.open(path_to_file)
        self.header = hdul[0].header
        
        logwave = False

        if 'SURVEY' in hdul[0].header.keys():
            n_pixels = hdul[0].header['NAXIS1']
            crval = hdul[0].header['CRVAL1']
            crpix = hdul[0].header['CRPIX1']
            self.survey = hdul[0].header['SURVEY']
            if self.survey == 'MGC':
                flux = hdul[0].data[0]
                flux_err = hdul[0].data[1]
                cdel = hdul[0].header['CD1_1']
            elif self.survey == 'SDSS':
                flux = hdul[0].data[0]
                flux_err = hdul[0].data[1]**0.5
                cdel = hdul[0].header['CD1_1']
                logwave = True
            elif self.survey.find('2dF') > -1:
                flux = hdul[0].data[0]
                flux_err = hdul[0].data[1]**0.5
                cdel = hdul[0].header['CDELT1']
            elif self.survey == 'WiggleZ':
                flux = hdul[0].data
                flux_err = hdul[1].data**0.5
                cdel = hdul[0].header['CDELT1']
            elif self.survey == '2QZ':
                flux = hdul[0].data
                flux_err = hdul[2].data**0.5
                cdel = hdul[0].header['CD1_1']
            elif self.survey == '6dFGS':
                flux = hdul[0].data[0]
                flux_err = hdul[0].data[1]**0.5
                cdel = hdul[0].header['CD1_1']
            elif self.survey == '2SLAQ-QSO':
                flux = hdul[0].data
                flux_err = hdul[1].data**0.5
                cdel = hdul[0].header['CDELT1']
            else:
                cdel = hdul[0].header['AAA']

            # flux_err = hdul[0].data[1]**0.5
        else:
            self.survey = 'GAMA'
            if hdul[0].header['ORIGIN'] == 'GAMA':
                n_pixels = hdul[0].header['NAXIS1']
                crval = hdul[0].header['CRVAL1']
                crpix = hdul[0].header['CRPIX1']
                cdel = hdul[0].header['CD1_1']

                flux = hdul[0].data[0] #* 1e-17
                flux_err = hdul[0].data[1]

            elif hdul[0].header['ORIGIN'] == 'Liverpool JMU':
                n_pixels = hdul[0].header['NAXIS']
                crval = hdul[0].header['CRVAL']
                crpix = hdul[0].header['CRPIX']
                cdel = hdul[0].header['CDELT']

                flux = hdul[0].data
                flux_err = np.full_like(flux, fill_value=np.nan)

        pixels = np.arange(1, n_pixels + 1, 1)
        wavelength = crval + (pixels - crpix) * cdel

        if logwave:
            wavelength = 10**wavelength

        super().__init__(
            wavelength=wavelength,
            flux=flux,
            flux_error=flux_err,
            redshift=hdul[0].header['Z'])
        hdul.close()


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    spec = VipersSpectra(
        "/home/pablo/Research/obs_data/VIPERS/VIPERS_W4_SPECTRA_1D_PDR2/VIPERS_411161765.fits")
    spec.to_rest_frame()

    plt.figure()
    plt.step(spec.wavelength, spec.flux, lw=0.7, c='k')
    plt.axvline(4861)
    # plt.xlim(4800, 5000)