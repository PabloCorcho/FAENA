#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 16:37:32 2022

@author: pablo
"""

from astropy.io import fits


class Spectra(object):
    """..."""

    def __init__(self, wavelength, flux, flux_error, **kwargs):
        self.wavelength, self.flux, self.flux_error = (
            wavelength, flux, flux_error)
        # Redshift and rest frame
        self.redshift = kwargs.get('redshift', None)
        self.rest_frame = False

    def to_rest_frame(self):
        """Set wavelength array to rest frame."""
        if self.redshift is not None:
            self.wavelength /= 1 + self.redshift
            print("[Spectra] Wavelength array to rest-frame (z={:.4})"
                  .format(self.redshift))
            self.rest_frame = True
        else:
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



if __name__ == '__main__':
    from matplotlib import pyplot as plt
    spec = VipersSpectra(
        "/home/pablo/Research/obs_data/VIPERS/VIPERS_W4_SPECTRA_1D_PDR2/VIPERS_411161765.fits")
    spec.to_rest_frame()

    plt.figure()
    plt.plot(spec.wavelength, spec.flux, lw=0.7, c='k')
    plt.axvline(4861)
    plt.xlim(4800, 5000)