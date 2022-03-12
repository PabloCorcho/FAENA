#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 13:02:18 2021

@author: pablo
"""

from astropy.io import fits
from .SSP import SSP
from .SFH import SFH
from .ELINES import ELINES


class Flipas(object):
    """todo."""

    def __init__(self, path, readSSP=True, readSFH=True, readELINES=True,
                 readINDICES=True, verbose=True):
        self.verbose = verbose
        if self.verbose:
            print('---------------------\n Initialising FLIPAS'
                  + '\n---------------------')
            print('· READING --> Opening file: '+path)

        self.hdul = fits.open(path)  # Pipe3D complete output (compressed .gz file)

        self.read_headers(self.hdul)

        if 'ORG_HDR' in self.hdr_names:
            self.original_hdr = self.hdul['ORG_HDR'].header

        if readSSP:
            self.SSP = SSP(self.hdul[self.hdr_names.index('SSP')],
                           verbose=self.verbose)
            # z = self.original_hdr['MED_VEL']/3e5
            # self.SSP.get_redshift(z)
            # print('· [WARNING] FITS FILE DOES NOT CONTAIN SSP EXTENSION!¯')
        if readSFH:
            self.SFH = SFH(self.hdul[self.hdr_names.index('SFH')], SSP=self.SSP,
                           verbose=self.verbose)

        if readELINES:
            self.ELINES = ELINES(self.hdul[self.hdr_names.index('FLUX_ELINES')],
                                 verbose=self.verbose)
            # print('· [WARNING] FITS FILE DOES NOT CONTAIN SFH EXTENSION!')
        # hdul.close()
# -----------------------------------------------------------------------------
    def read_headers(self, hdul_file):
        """todo."""
        if self.verbose:
            print('· READING --> Headers')
        n_entries = len(hdul_file)
        self.hdr_names = []
        for ii in range(n_entries):
            self.hdr_names.append(hdul_file[ii].name)


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from utils import extinction_law
    import numpy as np
    gal_path = '/home/pablo/obs_data/CALIFA/DR4/V500/Pipe3D/NGC2730.Pipe3D.cube.fits.gz'
    pipe3d = Flipas(path=gal_path)
    
    ha_flux, ha_flux_err = pipe3d.ELINES.get_line_flux('Ha')
    hb_flux, hb_flux_err = pipe3d.ELINES.get_line_flux('Hb')

    hb_mask = hb_flux/hb_flux_err > 20
    
    av, av_err  = pipe3d.SSP.get_Av_map()
    
    av_dummy = np.linspace(0, 2, 30)

    wave = np.array([4861., 6563.])

    c_lambda = extinction_law(wave=wave, a_v=1, r_v=3.1)
    hahb_ratio = 10**(-0.4 * av_dummy * (c_lambda[1] - c_lambda[0])) * 2.86
    av_gas = np.interp(ha_flux.flatten()/hb_flux.flatten(), hahb_ratio, av_dummy)
    
    plt.figure()
    plt.plot(av_dummy, 10**(-0.4 * av_dummy * (c_lambda[1] - c_lambda[0])) * 3.02,
             'r--')
    plt.plot(av_dummy, 10**(-0.4 * av_dummy * (c_lambda[1] - c_lambda[0])) * 2.86,
             'r')
    plt.plot(av_dummy, 10**(-0.4 * av_dummy * (c_lambda[1] - c_lambda[0])) * 2.75,
             'r--')
    plt.scatter(av[hb_mask], ha_flux[hb_mask]/hb_flux[hb_mask], s=1,
                c=np.log10(ha_flux[hb_mask].value), vmin=-1, cmap='nipy_spectral')
    plt.colorbar()
    plt.ylim(0, 5)
    
    
    plt.figure()
    plt.imshow(ha_flux/hb_flux, vmax=5, vmin=2.85,cmap='nipy_spectral')
    plt.colorbar()
    
    plt.figure()
    plt.scatter(av.flatten(), av_gas, s=1, alpha=0.1)
    
    plt.figure()
    plt.imshow(ha_flux/hb_flux, vmax=5, vmin=2.85,cmap='nipy_spectral')
    plt.colorbar()