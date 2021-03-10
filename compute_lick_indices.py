#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 14:36:58 2021

@author: pablo
"""

from glob import glob
import os

import numpy as np
from matplotlib import pyplot as plt
from read_cube import CALIFACube
from measurements.lick_indices import Lick_index

import astropy.io.fits as fits

                
                
class ComputeLick(object):
    def __init__(self, cube, indices):
        
        self.cube = cube
        self.indices = indices
        
        
        self.lick_map = np.empty((len(self.indices), 
                                        self.cube.flux.shape[1], 
                                        self.cube.flux.shape[2]))                
        self.lick_map[:] = np.nan
        self.lick_map_err = np.zeros_like(self.lick_map)
        self.lick_map_err[:] = np.nan
        
    def compute_lick(self):
                
        flux = self.cube.flux*self.cube.wl[:, np.newaxis, np.newaxis]
        flux_error = self.cube.flux_error*self.cube.wl[:, np.newaxis, np.newaxis]
            
        for i_elem in range(flux.shape[1]):
            for j_elem in range(flux.shape[2]):
    
                # print('Computing region ({},{})'.format(i_elem, j_elem))
                flux_ij = flux[:, i_elem, j_elem]                                            
                flux_ij_err = flux_error[:, i_elem, j_elem]
                
                for element in range(len(self.indices)):
                    Lick = Lick_index(lick_index_name=self.indices[element], 
                                      lamb=self.cube.wl, 
                                      flux=flux_ij,
                                      flux_err=flux_ij_err)
                    lick_index, lick_index_err = Lick.lick_index, Lick.lick_index_err
                    self.lick_map[element, i_elem, j_elem] = lick_index
                    self.lick_map_err[element, i_elem, j_elem] = lick_index_err                    
        
    def save_phot_fits(self, path):
        hdr = fits.Header()
        image_list = []        
        
        for ith in range(len(self.indices)):
            hdr['COMMENT '+str(ith)] = self.indices[ith]+"-Lick index IMAGE"
        
        image_list.append(fits.PrimaryHDU(header=hdr))
        
        for ith in range(len(self.indices)):
            image_list.append(fits.ImageHDU(self.lick_map[ith, :, :]))
            image_list.append(fits.ImageHDU(self.lick_map_err[ith, :, :]))
            
        hdu = fits.HDUList(image_list)

        hdu.writeto(path, overwrite=True)
        print('File saved as: '+ path)
        
class ComputeBinnedLick(object):
    """
    This object computes lick from binned cubes with Voronoi 
    (Capellari et al. 2003)
    """
    def __init__(self, cube, indices):
        
        
        self.cube = cube
        self.indices = indices
                
        self.lick_list = np.empty((len(self.indices), 
                                        self.cube.binned_flux.shape[1])
                                       )                
        self.lick_list_err = np.empty_like(self.lick_list)
        
        
        self.lick_map = np.empty((len(self.indices), 
                                        self.cube.flux.shape[1],
                                        self.cube.flux.shape[2])
                                       )                
        
        self.lick_map[:] = np.nan
        self.lick_map_err = np.zeros_like(self.lick_map)
        self.lick_map_err[:] = np.nan
        
    def compute_lick(self):
        
        flux = self.cube.binned_flux*self.cube.wl[:, np.newaxis]
        flux_error = self.cube.binned_flux_error*self.cube.wl[:, np.newaxis]
                
        for i_elem in range(self.cube.nbins):                        
                flux_i = flux[:, i_elem]                                                            
                flux_i_err = flux_error[:, i_elem]
                bad_px = np.isnan(flux_i)
                mask = self.cube.bin_map == i_elem                
                for element in range(len(self.indices)):
                    Lick = Lick_index(
                                        lick_index_name=self.indices[element], 
                                        lamb=self.cube.wl[~bad_px], 
                                        flux=flux_i[~bad_px],
                                        flux_err=flux_i_err[~bad_px]
                                        )
                    lick_index, lick_index_err = Lick.lick_index, Lick.lick_index_err
                    self.lick_list[element, i_elem] = lick_index
                    self.lick_list_err[element, i_elem] = lick_index_err
                    
                    self.lick_map[element, mask] = lick_index
                    self.lick_map_err[element, mask] = lick_index_err                    
        
    def save_phot_fits(self, path):
        hdr = fits.Header()
        image_list = []        
        
        for ith in range(len(self.indices)):
            hdr['COMMENT '+str(ith)] = self.indices[ith]+"-Lick index IMAGE"
        
        image_list.append(fits.PrimaryHDU(header=hdr))
        
        for ith in range(len(self.indices)):
            image_list.append(fits.ImageHDU(self.lick_map[ith, :, :]))
            image_list.append(fits.ImageHDU(self.lick_map_err[ith, :, :]))
            
        hdu = fits.HDUList(image_list)

        hdu.writeto(path, overwrite=True)
        print('File saved as: '+ path)        
        

if __name__=='__main__':        
    cube = CALIFACube(mode='V500', path='NGC0160')
    cube.get_flux()        
    cube.get_wavelength(to_rest_frame=True)            
    cube.get_bad_pixels()
    cube.mask_bad()
    
    wl = cube.wl
    flux = cube.flux
    flux_error = cube.flux_error
    
    red_band = (wl>6540)&(wl<6580)
    
    ref_image = np.nanmean(flux[red_band, :, :], axis=0)
    ref_image[ref_image<=0] = np.nan
    # noise_i = np.sqrt(np.nansum(error[red_band, :, :]**2, axis=0))
    ref_noise = np.nanmean(flux_error[red_band, :, :], axis=0)
    ref_noise[ref_noise<=0] = np.nan
    
    very_low_sn = ref_image/ref_noise < 0.01
    ref_image[very_low_sn] = np.nan
    ref_noise[very_low_sn] = np.nan

    cube.voronoi_binning(ref_image=ref_image, ref_noise=ref_noise, targetSN=50)
    cube.bin_cube()
    
    lick = ComputeBinnedLick(cube, indices=['Lick_Hb', 'Lick_Mgb',
                                            'Lick_Fe5270', 'Lick_Fe5335'])
    
    lick.compute_lick()
    
    # photo.save_phot_fits('/home/pablo/obs_data/CALIFA/DR3/COMB/Lick/NGC0001.fits')
    
    hbeta = lick.lick_map[0, :, :]
    mgfe = np.sqrt(lick.lick_map[1, :, :]*(0.72*lick.lick_map[2, :, :]+\
                                           0.28*lick.lick_map[3, :, :]))
        
    plt.figure()
    plt.imshow(hbeta, cmap='jet')
    plt.colorbar()
        
    
    plt.figure()
    plt.imshow(mgfe, cmap='jet', vmin=0.5, vmax=4)
    plt.colorbar()
    
    lick = ComputeLick(cube, indices=['Lick_Hb', 'Lick_Mgb',
                                            'Lick_Fe5270', 'Lick_Fe5335'])
    
    lick.compute_lick()
    
    # photo.save_phot_fits('/home/pablo/obs_data/CALIFA/DR3/COMB/Lick/NGC0001.fits')
    
    hbeta = lick.lick_map[0, :, :]
    hbeta_err = lick.lick_map_err[0, :, :]
    mgfe = np.sqrt(lick.lick_map[1, :, :]*(0.72*lick.lick_map[2, :, :]+\
                                           0.28*lick.lick_map[3, :, :]))
    plt.figure()
    plt.imshow(hbeta, cmap='jet', vmin=0.5, vmax=3)
    plt.colorbar()

    plt.figure()
    plt.imshow(hbeta_err/np.abs(hbeta), cmap='jet', vmax=.1, vmin=0)
    plt.colorbar()
    
    plt.figure()
    plt.imshow(mgfe, cmap='jet', vmin=0.5, vmax=4)
    plt.colorbar()
    