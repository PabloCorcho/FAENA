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
from measurements import photometry 

import astropy.io.fits as fits

                
                
class ComputePhotometry(object):
    def __init__(self, cube, bands, system='AB'):
        
        self.cube = cube
        self.bands = bands
        self.system = system
        
        self.photometry_map = np.empty((len(self.bands), 
                                        self.cube.flux.shape[1], 
                                        self.cube.flux.shape[2]))                
        self.photometry_map[:] = np.nan
        self.photometry_map_err = np.zeros_like(self.photometry_map)
        self.photometry_map_err[:] = np.nan
        
    def compute_photometry(self, abs_phot=False):
        
        if abs_phot:
            flux = self.cube.luminosity*self.cube.wl[:, np.newaxis, np.newaxis]
            flux_error = self.cube.luminosity_error*self.cube.wl[:, np.newaxis, np.newaxis]
        else: 
            flux = self.cube.flux*self.cube.wl[:, np.newaxis, np.newaxis]
            flux_error = self.cube.flux_error*self.cube.wl[:, np.newaxis, np.newaxis]
            
        for i_elem in range(flux.shape[1]):
            for j_elem in range(flux.shape[2]):
    
                # print('Computing region ({},{})'.format(i_elem, j_elem))
                flux_ij = flux[:, i_elem, j_elem]                                            
                flux_ij_err = flux_error[:, i_elem, j_elem]
                
                for element in range(len(self.bands)):
                    mag = photometry.magnitude(absolute=abs_phot, 
                                           filter_name=self.bands[element], 
                                           wavelength=self.cube.wl, 
                                           flux=flux_ij,
                                           flux_err=flux_ij_err,
                                           photometric_system=self.system)
                    mag, mag_err = mag.AB()
                    self.photometry_map[element, i_elem, j_elem] = mag
                    self.photometry_map_err[element, i_elem, j_elem] = mag_err
                
    
        
    def save_phot_fits(self, path):
        hdr = fits.Header()
        image_list = []        
        
        for ith in range(len(self.bands)):
            hdr['COMMENT '+str(ith)] = self.bands[ith]+"-band IMAGE AB mags"
        
        image_list.append(fits.PrimaryHDU(header=hdr))
        
        for ith in range(len(self.bands)):
            image_list.append(fits.ImageHDU(self.photometry_map[ith, :, :]))
            image_list.append(fits.ImageHDU(self.photometry_map_err[ith, :, :]))
            
        hdu = fits.HDUList(image_list)

        hdu.writeto(path, overwrite=True)
        hdu.close()
        print('File saved as: '+ path)
        
class ComputeBinnedPhotometry(object):
    """
    This object computes photometry from binned cubes with Voronoi 
    (Capellari et al. 2003)
    """
    def __init__(self, cube, bands, system='AB'):
        
        
        self.cube = cube
        self.bands = bands
        self.system = system
        
        self.photometry_list = np.empty((len(self.bands), 
                                        self.cube.binned_flux.shape[1])
                                       )                
        self.photometry_list_err = np.empty_like(self.photometry_list)
        
        
        self.photometry_map = np.empty((len(self.bands), 
                                        self.cube.flux.shape[1],
                                        self.cube.flux.shape[2])
                                       )                
        
        self.photometry_map[:] = np.nan
        self.photometry_map_err = np.zeros_like(self.photometry_map)
        self.photometry_map_err[:] = np.nan
        
    def compute_photometry(self, abs_phot=False, surface_brightness=False):
        
        flux = self.cube.binned_flux*self.cube.wl[:, np.newaxis]
        flux_error = self.cube.binned_flux_error*self.cube.wl[:, np.newaxis]
        
        if surface_brightness:
            flux = flux/self.cube.bin_surface[np.newaxis, :]
            flux_error = flux_error/self.cube.bin_surface[np.newaxis, :]
        
        for i_elem in range(self.cube.nbins):                        
                flux_i = flux[:, i_elem]                                                            
                flux_i_err = flux_error[:, i_elem]
                bad_px = np.isnan(flux_i)
                mask = self.cube.bin_map == i_elem                
                for element in range(len(self.bands)):
                    mag = photometry.magnitude(absolute=abs_phot, 
                                           filter_name=self.bands[element], 
                                           wavelength=self.cube.wl[~bad_px], 
                                           flux=flux_i[~bad_px],
                                           flux_err=flux_i_err[~bad_px],
                                           photometric_system=self.system)
                    mag, mag_err = mag.AB()
                    self.photometry_list[element, i_elem] = mag
                    self.photometry_list_err[element, i_elem] = mag_err
                    
                    self.photometry_map[element, mask] = mag
                    self.photometry_map_err[element, mask] = mag_err                    
        
    def save_phot_fits(self, path):
        hdr = fits.Header()
        image_list = []        
        
        for ith in range(len(self.bands)):
            hdr['COMMENT '+str(ith)] = self.bands[ith]+"-band IMAGE AB mags"
        hdr['COMMENT '+str(ith+1)] = "Bins map"
        
        image_list.append(fits.PrimaryHDU(header=hdr))
        
        for ith in range(len(self.bands)):
            image_list.append(fits.ImageHDU(self.photometry_map[ith, :, :]))
            image_list.append(fits.ImageHDU(self.photometry_map_err[ith, :, :]))
            
        image_list.append(fits.ImageHDU(self.cube.bin_map))
        
        hdu = fits.HDUList(image_list)

        hdu.writeto(path, overwrite=True)
        hdu.close()
        print('File saved as: '+ path)        
        

if __name__=='__main__':        
    cube = CALIFACube(path='NGC2487')
    cube.get_flux()        
    cube.get_wavelength(to_rest_frame=True)            
    cube.get_bad_pixels()
    cube.mask_bad()
    
    wl = cube.wl
    flux = cube.flux
    flux_error = cube.flux_error
    
    # red_band = (wl>6540)&(wl<6580)
    
    # ref_image = np.nanmean(flux[red_band, :, :], axis=0)
    # ref_image[ref_image<=0] = np.nan
    # # noise_i = np.sqrt(np.nansum(error[red_band, :, :]**2, axis=0))
    # ref_noise = np.nanmean(flux_error[red_band, :, :], axis=0)
    # ref_noise[ref_noise<=0] = np.nan
    
    # very_low_sn = ref_image/ref_noise < 0.01
    # ref_image[very_low_sn] = np.nan
    # ref_noise[very_low_sn] = np.nan

    # cube.voronoi_binning(ref_image=ref_image, ref_noise=ref_noise, targetSN=50)
    # cube.bin_cube()
    
    # photo = ComputeBinnedPhotometry(cube, bands=['g', 'r'])
    
    # photo.compute_photometry(surface_brightness=True)
    
    # photo.save_phot_fits('/home/pablo/obs_data/CALIFA/DR3/COMB/Photometry/NGC0001.fits')
    
    # my_g_r = photo.photometry_map[0, :, :]-photo.photometry_map[1, :, :]
    # plt.figure()
    # plt.imshow(my_g_r, vmax=0, vmin=1, cmap='jet')
    # plt.colorbar()
        
    # plt.figure()
    # plt.imshow(photo.photometry_map[1, :, :], vmin=17, vmax=26, cmap='nipy_spectral')
    # plt.colorbar()
    
    # plt.figure()
    # plt.imshow(np.log10(photo.photometry_map_err[1, :, :]),   cmap='nipy_spectral')
    # plt.colorbar()
    
    photo = ComputePhotometry(cube, bands=['g', 'r'])
    
    photo.compute_photometry()
    
    # photo.save_phot_fits('/home/pablo/obs_data/CALIFA/DR3/COMB/Photometry/NGC0001.fits')
    
    my_g_r = photo.photometry_map[0, :, :]-photo.photometry_map[1, :, :]
    plt.figure()
    plt.imshow(my_g_r, vmax=0, vmin=1, cmap='jet')
    plt.colorbar()
        
    plt.figure()
    plt.imshow(photo.photometry_map[1, :, :], vmin=17, vmax=24, cmap='nipy_spectral')
    plt.colorbar()
    
    plt.figure()
    plt.imshow(photo.photometry_map_err[1, :, :],   cmap='nipy_spectral',
               vmax=0.1)
    plt.colorbar()
    