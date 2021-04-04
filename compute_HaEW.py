#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 16:41:10 2021

@author: pablo
"""

import numpy as np
from matplotlib import pyplot as plt
from read_cube import CALIFACube
import astropy.io.fits as fits
from measurements.compute_ha_ew import  Compute_HaEW
from scipy.ndimage import gaussian_filter1d
                                
                
class Compute_equivalent_width(object):
    def __init__(self, cube):
        
        self.cube = cube
        
        if not cube.rest_frame:
            print('WARNING: SED IS NOT IN REST FRAME')
            
        self.ew_map = np.empty((self.cube.flux.shape[1], 
                                self.cube.flux.shape[2]))                
        
        self.ew_map[:] = np.nan
        self.ew_map_err = np.zeros_like(self.ew_map)
        self.ew_map_err[:] = np.nan
        self.ew_flag_map = np.zeros((self.cube.flux.shape[1], 
                                        self.cube.flux.shape[2]))
        self.ew_flag_map[:] = np.nan
    def compute_ew(self, plot=False):
        flux = self.cube.flux
        flux_error = self.cube.flux_error
        
        for i_elem in range(self.cube.flux.shape[1]):
            for j_elem in range(self.cube.flux.shape[2]):
        # for i_elem in [18]:
        #     for j_elem in [60]:                
                good_pixels = ~np.isnan(flux[:, i_elem, j_elem]                                            )
                wl = self.cube.wl[good_pixels]                
                flux_ij = flux[good_pixels, i_elem, j_elem]                                            
                flux_ij_err = flux_error[good_pixels, i_elem, j_elem]
                
                if np.isnan(flux_ij).all()|(np.nansum(flux_ij)<0):
                    self.ew_flag_map[i_elem, j_elem] = 0
                    continue
                elif (flux_ij.size<good_pixels.size*0.7):                
                    flux_ij = np.interp(self.cube.wl, 
                                        wl, flux_ij)
                    flux_ij_err = np.interp(self.cube.wl, 
                                        wl, flux_ij_err)                                        
                    wl = self.cube.wl
                    self.ew_flag_map[i_elem, j_elem] = 2
                else:
                    self.ew_flag_map[i_elem, j_elem] = 1
                    
                equivalent_width = Compute_HaEW(
                                                flux_ij,
                                                wl, 
                                                flux_ij_err
                                                )
                EW, EW_err = equivalent_width.EW, equivalent_width.EW_err
                
                self.ew_map[i_elem, j_elem] = EW
                self.ew_map_err[i_elem, j_elem] = EW_err
                if plot:
                    fig = equivalent_width.plot_ew()
                    fig.savefig('ew_test/region_{:}_{:}.png'.format(i_elem, 
                                                                    j_elem))
                    plt.close()
        
    def save_fits(self, path):
        hdr = fits.Header()
        image_list = []        
        
        hdr['COMMENT 1'] = "H alpha equivalent width map"
        hdr['COMMENT 2'] = "H alpha equivalent width error map"
        
        image_list.append(fits.PrimaryHDU(header=hdr))        
        image_list.append(fits.ImageHDU(self.ew_map))
        image_list.append(fits.ImageHDU(self.ew_map_err))
                    
        hdu = fits.HDUList(image_list)

        hdu.writeto(path, overwrite=True)
        hdu.close()
        print('File saved as: '+ path)
        
class Compute_binned_equivalent_width(object):
    def __init__(self, cube):
        
        self.cube = cube
        
        if not cube.rest_frame:
            print('WARNING: SED IS NOT IN REST FRAME')
            
        self.ew_map = np.empty((self.cube.flux.shape[1], 
                                self.cube.flux.shape[2]))                
        self.ew_list = np.empty((self.cube.nbins))                
        self.ew_list_err = np.empty((self.cube.nbins))                
        
        self.ew_map[:] = np.nan
        self.ew_map_err = np.zeros_like(self.ew_map)
        self.ew_map_err[:] = np.nan
        
    def compute_ew(self, plot=False):
        
        for i_elem in range(self.cube.nbins):                        

            # print('Computing region ({},{})'.format(i_elem, j_elem))
            flux_i = self.cube.binned_flux[:, i_elem]                                            
            flux_i_err = self.cube.binned_flux_error[:, i_elem]
            bad_px = np.isnan(flux_i)            
            mask = self.cube.bin_map == i_elem                
            
            if flux_i.sum() <= 0:
                continue            
            equivalent_width = Compute_HaEW(
                                            flux_i[~bad_px],
                                            self.cube.wl[~bad_px], 
                                            flux_i_err[~bad_px]
                                            )
            EW, EW_err = equivalent_width.EW, equivalent_width.EW_err
            
            self.ew_list[i_elem] = EW
            self.ew_list_err[i_elem] = EW_err
            
            self.ew_map[mask] = EW
            self.ew_map_err[mask] = EW_err
            if plot:
                fig = equivalent_width.plot_ew()
                fig.savefig('ew_test/binned_region_{:}.png'.format(i_elem))
                plt.close()
        
    def save_fits(self, path):
        hdr = fits.Header()
        image_list = []        
        
        hdr['COMMENT 1'] = "H alpha equivalent width map"
        hdr['COMMENT 2'] = "H alpha equivalent width error map"
        hdr['COMMENT 3'] = "Bins map"
        
        image_list.append(fits.PrimaryHDU(header=hdr))        
        image_list.append(fits.ImageHDU(self.ew_map))
        image_list.append(fits.ImageHDU(self.ew_map_err))
        image_list.append(fits.ImageHDU(self.cube.bin_map))
                    
        hdu = fits.HDUList(image_list)

        hdu.writeto(path, overwrite=True)
        hdu.close()
        print('File saved as: '+ path)        
        

if __name__=='__main__': 
        

    cube = CALIFACube(path='NGC3990', mode='V500')
    cube.get_flux()        
    cube.get_wavelength(to_rest_frame=True)            
    cube.get_bad_pixels()

    wl = cube.wl
    flux = cube.flux
    flux_error = cube.flux_error
    
    cube.mask_bad()
    
    flux_masked = cube.flux
    flux_error_masked = cube.flux_error
    
    
    sn = flux/flux_error
    
    ha_band = np.where((wl >6550.)&(wl<6575.))[0]
    ha_sn = np.nansum(sn[ha_band, :, :], axis=0)
    
    plt.figure()
    plt.imshow(cube.n_bad_pix)
    plt.colorbar()
    
    ew = Compute_equivalent_width(cube)
    ew.compute_ew(plot=False)
    
    plt.figure()
    plt.imshow(-ew.ew_map_err, vmax=20, vmin=-5, cmap='jet_r', origin='lower')
    plt.colorbar()
    
    plt.figure()
    plt.imshow(ew.ew_map_err, vmin=0, cmap='jet_r', origin='lower')
    plt.colorbar()
    
    