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
        
    def compute_ew(self, plot=False):
        
        for i_elem in range(self.cube.flux.shape[1]):
            for j_elem in range(self.cube.flux.shape[2]):
        # for i_elem in [18]:
        #     for j_elem in [60]:
    
                # print('Computing region ({},{})'.format(i_elem, j_elem))
                flux_ij = self.cube.flux[:, i_elem, j_elem]                                            
                flux_ij_err = self.cube.flux_error[:, i_elem, j_elem]
                
                if flux_ij.sum() <= 0:
                    continue
                
                equivalent_width = Compute_HaEW(
                                                flux_ij,
                                                self.cube.wl, 
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
        
    from matplotlib import cm
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap
    
    blues = cm.get_cmap('winter_r', 100)
    reds = cm.get_cmap('autumn', 100)


    newcolors = np.vstack((reds(np.linspace(0, 1, 5)),
                           blues(np.linspace(0, 1, 30))))
    
    newcmp = ListedColormap(newcolors, name='RedsBlues')
    
    cube = CALIFACube(path='IC5376')
    cube.get_flux()        
    cube.get_wavelength(to_rest_frame=True)            
    cube.get_bad_pixels()
    cube.mask_bad()
    
    wl = cube.wl
    flux = cube.flux
    flux_error = cube.flux_error
    
    
    ew = Compute_equivalent_width(cube)
    ew.compute_ew(plot=True)
    
    
    # cube.voronoi_binning(ref_image=ref_image, ref_noise=ref_noise, targetSN=50)
    # cube.bin_cube()
    
    # photo = Compute_binned_equivalent_width(cube)        
    # photo.compute_ew()
    
    # plt.figure()
    # plt.imshow(-photo.ew_map, vmax=30, vmin=-5, cmap=newcmp, origin='lower')
    # plt.colorbar()
    # plt.figure()
    # plt.imshow(np.log10(photo.ew_map_err/np.abs(photo.ew_map)), origin='lower',
    #            cmap='rainbow')
    # plt.colorbar(label=r'$\sigma(EW)/EW$')  
    
    # # photo.save_fits('/home/pablo/obs_data/CALIFA/DR3/COMB/Photometry/NGC6004.fits')
    
    # photo = Compute_equivalent_width(cube)        
    # photo.compute_ew()
    
    # plt.figure()
    # plt.imshow(-photo.ew_map, vmax=30, vmin=-5, cmap=newcmp, origin='lower')
    # plt.colorbar()
    # plt.figure()
    # plt.imshow(np.log10(photo.ew_map_err/np.abs(photo.ew_map)), origin='lower',
    #            cmap='rainbow')
    # plt.colorbar(label=r'$\sigma(EW)/EW$')  
    
    # # photo.save_fits('/home/pablo/obs_data/CALIFA/DR3/COMB/Photometry/NGC6004.fits')
    
    # # import pyphot
    # # from pyphot import unit
    # # lib = pyphot.get_library()

    # # sdss_r = lib['SDSS_r']
    # # sdss_g = lib['SDSS_g']
    # # g_r = np.zeros((cube.flux.shape[1],cube.flux.shape[2]))
    
    # # for ith in range(cube.flux.shape[1]):
    # #     for jth in range(cube.flux.shape[2]):
    # #         print(ith, jth)
    # #         spectra = cube.flux[:, ith, jth]
            
            
    # #         r_flux = sdss_r.get_flux(cube.wl, spectra)
    # #         g_flux = sdss_g.get_flux(cube.wl, spectra)

    # #         r_mag = -2.5 * np.log10(r_flux.value) - sdss_r.AB_zero_mag
    # #         g_mag = -2.5 * np.log10(g_flux.value) - sdss_g.AB_zero_mag
            
    # #         g_r[ith, jth] = g_mag - r_mag
    
    
    # # plt.imshow(my_g_r, vmin=0., vmax=1, cmap='jet')
    # # plt.colorbar()