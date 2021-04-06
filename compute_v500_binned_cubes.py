#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 11:39:18 2021

@author: pablo
"""


import numpy as np 
from read_cube import CALIFACube

from glob import glob

from compute_HaEW import Compute_binned_equivalent_width
from compute_photometry import ComputeBinnedPhotometry
from compute_lick_indices import ComputeBinnedLick

from astropy.io import fits

import timeit

color_bin_edges=np.linspace(0, 1.25, 20)
ew_bin_edges = np.linspace(-5, 100, 50)
ew_bin_edges = np.hstack((np.arange(-5.6, 1, 0.2), np.logspace(0, 2.3, 30)))

color_bins = (color_bin_edges[:-1]+color_bin_edges[1:])/2
ew_bins = (ew_bin_edges[1:]+ew_bin_edges[:-1])/2

# paths = glob('/media/pablo/Elements/CALIFA/DR3/V500/cubes/*.fits.gz')
# output_folder = '/media/pablo/Elements/CALIFA/DR3/V500/binned/sn30/'
paths = glob('/home/pablo/obs_data/CALIFA/DR3/V500/cubes/*.fits.gz')
output_folder = '/home/pablo/obs_data/CALIFA/DR3/V500/binned/'

   

for i in range(len(paths)):    
    path_i = paths[i]
    name_i = path_i[43:-20]
    print(i)    
    start = timeit.default_timer()
    
    
    cube = CALIFACube(path=path_i, abs_path=True)
    
    cube.get_flux()        
    cube.get_wavelength(to_rest_frame=True)
    cube.get_bad_pixels()
    cube.mask_bad()
    
    wl = cube.wl
    flux = cube.flux
    flux_error = cube.flux_error
    bad_pixels = cube.bad_pix
    red_band = (wl>6540)&(wl<6580)
    #### BINNING
    ref_image = np.nanmean(flux[red_band, :, :], axis=0)
    ref_image[ref_image<=0] = np.nan    
    ref_noise = np.nanmedian(flux_error[red_band, :, :], axis=0)
    ref_noise[ref_noise<=0] = np.nan
    
    sn = ref_image/ref_noise
    
    very_low_sn = sn < 0.01
    ref_image[very_low_sn] = np.nan
    ref_noise[very_low_sn] = np.nan

    try:
        cube.voronoi_binning(ref_image=ref_image, ref_noise=ref_noise, targetSN=30)
        cube.bin_cube()
    except:
        continue
    #### EW
    
    class_binned_ew = Compute_binned_equivalent_width(cube)
    class_binned_ew.compute_ew()
    
    class_binned_lick = ComputeBinnedLick(cube, indices=['Lick_Mgb',
                                                         'Lick_Fe5270',
                                                         'Lick_Fe5335'])
    class_binned_lick.compute_lick()
    
    
    #### PHOTOMETRY 
    cube.get_wavelength(to_rest_frame=False)
        
    class_binned_photometry = ComputeBinnedPhotometry(cube=cube, bands=['g', 'r'])    
    class_binned_photometry.compute_photometry(surface_brightness=True)
        
    cube.close_cube()
    
    class_binned_ew.save_fits(output_folder+'EW/'+name_i+'.fits')
    class_binned_photometry.save_phot_fits(output_folder+'Photometry/'+name_i+'.fits')
    class_binned_lick.save_phot_fits(output_folder+'Lick/'+name_i+'.fits')
            
    stop = timeit.default_timer()
    print('Time: {:.3} secs'.format(stop - start))          
    
    