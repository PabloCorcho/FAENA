#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 18:16:16 2021

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
import os

from matplotlib import pyplot as plt

target_signal_to_noise = 30
# paths = glob('/media/pablo/Elements/CALIFA/DR3/V500/cubes/*.fits.gz')
# output_folder = '/media/pablo/Elements/CALIFA/DR3/V500/binned/sn'+str(target_signal_to_noise)
paths = glob('/home/pablo/obs_data/CALIFA/DR3/V500/cubes/*.fits.gz')
output_folder = '/home/pablo/obs_data/CALIFA/DR3/V500/binned/sn'+str(target_signal_to_noise)

names = np.loadtxt(
                'ned_califa_redshifts.txt',
                       usecols=(0), dtype=str, delimiter=', ')

califa_redshift, ned_redshift, bad_califa = np.loadtxt(
                'ned_califa_redshifts.txt',
                       usecols=(1,2,3), unpack=True, delimiter=', ')
    
selected_galaxies = np.where(bad_califa == 1)        
names = names[selected_galaxies]
# %%
if not os.path.isdir(output_folder):
    os.mkdir(output_folder)
    os.mkdir(output_folder+'/EW')
    os.mkdir(output_folder+'/Lick')
    os.mkdir(output_folder+'/Photometry')
    print('New directories created at ', output_folder)
        
output_folder+='/'

#paths = glob('/home/pablo/obs_data/CALIFA/DR3/V500/cubes/*.fits.gz')
#output_folder = '/home/pablo/obs_data/CALIFA/DR3/V500/binned/'

   

for name_i in names:    
    
    start = timeit.default_timer()
    
    # %%
    cube = CALIFACube(path=name_i, abs_path=False)
    
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

    
    cube.voronoi_binning(ref_image=ref_image, ref_noise=ref_noise, 
                              targetSN=target_signal_to_noise)
    cube.bin_cube()

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
                
    del class_binned_ew, class_binned_lick, class_binned_photometry
    stop = timeit.default_timer()
    print('Time: {:.3} secs'.format(stop - start))          
    
    
