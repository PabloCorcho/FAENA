#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 16:58:44 2021

@author: pablo
"""


import numpy as np 
from read_cube import CALIFACube

from glob import glob

from compute_HaEW import Compute_equivalent_width, Compute_binned_equivalent_width
from compute_photometry import ComputePhotometry, ComputeBinnedPhotometry
from compute_lick_indices import ComputeLick

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from matplotlib import cm
from matplotlib.colors import LogNorm

# paths = glob('/media/pablo/Elements/CALIFA/DR3/V500/cubes/*.fits.gz')
# output_folder = '/media/pablo/Elements/CALIFA/DR3/V500/'
paths = glob('/home/pablo/obs_data/CALIFA/DR3/V500/cubes/*.fits.gz')
# output_folder = '/media/pablo/Elements/CALIFA/DR3/V500/'

for i in range(len(paths)):    
    path_i = paths[i]
    name_i = path_i[44:-20]
    print(i)    
    
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
    ref_noise = np.nanmean(flux_error[red_band, :, :], axis=0)
    ref_noise[ref_noise<=0] = np.nan
    
    sn = ref_image/ref_noise
    
    very_low_sn = sn < 0.01
    ref_image[very_low_sn] = np.nan
    ref_noise[very_low_sn] = np.nan

    cube.voronoi_binning(ref_image=ref_image, ref_noise=ref_noise, targetSN=50)
    cube.bin_cube()
    #### EW
    
    class_ew = Compute_equivalent_width(cube)
    class_ew.compute_ew()
    
    class_binned_ew = Compute_binned_equivalent_width(cube)
    class_binned_ew.compute_ew()
    
    EW = -class_ew.ew_map
    bin_EW = -class_binned_ew.ew_list
    #### PHOTOMETRY 
    cube.get_wavelength(to_rest_frame=False)
    
    class_photometry = ComputePhotometry(cube=cube, bands=['g', 'r'])    
    class_photometry.compute_photometry()
    
    class_binned_photometry = ComputeBinnedPhotometry(cube=cube, bands=['g', 'r'])    
    class_binned_photometry.compute_photometry(surface_brightness=True)
    
    g_r =class_photometry.photometry_map[0, :] - class_photometry.photometry_map[1, :]    
    bin_g_r =class_binned_photometry.photometry_list[0, :] - class_binned_photometry.photometry_list[1, :]
    
    # upper_ew = np.nanpercentile(-class_ew.ew_map, q=90)
    plt.figure()
    plt.subplot(221)
    plt.imshow(-class_ew.ew_map, vmin=-3, vmax=30, cmap='jet_r', origin='lower')
    plt.colorbar()
    plt.subplot(222)
    plt.imshow(-class_binned_ew.ew_map, vmin=-3, vmax=30, cmap='jet_r', origin='lower')
    plt.colorbar()
    plt.subplot(223)
    plt.imshow(class_ew.ew_map_err, cmap='nipy_spectral', origin='lower',
               norm=LogNorm(vmin=.1, vmax=10))
    plt.colorbar()
    plt.subplot(224)
    plt.imshow(class_binned_ew.ew_map_err, cmap='nipy_spectral', origin='lower',
               norm=LogNorm(vmin=.1, vmax=10))
    plt.colorbar()
    
    plt.figure()        
    plt.scatter(g_r.flatten(), EW.flatten(), s=1, c='r')    
    plt.scatter(bin_g_r, bin_EW, s=1)
    plt.yscale('symlog')
    
    
    
    break    

        
    