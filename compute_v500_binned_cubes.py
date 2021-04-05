#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 11:39:18 2021

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

from astropy.io import fits

import timeit

color_bin_edges=np.linspace(0, 1.25, 20)
ew_bin_edges = np.linspace(-5, 100, 50)
ew_bin_edges = np.hstack((np.arange(-5.6, 1, 0.2), np.logspace(0, 2.3, 30)))

color_bins = (color_bin_edges[:-1]+color_bin_edges[1:])/2
ew_bins = (ew_bin_edges[1:]+ew_bin_edges[:-1])/2

paths = glob('/media/pablo/Elements/CALIFA/DR3/V500/cubes/*.fits.gz')
output_folder = '/media/pablo/Elements/CALIFA/DR3/V500/binned/sn30/'
#paths = glob('/home/pablo/obs_data/CALIFA/DR3/V500/cubes/*.fits.gz')
#output_folder = '/home/pablo/obs_data/CALIFA/DR3/V500/binned/'

   

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
    
    # class_ew = Compute_equivalent_width(cube)
    # class_ew.compute_ew()
    
    class_binned_ew = Compute_binned_equivalent_width(cube)
    class_binned_ew.compute_ew()
    
    # EW = -class_ew.ew_map
    # bin_EW = -class_binned_ew.ew_list
    
    #### PHOTOMETRY 
    cube.get_wavelength(to_rest_frame=False)
    
    # class_photometry = ComputePhotometry(cube=cube, bands=['g', 'r'])    
    # class_photometry.compute_photometry()
    
    class_binned_photometry = ComputeBinnedPhotometry(cube=cube, bands=['g', 'r'])    
    class_binned_photometry.compute_photometry(surface_brightness=True)
    
    
    cube.close_cube()
    
    class_binned_ew.save_fits(output_folder+'EW/'+name_i+'.fits')
    class_binned_photometry.save_phot_fits(output_folder+'Photometry/'+name_i+'.fits')
        
    

    stop = timeit.default_timer()
    print('Time: {:.3} secs'.format(stop - start))          
    
    # g_r =class_photometry.photometry_map[0, :] - class_photometry.photometry_map[1, :]    
    # r = class_photometry.photometry_map[1, :]    
    
    # bin_g_r =class_binned_photometry.photometry_list[0, :] - class_binned_photometry.photometry_list[1, :]
    # bin_r = class_binned_photometry.photometry_list[1, :]
    
    # ## ageing diagram histograms
    
    # H_ageing, _,_ = np.histogram2d(g_r[r<24], EW[r<24], 
    #                                 bins=[color_bin_edges, ew_bin_edges],
    #                                 density=True)
    # mass_histogram = H_ageing*np.diff(color_bin_edges)[:, np.newaxis]*\
    #               np.diff(ew_bin_edges)[np.newaxis, :]
                  
    # sorted_flat = np.argsort(H_ageing, axis=None)
    # sorted_2D = np.unravel_index(sorted_flat, H_ageing.shape)
    # density_sorted = H_ageing.flatten()[sorted_flat]
    # cumulative_mass = np.cumsum(mass_histogram[sorted_2D])
    # fraction_sorted = cumulative_mass/cumulative_mass[-1]
    # fraction = np.interp(H_ageing, density_sorted, fraction_sorted)


    # H_ageing_binned, _, _ = np.histogram2d(bin_g_r, bin_EW,
    #                                         bins=[color_bin_edges, ew_bin_edges],
    #                                         density=True)
    
    # mass_histogram = H_ageing_binned*np.diff(color_bin_edges)[:, np.newaxis]*\
    #               np.diff(ew_bin_edges)[np.newaxis, :]
                  
    # sorted_flat = np.argsort(H_ageing_binned, axis=None)
    # sorted_2D = np.unravel_index(sorted_flat, H_ageing_binned.shape)
    # density_sorted = H_ageing_binned.flatten()[sorted_flat]
    # cumulative_mass = np.cumsum(mass_histogram[sorted_2D])
    # fraction_sorted = cumulative_mass/cumulative_mass[-1]
    # binned_fraction = np.interp(H_ageing_binned, density_sorted, fraction_sorted)

    # upper_ew = np.nanpercentile(-class_ew.ew_map, q=90)
    # plt.figure()
    # plt.subplot(221)
    # plt.imshow(-class_ew.ew_map, vmin=-3, vmax=30, cmap='jet_r', origin='lower')
    # plt.colorbar()
    # plt.subplot(222)
    # plt.imshow(-class_binned_ew.ew_map, vmin=-3, vmax=30, cmap='jet_r', origin='lower')
    # plt.colorbar()
    # plt.subplot(223)
    # plt.imshow(class_ew.ew_map_err, cmap='nipy_spectral', origin='lower',
    #             norm=LogNorm(vmin=.1, vmax=10))
    # plt.colorbar()
    # plt.subplot(224)
    # plt.imshow(class_binned_ew.ew_map_err, cmap='nipy_spectral', origin='lower',
    #             norm=LogNorm(vmin=.1, vmax=10))
    # plt.colorbar()
    
    # plt.figure()        
    # plt.title(label=name_i)
    # plt.scatter(g_r[r<24], EW[r<24], s=1, c='r')    
    # # plt.contour(color_bins, ew_bins, H_ageing.T, colors='k')
    # plt.scatter(bin_g_r, bin_EW, s=1, c='b')
    # plt.contour(bingr, binew, fp, levels, colors='k')
    # plt.contour(color_bins, ew_bins, binned_fraction.T, levels=[0.1], colors='b')
    # plt.contour(color_bins, ew_bins, fraction.T, levels=[0.1], colors='r')
    # plt.yscale('symlog', linthresh=3)
    # plt.xlim(0, 1.2)
    # plt.ylim(-4, 200)
    # plt.savefig('tests/binning/'+name_i+'_ageing_diagrama.png')
    
   
