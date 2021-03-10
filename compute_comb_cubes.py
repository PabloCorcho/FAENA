#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 18:17:19 2021

@author: pablo
"""

import numpy as np 
from read_cube import CALIFACube

from glob import glob

from compute_HaEW import Compute_equivalent_width, Compute_binned_equivalent_width
from compute_photometry import ComputePhotometry, ComputeBinnedPhotometry

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

blues = cm.get_cmap('winter', 100)
reds = cm.get_cmap('autumn', 100)


newcolors = np.vstack((reds(np.linspace(0, 1, 5)),
                       blues(np.linspace(0, 1, 30))))

newcmp = ListedColormap(newcolors, name='RedsBlues')

    
paths = glob('/home/pablo/obs_data/CALIFA/DR3/COMB/cubes/*.fits.gz')

for i in range(len(paths)):    
    path_i = paths[i][43:-20]
    print(i)    
    
    cube = CALIFACube(path=path_i, abs_path=False)
    
    cube.get_flux()        
    cube.get_wavelength(to_rest_frame=True)
    cube.get_bad_pixels()
    cube.mask_bad()
    
    equivalent_width = Compute_equivalent_width(cube)        
    equivalent_width.compute_ew()
    
    photo = ComputePhotometry(cube, bands=['g', 'r'])    
    photo.compute_photometry()
    
    equivalent_width.save_fits(
        '/home/pablo/obs_data/CALIFA/DR3/COMB/EW/'+path_i+'.fits')
    
    photo.save_phot_fits(
        '/home/pablo/obs_data/CALIFA/DR3/COMB/Photometry/'+path_i+'.fits')
    
    ##########################################################################
    ew_fig = plt.figure(figsize=(8,4))
    
    ax = ew_fig.add_subplot(221, title=path_i)
    mappable = ax.imshow(-equivalent_width.ew_map, vmax=30, vmin=-5, cmap=newcmp,
               aspect='auto')
    plt.colorbar(mappable=mappable, ax=ax, label='EW')
    ax.tick_params(labelleft=False, left=False, labelbottom=False, bottom=False)
    ax = ew_fig.add_subplot(222)
    mappable= ax.imshow(equivalent_width.ew_map_err/np.abs(equivalent_width.ew_map),
               norm=LogNorm(vmin=0.1, vmax=10), cmap='inferno',
               aspect='auto')
    plt.colorbar(mappable=mappable, ax=ax, label=r'$\frac{\sigma}{|EW|}$')
    ax.tick_params(labelleft=False, left=False, labelbottom=False, bottom=False)
    
    ##########################################################################
    
    g_r = photo.photometry_map[0, :, :]-photo.photometry_map[1, :, :]
    mu_r = photo.photometry_map[1, :, :]+2.5*np.log10(cube.fiber_surface)
    
    ageing_fig = plt.figure(figsize=(8,4))
    ax = ageing_fig.add_subplot(121, title=path_i)
    mappable= ax.scatter(g_r[mu_r<26],
            -equivalent_width.ew_map[mu_r<26], 
            c=mu_r[mu_r<26],
            s=1, vmin=17, vmax=26, cmap='brg')
    plt.colorbar(mappable=mappable, ax=ax)
    ax.set_xlim(0., 1.1)
    ax.set_ylim(-10, 100)
    ax.set_yscale('symlog', linthres=0.5)
    
    # =========================================================================
    #     BINNED
    # =========================================================================
    wl = cube.wl
    flux = cube.flux
    flux_error = cube.flux_error
    
    red_band = (wl>6540)&(wl<6580)
    
    
    ref_image = np.nanmean(flux[red_band, :, :], axis=0)
    ref_image[ref_image<=0] = np.nan    
    ref_noise = np.nanmean(flux_error[red_band, :, :], axis=0)
    ref_noise[ref_noise<=0] = np.nan
    
    very_low_sn = ref_image/ref_noise < 0.01
    ref_image[very_low_sn] = np.nan
    ref_noise[very_low_sn] = np.nan

    cube.voronoi_binning(ref_image=ref_image, ref_noise=ref_noise, targetSN=50)
    cube.bin_cube()
        
    equivalent_width = Compute_binned_equivalent_width(cube)        
    equivalent_width.compute_ew()
    
    photo = ComputeBinnedPhotometry(cube, bands=['g', 'r'])    
    photo.compute_photometry()
    
    equivalent_width.save_fits(
        '/home/pablo/obs_data/CALIFA/DR3/COMB/EW/'+path_i+'_binned.fits')
    
    photo.save_phot_fits(
        '/home/pablo/obs_data/CALIFA/DR3/COMB/Photometry/'+path_i+'_binned.fits')
    ##########################################################################
    ax = ew_fig.add_subplot(223, title=path_i)
    mappable = ax.imshow(-equivalent_width.ew_map, vmax=30, vmin=-5, cmap=newcmp,
               aspect='auto')
    plt.colorbar(mappable=mappable, ax=ax, label='EW')
    ax.tick_params(labelleft=False, left=False, labelbottom=False, bottom=False)
    ax = ew_fig.add_subplot(224)
    mappable = ax.imshow(equivalent_width.ew_map_err/np.abs(equivalent_width.ew_map),
               norm=LogNorm(vmin=0.1, vmax=10), cmap='inferno',
               aspect='auto')
    plt.colorbar(mappable=mappable, ax=ax, label=r'$\frac{\sigma}{|EW|}$')
    ax.tick_params(labelleft=False, left=False, labelbottom=False, bottom=False)
    
    
    ew_fig.savefig('/home/pablo/obs_data/CALIFA/DR3/COMB/EW/'+path_i+'.png')
    plt.close()
    ##########################################################################
    g_r = photo.photometry_list[0, :]-photo.photometry_list[1, :]
    mu_r = photo.photometry_list[1, :]+2.5*np.log10(cube.bin_surface)
        
    ax = ageing_fig.add_subplot(122, title=path_i)
    mappable = ax.scatter(g_r[mu_r<26],
            -equivalent_width.ew_list[mu_r<26], 
            c=mu_r[mu_r<26],
            s=1, vmin=17, vmax=26, cmap='brg')
    plt.colorbar(mappable=mappable, ax=ax)
    ax.set_xlim(0., 1.1)
    ax.set_ylim(-10, 100)
    ax.set_yscale('symlog', linthres=0.5)
    ageing_fig.savefig('/home/pablo/obs_data/CALIFA/DR3/COMB/ageing_diagram/'+path_i+'.png')
    plt.close()
        
    # if i == 10:
    #     break