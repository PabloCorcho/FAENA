#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 08:33:24 2021

@author: pablo
"""

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

   

for path_i in paths:    
    
    start = timeit.default_timer()
    
    # %%
    cube = CALIFACube(path=path_i, abs_path=True)
    name_i = cube.name
    cube.get_flux()        
    cube.get_wavelength(to_rest_frame=True)
    cube.get_bad_pixels()
    cube.mask_bad()
    
    wl = cube.wl
    flux = cube.flux
    
    integrated_flux = np.nansum(flux, axis=(1,2))
    integrated_flux /= np.nansum(integrated_flux)
    central_flux = np.nansum(flux[:, 30:-30, 30:-30], axis=(1,2))
    central_flux /= np.nansum(central_flux)
    
        
    plt.figure()
    plt.subplot(121)
    plt.annotate(name_i, xy=(0.05, 0.95), xycoords='axes fraction', va='top',
                 ha='left')
    
    plt.plot(wl, integrated_flux, 'b-')
    plt.plot(wl, central_flux, 'r-')
    plt.axvline(6563, color='orange', ls='--')
    plt.xlim(6500, 6640)
    plt.subplot(122)    
    plt.plot(wl, integrated_flux, 'b.-')
    plt.plot(wl, central_flux, 'r.-')
    plt.axvline(5007, color='lime', ls='--')
    plt.axvline(4861, color='b', ls='--')
    plt.xlim(4800, 5050)
    plt.savefig('tests/QC/redshift_calibration/'+name_i+'.png')
    
    