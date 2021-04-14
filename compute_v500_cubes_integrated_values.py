#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 10:33:40 2021

@author: pablo
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 12:48:39 2021

@author: pablo
"""


import numpy as np 
from read_cube import CALIFACube

from glob import glob
import pandas as pd
from measurements import compute_ha_ew
from measurements.photometry import magnitude
from measurements.lick_indices import Lick_index
import timeit


paths = glob('/home/pablo/obs_data/CALIFA/DR3/V500/cubes/*.fits.gz')
# output_folder = '/media/pablo/Elements/CALIFA/DR3/V500/'
    
# paths = glob('/media/pablo/Elements/CALIFA/DR3/V500/cubes/*.fits.gz')
# output_folder = '/media/pablo/Elements/CALIFA/DR3/V500/'
integrated_photometry = np.zeros((2, len(paths)))
integrated_photometry_err = np.zeros((2, len(paths)))
integrated_ew = np.zeros((len(paths)))
integrated_ew_err = np.zeros((len(paths)))
integrated_lick = np.zeros((3, len(paths)))
integrated_lick_err = np.zeros((3, len(paths)))
names = []

for i in range(len(paths)):    
    path_i = paths[i]
    name_i = path_i[44:-20]
    names.append(name_i)
    print(i,  name_i)    
    start = timeit.default_timer()
    
    cube = CALIFACube(path=path_i, abs_path=True)
    
    cube.get_flux()        
    cube.get_wavelength(to_rest_frame=False)
    cube.get_bad_pixels()
    cube.mask_bad()
    
    wl = cube.wl
    integrated_flux = np.nansum(cube.flux, axis=(1,2))
    integrated_flux_error = np.sqrt(np.nansum(cube.flux_error**2, axis=(1,2)))
    
    bands=['g', 'r']
    for ii, band_i in enumerate(bands):
        Mag = magnitude(absolute=False, filter_name=band_i, 
                                           wavelength=wl, 
                                           flux=integrated_flux*wl,
                                           flux_err=integrated_flux_error*wl,
                                           photometric_system='AB')
        mag, mag_err = Mag.AB()
        integrated_photometry[ii, i] = mag
        integrated_photometry_err[ii, i] = mag_err
                
        
    
    
    cube = CALIFACube(path=path_i, abs_path=True)
    
    cube.get_flux()        
    cube.get_wavelength(to_rest_frame=True)
    cube.get_bad_pixels()
    cube.mask_bad()
    
    wl = cube.wl
    integrated_flux = np.nansum(cube.flux, axis=(1,2))
    integrated_flux_error = np.sqrt(np.nansum(cube.flux_error**2, axis=(1,2)))
    
    EW = compute_ha_ew.Compute_HaEW(integrated_flux, wl, integrated_flux_error)
    # print(EW.EW, EW.EW_err)
    integrated_ew[i] = EW.EW
    integrated_ew_err[i] = EW.EW_err
        
    
    for ii, lick_i in enumerate(['Lick_Mgb', 'Lick_Fe5270', 'Lick_Fe5335']):
        Lick = Lick_index(flux=integrated_flux, flux_err=integrated_flux_error,
                          lamb=wl, lick_index_name=lick_i)
        integrated_lick[ii, i] = Lick.lick_index
        integrated_lick_err[ii, i] = Lick.lick_index_err
                        
        
    stop = timeit.default_timer()
    print('Time: {:.3} secs'.format(stop - start))          
    break

data = {'name':names,
        'g':integrated_photometry[0, :],
        'g_err':integrated_photometry_err[0, :],
        'r':integrated_photometry[1, :],
        'r_err':integrated_photometry_err[1, :],
        'EW':integrated_ew,
        'EW_err':integrated_ew_err,
        'Mgb':integrated_lick[0, :],
        'Mgb_err':integrated_lick_err[0, :],
        'Fe5270':integrated_lick[1, :],
        'Fe5270_err':integrated_lick_err[1, :],
        'Fe5335':integrated_lick[2, :],
        'Fe5335_err':integrated_lick_err[2, :],
        }

df = pd.DataFrame(data)
df.to_csv('integrated_properties.csv')