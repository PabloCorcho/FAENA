#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 12:48:39 2021

@author: pablo
"""


import numpy as np 
from read_cube import CALIFACube

from glob import glob

from compute_HaEW import Compute_equivalent_width
from compute_photometry import ComputePhotometry
from compute_lick_indices import ComputeLick

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import timeit

blues = cm.get_cmap('winter', 100)
reds = cm.get_cmap('autumn', 100)


newcolors = np.vstack((reds(np.linspace(0, 1, 5)),
                       blues(np.linspace(0, 1, 30))))

newcmp = ListedColormap(newcolors, name='RedsBlues')

    
paths = glob('/media/pablo/Elements/CALIFA/DR3/V500/cubes/*.fits.gz')

output_folder = '/media/pablo/Elements/CALIFA/DR3/V500/'

for i in range(len(paths)):    
    path_i = paths[i]
    name_i = path_i[44:-20]
    print(i)    
    start = timeit.default_timer()
    
    cube = CALIFACube(path=path_i, abs_path=True)
    
    cube.get_flux()        
    cube.get_wavelength(to_rest_frame=False)
    cube.get_bad_pixels()
    cube.mask_bad()

    photo = ComputePhotometry(cube, bands=['g', 'r'])    
    photo.compute_photometry()

    cube = CALIFACube(path=path_i, abs_path=True)
    
    cube.get_flux()        
    cube.get_wavelength(to_rest_frame=True)
    cube.get_bad_pixels()
    cube.mask_bad()
    
    equivalent_width = Compute_equivalent_width(cube)        
    equivalent_width.compute_ew()
    
    lick = ComputeLick(cube,
                       indices=['Lick_Mgb', 'Lick_Fe5270', 'Lick_Fe5335'])
    lick.compute_lick()
    
    cube.close_cube()
    
    equivalent_width.save_fits(output_folder+'EW/'+name_i+'.fits')
    
    photo.save_phot_fits(output_folder+'Photometry/'+name_i+'.fits')
    
    lick.save_phot_fits(output_folder+'Lick/'+name_i+'_binned.fits')
    

    stop = timeit.default_timer()
    print('Time: {:.3} secs'.format(stop - start))          
    
