#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 10:28:37 2021

@author: pablo
"""

from read_cube import MANGACube
import numpy as np
from glob import glob
from matplotlib import pyplot as plt

from deblend_stars import deblend_stars
import pyphot
import extinction
from measurements.equivalent_width import compute_all_ew
from measurements.photometry import AB_mags
import os

lib = pyphot.get_library()
filters = [lib[filter_i] for filter_i in ['SDSS_g', 'SDSS_r']]

cubes_path = '/media/pablo/Elements/MANGA/DR17/cubes/*.fits.gz'
cubes = glob(cubes_path)

results_dir = 'MANGA-results/'

all_ew = []
all_gr = []
all_d4000 = []
for i, cube_path in enumerate(cubes):
    print(i+1)
    manga = MANGACube(path=cube_path, abs_path=True)
    galaxy_id = manga.plate + '-' + manga.ifudesign
    manga.get_flux()
    manga.get_redshift()
    manga.get_wavelength(to_rest_frame=True)
    r_image = manga.get_image('r')
    mask = deblend_stars(r_image,
                         output_dir=results_dir + 'stars/' + galaxy_id + '_')

    flux = manga.flux
    flux_err = manga.flux_error
    integrated_flux = np.nansum(flux[:, ~mask], axis=1)
    integrated_flux_err = np.nansum(flux_err[:, ~mask], axis=1)
    balmer_break, ha, hb, hg, hd = compute_all_ew(wl=manga.wl,
                                                  spec=integrated_flux)
    g_flux, g_mag = AB_mags(wl=manga.wl, spec=integrated_flux,
                            filter=filters[0])
    r_flux, r_mag = AB_mags(wl=manga.wl, spec=integrated_flux,
                            filter=filters[1])
    all_ew.append(ha[-1])
    all_gr.append(g_mag - r_mag)
    all_d4000.append(balmer_break)

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(212)
    ax.semilogy(manga.wl, integrated_flux / integrated_flux_err, lw=1, c='k')
    ax.grid(b=True)
    ax.set_ylim(1, 100)
    ax.set_ylabel(r'$F/\sigma$')
    ax = fig.add_subplot(211)
    ax.semilogy(manga.wl, integrated_flux, lw=1, c='k')
    ax.set_ylabel(r'$F_\lambda$')
    ax.axvline(6563, color='r', alpha=0.3, lw=1)
    ax.axvline(4861, color='r', alpha=0.3, lw=1)
    ax.set_ylim(integrated_flux.max()*0.01, integrated_flux.max())
    ax = fig.add_axes([.25, 1, .5, .5])
    ax.imshow(np.log10(r_image), origin='lower', cmap='terrain',
              interpolation='none')
    ax.contour(np.array(mask, dtype=int), levels=[1], colors='orange')
    fig.savefig(results_dir + '/QC/' + galaxy_id)
    fig.clear()
    plt.close()

    np.savetxt(results_dir + '/spec/' + galaxy_id,
               np.array([manga.wl, integrated_flux, integrated_flux_err]).T)
    if i == 5:
        break
all_ew = np.array(all_ew)
all_gr = np.array(all_gr)
all_d4000 = np.array(all_d4000)

np.savetxt(results_dir + 'manga_proxies',
           np.array([all_ew, all_gr, all_d4000]).T,
           header='EW(Ha), g-r, D4000',
           fmt='%.4f')
