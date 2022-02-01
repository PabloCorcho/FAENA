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
from measurements.equivalent_width import compute_all_ew
from measurements.photometry import AB_mags
import os
from tqdm import tqdm

lib = pyphot.get_library()
filters = [lib[filter_i] for filter_i in ['SDSS_g', 'SDSS_r']]

# cubes_path = '/media/pablo/Elements/MANGA/DR17/cubes/*.fits.gz' # endurance
cubes_path = '/home/pablo/obs_data/MANGA/cubes/*.fits.gz' # roach
cubes = glob(cubes_path)

results_dir = 'MANGA-results/'

all_ew_ha = []
all_ew_hb = []
all_ew_hg = []
all_ew_hd = []
all_gr = []
all_d4000 = []
all_id = []
stars_in_field = []
for i in tqdm(range(len(cubes))):
    cube_path = cubes[i]
    manga = MANGACube(path=cube_path, abs_path=True)
    galaxy_id = manga.plate + '-' + manga.ifudesign
    manga.get_flux()
    manga.get_redshift()
    manga.get_wavelength(to_rest_frame=True)
    r_image = manga.get_image('r')
    try:
        mask, nstars = deblend_stars(r_image,
                                     output_dir=results_dir + 'stars/' + galaxy_id + '_')
    except:
        mask = np.zeros_like(r_image, dtype=bool)
        nstars = -99
        print('WARNING: Failure during stellar masking, assuming 0 stars')

    all_id.append(galaxy_id)
    stars_in_field.append(nstars)

    flux = manga.flux
    flux_err = manga.flux_error
    integrated_flux = np.nansum(flux[:, ~mask], axis=1)
    integrated_flux_err = np.sqrt(np.nansum(flux_err[:, ~mask]**2, axis=1))

    balmer_break, ha, hb, hg, hd = compute_all_ew(wl=manga.wl,
                                                  spec=integrated_flux)
    g_flux, g_mag = AB_mags(wl=manga.wl, spec=integrated_flux,
                            filter=filters[0])
    r_flux, r_mag = AB_mags(wl=manga.wl, spec=integrated_flux,
                            filter=filters[1])
    all_ew_ha.append(ha[-1])
    all_ew_hb.append(hb[-1])
    all_ew_hg.append(hg[-1])
    all_ew_hd.append(hd[-1])
    all_gr.append(g_mag - r_mag)
    all_d4000.append(balmer_break)

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(212)
    ax.semilogy(manga.wl, integrated_flux / integrated_flux_err, lw=1, c='k')
    ax.grid(b=True)
    ax.set_ylim(1, 1000)
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
    fig.savefig(results_dir + '/QC/' + galaxy_id, bbox_inches='tight')
    fig.clear()
    plt.close()

    np.savetxt(results_dir + '/spec/' + galaxy_id,
               np.array([manga.wl, integrated_flux, integrated_flux_err]).T)
    manga.close_cube()

    # if i == 5:
    #     break
all_ew_ha = np.array(all_ew_ha)
all_ew_hb = np.array(all_ew_hb)
all_ew_hg = np.array(all_ew_hg)
all_ew_hd = np.array(all_ew_hd)

all_gr = np.array(all_gr)
all_d4000 = np.array(all_d4000)
all_id = np.array(all_id)
stars_in_field = np.array(stars_in_field)

np.savetxt(results_dir + 'stars_detected',
           np.array([all_id, stars_in_field]).T,
           header='ID, Nstars',
           fmt='%s')
np.savetxt(results_dir + 'manga_proxies_',
           np.array([all_id, all_ew_ha, all_ew_hb, all_ew_hg, all_ew_hd,
                     all_gr, all_d4000]).T,
           header='ID, EW(Ha), EW(Hb), EW(Hg), EW(Hd), g-r, D4000',
           fmt='%s')
