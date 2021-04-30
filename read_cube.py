#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 11:33:31 2021

@author: pablo
"""

from astropy.io import fits

from astropy.cosmology import FlatLambdaCDM

import numpy as np
from matplotlib import pyplot as plt

# =============================================================================
# COSMOLOGY
# =============================================================================
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

# =============================================================================
#
# =============================================================================


class Cube(object):
    """
    Parent class for any IFS cube. 

    Cube properties:
        - Dimensions
        - Wavelength (wl) range
        - Flux 

    Default flux units: 'erg/s/cm2/AA'        
    Default wavelength units: AA

    """

    def __init__(self):

        self.read_cube()

    def read_cube(self):
        self.cube = fits.open(self.path_to_cube)

    def close_cube(self):
        self.cube.close()

    def get_axis(self):
        # n_hdu = len(self.cube)
        pass

    def get_wcs(self):
        pass

    def get_luminosity(self):
        try:
            self.redshift
        except:
            self.get_redshift()

        self.comoving_distance = cosmo.comoving_distance(self.redshift)  # Mpc
        self.comoving_distance = self.comoving_distance.to('cm').value

        self.luminosity = self.flux * 4*np.pi*self.comoving_distance**2
        self.luminosity_error = self.flux_error * 4*np.pi*self.comoving_distance**2

    def get_surface_brightness(self):
        self.surface_brightness = self.flux/self.fiber_surface
        self.surface_brightness_error = self.flux_error/self.fiber_surface

    def to_rest_frame(self):
        try:
            self.redshift
        except:
            self.get_redshift()

        if not self.rest_frame:
            self.wl = self.wl/(1+self.redshift)
            self.rest_frame = True
        print('Wavelength in rest frame')

    def voronoi_binning(self, ref_image, ref_noise, targetSN, plot_binning=False):
        from vorbin.voronoi_2d_binning import voronoi_2d_binning

        self.ref_bad_pix = np.isnan(ref_image) | np.isnan(ref_noise)

        self.x_pixels = np.arange(0, self.flux.shape[1])

        self.y_pixels = np.arange(0, self.flux.shape[2])

        Y, X = np.meshgrid(self.y_pixels, self.x_pixels)

        self.binNum, self.xBin, self.yBin, self.xBar, self.yBar, self.sn, \
            self.nPixels, self.scale = voronoi_2d_binning(X[~self.ref_bad_pix],
                                                          Y[~self.ref_bad_pix],
                                                          ref_image[~self.ref_bad_pix],
                                                          ref_noise[~self.ref_bad_pix],
                                                          targetSN=targetSN,
                                                          cvt=True, pixelsize=None, plot=plot_binning,
                                                          quiet=True, sn_func=None, wvt=True)

        self.bin_map = np.zeros_like(X)
        self.bin_map[:, :] = -1
        self.bin_map[~self.ref_bad_pix] = self.binNum

        self.nbins = np.unique(self.binNum).size

        self.bin_surface = self.nPixels*self.pixel_surface  # arcsec^2

    def bin_cube(self):
        self.binned_flux = np.zeros((self.flux.shape[0], self.nbins))
        self.binned_flux_error = np.zeros_like(self.binned_flux)

        for ith in range(self.nbins):
            mask = self.bin_map == ith
            flux_error_ith = self.flux_error[:, mask]
            flux_ith = self.flux[:, mask]
            bad = flux_error_ith/flux_ith > 1000
            flux_error_ith[bad] = flux_error_ith[~bad].max()

            weights = self.get_bin_err_weights(mask)

            self.binned_flux[:, ith] = np.nansum(self.flux[:, mask], axis=(1))
            self.binned_flux_error[:, ith] = np.sqrt(
                np.nansum(flux_error_ith**2*weights**2,
                          axis=(1)))

        self.binned_flux_error[self.binned_flux_error == 0] = np.nan
        self.binned_flux[self.binned_flux == 0] = np.nan


# =============================================================================
#
# =============================================================================
class CALIFACube(Cube):
    """
    This class reads CALIFA cubes
    - mode: 'COMB' (only DR3), 'V500', 'V1200'
    - data_release: 'DR2', 'DR3'
    """

    def __init__(self, path, mode='V500', abs_path=False, data_release='DR3'):

        self.pixel_surface = 1  # arcsec^2
        self.mode = mode
        self.data_release = data_release

        if abs_path:
            self.path_to_cube = path
        else:
            self.path_to_cube = '/home/pablo/obs_data/CALIFA/'+data_release+'/' +\
                                mode+'/cubes/'+path + '.'+mode + '.rscube.fits.gz'

        print('\nOpening CALIFA cube: ', self.path_to_cube, '\n')

        Cube.__init__(self)

        self.califaid = self.cube[0].header['CALIFAID']
        self.name = self.cube[0].header['object']

    def get_flux(self):
        self.flux = self.cube[0].data*1e-16
        self.flux_units = 'erg/s/cm2/AA'
        self.flux_error = self.cube[1].data*1e-16

        print('Flux units:', self.flux_units)

    def get_wavelength(self, to_rest_frame=False):
        wavelength_0 = self.cube[0].header['CRVAL3']
        wl_step = self.cube[0].header['CDELT3']
        wl_pixel = self.cube[0].header['CRPIX3']
        wl_pixels = np.arange(wl_pixel-1, self.flux.shape[0], 1)
        self.wl = wavelength_0 + wl_pixels*wl_step
        self.rest_frame = False
        if to_rest_frame:
            self.to_rest_frame()
        else:
            print('Wavelength vector is not in rest frame')

    def get_bad_pixels(self):
        """BAD PIXELS: 1 == GOOD, 0 == BAD"""
        self.bad_pix = np.array(self.cube[3].data, dtype=bool)
        # Huge relative error
        # rel_err = self.flux_error/self.flux
        # self.bad_pix[rel_err>1e3] = True
        # Negative fluxes
        # self.bad_pix[self.flux<0] = False

        self.n_bad_pix = np.zeros_like(self.bad_pix, dtype=int)
        self.n_bad_pix[self.bad_pix] = 1
        self.n_bad_pix = np.sum(self.n_bad_pix, axis=(0))

    def get_errorweigths(self):
        self.error_weights = self.cube[2].data

    def get_redshift(self):
        if self.data_release == 'DR3':
            names = np.loadtxt(
                'ned_califa_redshifts.txt',
                usecols=(0), dtype=str, delimiter=', ')
            ned_pos = np.where(names == self.name)[0]
            califa_redshift, ned_redshift, bad_califa = np.loadtxt(
                'ned_califa_redshifts.txt',
                usecols=(1, 2, 3), unpack=True, delimiter=', ')
            if bad_califa[ned_pos] == 0:
                self.redshift = califa_redshift[ned_pos]
            else:
                print(
                    '\nWARNING: CALIFA REDSHIFT UNRELIABLE \n--> SELECTING NED REDSHIFT\n')
                self.redshift = ned_redshift[ned_pos]

        # recesion_vel = self.cube[0].header['MED_VEL']
        # self.redshift = recesion_vel/3e5

    def mask_bad(self):
        print('MASKING BAD PIXELS WITH "NAN"')
        self.flux[self.bad_pix] = np.nan
        self.flux_error[self.bad_pix] = np.nan
        bad = self.flux_error/self.flux > 1000
        self.flux_error[bad] = self.flux_error[~bad].max()

    def coadded_spectral_empirical_correlation(self, n_spaxels):
        alpha = {'V500': 1.10, 'V1200': 1.08, 'COMB': 1.08}
        beta = 1 + alpha[self.mode]*np.log10(n_spaxels)
        return beta

    def get_bin_err_weights(self, spaxels_mask):
        n_spaxels = spaxels_mask[spaxels_mask].size
        if n_spaxels < 80:
            weights = np.array(
                [self.coadded_spectral_empirical_correlation(
                    n_spaxels)]*n_spaxels
            )
        else:
            try:
                self.error_weights
            except:
                self.get_errorweigths()

            weights = self.error_weights[:, spaxels_mask]
        return weights

# =============================================================================
#
# =============================================================================


class MANGACube(Cube):
    def __init__(self, plate=None, ifudesign=None, path=None, logcube=True, abs_path=False):        
        self.pixel_surface = 0.5**2  # arcsec^2
        self.plate, self.ifudesign = plate, ifudesign
        cube_wl_sampling = {True: 'LOGCUBE.fits.gz', False: 'LINCUBE.fits.gz'}
        if abs_path:
            self.path_to_cube = path
        else:
            self.path_to_cube = '/home/pablo/obs_data/MANGA/cubes/' +\
                                'manga-'+self.plate+'-'+self.ifudesign+'-' +\
                                cube_wl_sampling[logcube]
        Cube.__init__(self)

    def get_flux(self):
        self.flux = self.cube[1].data*1e-17
        self.flux_units = 'erg/s/cm2/AA/spaxel'
        self.flux_error = 1/np.sqrt(self.cube[2].data) * 1e-17
        print('Flux units:', self.flux_units)

    def get_wavelength(self, to_rest_frame=False):
        self.wl = self.cube[6].data
        self.rest_frame = False
        if to_rest_frame:
            self.to_rest_frame()
        else:
            print('Wavelength vector is not in rest frame')

    def get_bad_pixels(self):
        """BAD PIXELS: 0 == GOOD, 1027 == BAD"""
        self.bad_pix = np.array(self.cube[3].data, dtype=bool)
        # Huge relative error
        # rel_err = self.flux_error/self.flux
        # self.bad_pix[rel_err>1e3] = True
        # Negative fluxes
        # self.bad_pix[self.flux<0] = False

        self.n_bad_pix = np.zeros_like(self.bad_pix, dtype=int)
        self.n_bad_pix[self.bad_pix] = 1
        self.n_bad_pix = np.sum(self.n_bad_pix, axis=(0))

    def mask_bad(self):
        print('MASKING BAD PIXELS WITH "NAN"')
        self.flux[self.bad_pix] = np.nan
        self.flux_error[self.bad_pix] = np.nan
        bad = self.flux_error/self.flux > 1000
        self.flux_error[bad] = self.flux_error[~bad].max()

    def get_redshift(self):        
        self.redshift = self.get_catalog(field='NSA_Z')
                
    def get_catalog(self, field=None):
        self.catalog_path = 'drpall-v2_4_3.fits'
        try:
            self.catalog
        except:
            self.catalog = fits.open(
                '/home/pablo/obs_data/MANGA/'+self.catalog_path)[1].data    
            self.cat_entry = np.where(
                self.catalog['plateifu']== self.plate+'-'+self.ifudesign)[0][0]
        if field:
            return self.catalog[field][self.cat_entry]
    
    def get_derived_catalog(self, field=None, line_field=None):
        self.derived_catalog_path = 'dapall-v2_4_3-2.2.1.fits'
        try:
            self.derived_catalog
        except:
            self.derived_catalog = fits.open(
                '/home/pablo/obs_data/MANGA/'+self.derived_catalog_path)[1].data    
            self.derived_catalog_lines = fits.open(
                '/home/pablo/obs_data/MANGA/'+self.derived_catalog_path)[0].header    
            self.derived_cat_entry = np.where(
                self.derived_catalog['plateifu']== self.plate+'-'+self.ifudesign)[0][0]
        if field:
            return self.derived_catalog[field][self.derived_cat_entry]        
        if line_field:            
            return self.derived_catalog[line_field[0]][self.derived_cat_entry, line_field[1]]        
        
if __name__ == '__main__':
    # cube = CALIFACube(path='NGC0677')
    # cube.get_flux()
    # cube.get_wavelength(to_rest_frame=True)
    # cube.get_bad_pixels()

    # wl = cube.wl
    # bad_pixels = cube.bad_pix

    # cols = np.arange(10, 70, 5)
    # rows = np.arange(10, 70, 5)
    # for row_i in rows:
    #     for col_i in cols:
    #         row = row_i
    #         col = col_i
    #         name = 'PGC11179_'+str(row)+'_'+str(col)
    #         plt.figure()
    #         plt.title(name)
    #         plt.errorbar(wl, cube.flux[:, row, col], yerr=cube.flux_error[:, row, col],
    #              alpha=1, fmt='.', ecolor='k')
    #         plt.scatter(wl[cube.bad_pix[:, row, col]],
    #             cube.flux[cube.bad_pix[:, row, col], row, col], c='r', zorder=10)
    #         plt.yscale('log')
    #         plt.ylim(1e-19, 1e-15)
    #         plt.savefig('tests/QC/'+name+'.png')
    #         plt.close()

    # cube.mask_bad()
    # flux = cube.flux
    # flux_error = cube.flux_error

    # red_band = (wl>6540)&(wl<6580)

    # ref_image = np.nanmean(flux[red_band, :, :], axis=0)
    # ref_image[ref_image<=0] = np.nan
    # # noise_i = np.sqrt(np.nansum(error[red_band, :, :]**2, axis=0))
    # ref_noise = np.nanmean(flux_error[red_band, :, :], axis=0)
    # ref_noise[ref_noise<=0] = np.nan

    # very_low_sn = ref_image/ref_noise < 0.01
    # ref_image[very_low_sn] = np.nan
    # ref_noise[very_low_sn] = np.nan

    # cube.voronoi_binning(ref_image=ref_image, ref_noise=ref_noise, targetSN=30)
    # cube.bin_cube()

    # plt.figure()
    # plt.imshow(cube.bin_map, cmap='flag')

    cube = MANGACube(plate='9035', ifudesign='3704')
    cube.get_flux()
    cube.get_wavelength(to_rest_frame=True)
    cube.get_bad_pixels()    
    ha = cube.get_derived_catalog(line_field=['EMLINE_SFLUX_1RE', 18])
    hb = cube.get_derived_catalog(line_field=['EMLINE_SFLUX_1RE', 11])
