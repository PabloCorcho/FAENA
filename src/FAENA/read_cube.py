#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 11:33:31 2021

@author: pablo
"""

from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
from astropy.coordinates import SkyCoord
from astropy import units as units

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

    Default flux units: 'erg/s/cm2/AA'
    Default wavelength units: AA

    """

    def __init__(self):

        self.read_cube()

    def read_cube(self):
        """todo."""
        self.cube = fits.open(self.path_to_cube)

    def close_cube(self):
        """todo."""
        self.cube.close()

    def get_axis(self):
        """todo."""
        # n_hdu = len(self.cube)
        pass

    def get_wcs(self):
        """todo."""
        from astropy.wcs import WCS

        self.wcs = WCS(self.cube[1].header)

    def get_obj_central_pixels(self):
        """If the centre coordinates of the object are provided this function will return the corresponding pixels values."""
        self.pix_centre = self.wcs.world_to_pixel(
            SkyCoord(self.obj_ra, self.obj_dec, unit='deg'),
            self.wl[0]*units.angstrom)
        self.pix_centre = (self.pix_centre[0], self.pix_centre[1])

    def get_luminosity(self):
        """todo."""
        try:
            self.redshift
        except Exception:
            self.get_redshift()

        self.comoving_distance = cosmo.comoving_distance(self.redshift)  # Mpc
        self.comoving_distance = self.comoving_distance.to('cm').value

        self.luminosity = self.flux * 4*np.pi*self.comoving_distance**2
        self.luminosity_error = (self.flux_error *
                                 4*np.pi * self.comoving_distance**2)

    def get_surface_brightness(self):
        """todo."""
        self.surface_brightness = self.flux/self.fiber_surface
        self.surface_brightness_error = self.flux_error/self.fiber_surface

    def to_rest_frame(self):
        """todo."""
        try:
            self.redshift
        except Exception:
            self.get_redshift()

        if not self.rest_frame:
            self.wl = self.wl/(1+self.redshift)
            self.rest_frame = True
        print('Wavelength in rest frame')

    def voronoi_binning(self, ref_image, ref_noise, targetSN,
                        plot_binning=False):
        """todo."""
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
        """todo."""
        self.binned_flux = np.zeros((self.flux.shape[0], self.nbins))
        self.binned_flux_error = np.zeros_like(self.binned_flux)

        for ith in range(self.nbins):
            mask = self.bin_map == ith + 1
            flux_error_ith = self.flux_error[:, mask]
            flux_ith = self.flux[:, mask]
            bad = flux_error_ith/flux_ith > 1000
            flux_error_ith[bad] = flux_error_ith[~bad].max()

            weights = self.get_bin_err_weights(mask)

            self.binned_flux[:, ith] = np.nansum(self.flux[:, mask], axis=(1))
            self.binned_flux_error[:, ith] = np.sqrt(
                np.nansum(flux_error_ith**2*weights**2,
                          axis=(1)))

        # self.binned_flux_error[self.binned_flux_error == 0] = np.nan
        # self.binned_flux[self.binned_flux == 0] = np.nan

    def compute_circular_aperture(self, centre, radius, flux_return=False,
                                  plot=False):
        """todo."""
        from photutils.aperture import CircularAperture
        aperture = CircularAperture(centre, r=radius)
        aperture_mask = aperture.to_mask()
        aperture_mask = np.array(
            aperture_mask.to_image(self.wcs.celestial.array_shape), dtype=bool)

        if flux_return:
            integrated_flux = np.nansum(self.flux[:, aperture_mask], axis=1)
            integrated_no_cov = np.nansum(self.flux_error[:, aperture_mask]**2,
                                          axis=1)
            # Accounting for covariance errors
            if aperture_mask[aperture_mask].size < 100:
                integrated_flux_cov = integrated_no_cov*(
                    1+1.62*np.log10(aperture_mask[aperture_mask].size))
                integrated_flux_err = np.sqrt(integrated_flux_cov)
            else:
                integrated_flux_cov = integrated_no_cov*4.2
                integrated_flux_err = np.sqrt(integrated_flux_cov)
        else:
            integrated_flux = None
            integrated_flux_err = None
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.imshow(self.flux[self.wl.size//2, :, :], cmap='gist_earth_r')
            aperture.plot(lw=1, color='r')
            ax.annotate(r'$R_e={:4.3}~(Kpc)$'.format(self.eff_radius_physical),
                        xy=(.9, .95), xycoords='axes fraction', va='top',
                        ha='right')
            ax.annotate(r'$R_e={:4.3}~(arcsec)$'.format(self.eff_radius),
                        xy=(.9, .9), xycoords='axes fraction', va='top',
                        ha='right')
            aperture.plot(lw=1, color='r')
            ax.annotate(r'$R_e={:4.3}~(pix)$'.format(self.eff_radius_pix),
                        xy=(.9, .85), xycoords='axes fraction', va='top',
                        ha='right')
            ax.annotate(r'$R_a={:4.3}~(pix)$'.format(radius),
                        xy=(.9, .80), xycoords='axes fraction', va='top',
                        ha='right', color='r')
            aperture.plot(lw=1, color='r')
            # plt.savefig('bpt_apertures/'+name_i+'.png')
            plt.show()
            plt.close()
            return (aperture, aperture_mask, integrated_flux,
                    integrated_flux_err, fig)
        else:
            return (aperture, aperture_mask, integrated_flux,
                    integrated_flux_err)

# =============================================================================
#
# =============================================================================


class CALIFACube(Cube):
    """
    Reads CALIFA cubes.

    - mode: 'COMB' (only DR3), 'V500', 'V1200'
    - data_release: 'DR2', 'DR3'
    """

    def __init__(self, path, mode='V500', abs_path=False, data_release='DR3'):
        self.pixel_size = 1  # arcsec
        self.pixel_surface = self.pixel_size**2  # arcsec^2
        self.mode = mode
        self.data_release = data_release

        if abs_path:
            self.path_to_cube = path
        else:
            self.path_to_cube = (
                '/home/pablo/obs_data/CALIFA/'+data_release+'/'
                + mode+'/cubes/'+path + '.'+mode + '.rscube.fits.gz')

        print('\nOpening CALIFA cube: ', self.path_to_cube, '\n')

        Cube.__init__(self)

        self.califaid = self.cube[0].header['CALIFAID']
        self.name = self.cube[0].header['object']

    def get_flux(self):
        """todo."""
        self.flux = self.cube[0].data*1e-16
        self.flux_units = 'erg/s/cm2/AA'
        self.flux_error = self.cube[1].data*1e-16

        print('Flux units:', self.flux_units)

    def get_wavelength(self, to_rest_frame=False):
        """todo."""
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
        """BAD PIXELS: 1 == GOOD, 0 == BAD."""
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
        """todo."""
        self.error_weights = self.cube[2].data

    def get_redshift(self):
        """todo."""
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
                    '\nWARNING: CALIFA REDSHIFT UNRELIABLE \n-->'
                    + 'SELECTING NED REDSHIFT\n')
                self.redshift = ned_redshift[ned_pos]

        # recesion_vel = self.cube[0].header['MED_VEL']
        # self.redshift = recesion_vel/3e5

    def mask_bad(self):
        """todo."""
        print('MASKING BAD PIXELS WITH "NAN"')
        self.flux[self.bad_pix] = np.nan
        self.flux_error[self.bad_pix] = np.nan
        bad = self.flux_error/self.flux > 1000
        self.flux_error[bad] = self.flux_error[~bad].max()

    def coadded_spectral_empirical_correlation(self, n_spaxels):
        """todo."""
        alpha = {'V500': 1.10, 'V1200': 1.08, 'COMB': 1.08}
        beta = 1 + alpha[self.mode]*np.log10(n_spaxels)
        return beta

    def get_bin_err_weights(self, spaxels_mask):
        """todo."""
        n_spaxels = spaxels_mask[spaxels_mask].size
        if n_spaxels < 80:
            weights = np.array(
                [self.coadded_spectral_empirical_correlation(
                    n_spaxels)]*n_spaxels)
        else:
            try:
                self.error_weights
            except Exception:
                self.get_errorweigths()

            weights = self.error_weights[:, spaxels_mask]
        return weights

# =============================================================================
#
# =============================================================================


class MANGACube(Cube):
    """todo."""

    def __init__(self, plate=None, ifudesign=None, path=None, logcube=True,
                 abs_path=False):
        self.pixel_size = 0.5  # arcsec
        self.pixel_surface = self.pixel_size**2  # arcsec^2
        self.plate, self.ifudesign = plate, ifudesign
        cube_wl_sampling = {True: 'LOGCUBE.fits.gz', False: 'LINCUBE.fits.gz'}
        if abs_path:
            self.path_to_cube = path
            self.plate, self.ifudesign = path.split('/')[-1].split('-')[1:3]
        else:
            self.path_to_cube = '/home/pablo/obs_data/MANGA/cubes/' +\
                                'manga-'+self.plate+'-'+self.ifudesign+'-' +\
                                cube_wl_sampling[logcube]
        Cube.__init__(self)

    def get_flux(self):
        """todo."""
        self.flux = self.cube[1].data*1e-17
        self.flux_units = 'erg/s/cm2/AA/spaxel'
        self.flux_error = 1/np.sqrt(self.cube[2].data) * 1e-17
        self.flux_error[np.isinf(self.flux_error)] = np.nan
        print('Flux units:', self.flux_units)

    def get_wavelength(self, to_rest_frame=False):
        """todo."""
        self.wl = self.cube[6].data
        self.rest_frame = False
        if to_rest_frame:
            self.to_rest_frame()
        else:
            print('Wavelength vector is not in rest frame')

    def get_bad_pixels(self):
        """todo."""
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

    def get_image(self, band='r'):
        """todo."""
        keys = {'g': 12, 'r': 13, 'i': 14, 'z': 15}
        return self.cube[keys[band]].data

    def mask_bad(self):
        """todo."""
        print('MASKING BAD PIXELS WITH "NAN"')
        self.flux[self.bad_pix] = np.nan
        self.flux_error[self.bad_pix] = np.nan
        bad = self.flux_error/self.flux > 1000
        self.flux_error[bad] = self.flux_error[~bad].max()

    def get_redshift(self):
        """todo."""
        self.redshift = self.get_catalog(field='NSA_Z')

    def get_catalog(self, field=None):
        """todo."""
        self.catalog_path = 'drpall-v3_1_1.fits'
        try:
            self.catalog
        except Exception:
            with fits.open('input_data/ifu_catalogs/MANGA/'
                           + self.catalog_path) as f:
                self.catalog = f[1].data
            self.cat_entry = np.where(
                self.catalog['plateifu'] == self.plate
                + '-' + self.ifudesign)[0][0]
        if field:
            return self.catalog[field][self.cat_entry]

    def get_derived_catalog(self, field=None, line_field=None):
        """todo."""
        self.derived_catalog_path = 'dapall-v3_1_1-3.1.0.fits'
        try:
            self.derived_catalog
        except Exception:
            with fits.open('input_data/ifu_catalogs/MANGA/'
                           + self.derived_catalog_path) as f:
                self.derived_catalog = f[1].data
                self.derived_catalog_lines = f[0].header
                self.derived_cat_entry = np.where(
                    self.derived_catalog['plateifu'] == self.plate
                    + '-' + self.ifudesign)[0][0]
        if field:
            return self.derived_catalog[field][self.derived_cat_entry]
        if line_field:
            return (self.derived_catalog[line_field[0]][self.derived_cat_entry,
                                                        line_field[1]])

    def get_galaxy_centre(self):
        """todo."""
        self.obj_ra = self.get_catalog(field='OBJRA')
        self.obj_dec = self.get_catalog(field='OBJDEC')

    def get_eff_radius(self):
        """todo."""
        self.eff_radius = self.get_catalog(field='NSA_SERSIC_TH50')  # arcsec
        self.eff_radius_pix = self.eff_radius / self.pixel_size
        angular_distance = cosmo.angular_diameter_distance(self.redshift).value
        self.eff_radius_physical = (angular_distance*self.eff_radius/3600
                                    * np.pi/180 * 1000)  # kpc

    def coadded_spectral_empirical_correlation(self, n_spaxels):
        """todo."""
        alpha = {'V500': 1.10, 'V1200': 1.08, 'COMB': 1.08}
        beta = 1 + alpha['V500']*np.log10(n_spaxels)
        return beta

    def get_bin_err_weights(self, spaxels_mask):
        """todo."""
        n_spaxels = spaxels_mask[spaxels_mask].size
        weights = np.array(
                [self.coadded_spectral_empirical_correlation(
                    n_spaxels)]*n_spaxels)
        return weights


class SAMICube(Cube):
    """
    SAMI CUBE ENTRIES:
    --------------------------------------------------------------------------
    0  PRIMARY       1 PrimaryHDU      83   (50, 50, 2048)   float64   
    1  VARIANCE      1 ImageHDU         9   (50, 50, 2048)   float64   
    2  WEIGHT        1 ImageHDU         9   (50, 50, 2048)   float64   
    3  COVAR         1 ImageHDU       464   (50, 50, 5, 5, 451)   float64   
    4  QC            1 BinTableHDU     33   7R x 12C   [20A, E, E, E, E, E, E, E, E, E, E, E]   
    5  DUST          1 ImageHDU         9   (2048,)   float64 
    --------------------------------------------------------------------------
    """

    def __init__(self, catid=None, arm=None, abs_path=False):                        
        if abs_path:
            self.path_to_cube = abs_path
        else:
            self.path_to_cube = '/home/pablo/obs_data/SAMI/cubes/' +\
                                '{}_A_cube_{}.fits.gz'.format(catid, arm)
        Cube.__init__(self)

    def get_flux(self):
        self.hdr = self.cube[0].header
        self.flux = self.cube[0].data
        self.flux_units = self.hdr['BUNIT']
        self.flux_error = np.sqrt(self.cube[1].data)
        # self.flux_error[np.isinf(self.flux_error)] = np.nan
        print('Flux units:', self.flux_units)

    def get_wavelength(self, to_rest_frame=False):
        self.wl = self.hdr['CRVAL3']+(np.arange(self.flux.shape[0])-self.hdr['CRPIX3'])*self.hdr['CDELT3']
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
        self.redshift = self.hdr['Z_SPEC'] # Heliocentric redshift from input catalogue
                
    
if __name__ == '__main__':
    # cube = CALIFACube(path='NGC0237')
    # cube.get_flux()
    # cube.get_wavelength(to_rest_frame=True)
    # cube.get_bad_pixels()

    # wl = cube.wl
    # bad_pixels = cube.bad_pix

    # # cols = np.arange(10, 70, 5)
    # # rows = np.arange(10, 70, 5)
    # # for row_i in rows:
    # #     for col_i in cols:
    # #         row = row_i
    # #         col = col_i
    # #         name = 'PGC11179_'+str(row)+'_'+str(col)
    # #         plt.figure()
    # #         plt.title(name)
    # #         plt.errorbar(wl, cube.flux[:, row, col], yerr=cube.flux_error[:, row, col],
    # #               alpha=1, fmt='.', ecolor='k')
    # #         plt.scatter(wl[cube.bad_pix[:, row, col]],
    # #             cube.flux[cube.bad_pix[:, row, col], row, col], c='r', zorder=10)
    # #         plt.yscale('log')
    # #         plt.ylim(1e-19, 1e-15)
    # #         plt.savefig('tests/QC/'+name+'.png')
    # #         plt.close()

    # cube.mask_bad()
    # flux = cube.flux
    # flux_error = cube.flux_error

    # red_band = (wl>4540)&(wl<6580)

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
    
    # from measurements.brightness_profile import SersicFit, half_mass
    # Sfit = SersicFit(ref_image)    
    # Sfit.fit()
    # Sfit.plot_fit()
    
    
    cube = MANGACube(plate='9035', ifudesign='3704')
    cube.get_flux()
    cube.get_wavelength(to_rest_frame=True)
    cube.get_bad_pixels()    
    cube.get_wcs()
    cube.get_galaxy_centre()
    cube.get_obj_central_pixels()
    cube.get_eff_radius()
    
    from measurements.brightness_profile import SersicFit
    
    r_image = cube.cube[13].data
    Sfit = SersicFit(r_image)    
    Sfit.fit()
    Sfit.plot_fit()
    a, b, c, d, e = cube.compute_circular_aperture(centre=cube.pix_centre, 
                                    radius=cube.eff_radius_pix, plot=True,
                                    flux_return=True)
    a, b, c, d, e = cube.compute_circular_aperture(centre=cube.pix_centre, 
                                    radius=Sfit.reff, plot=True,
                                    flux_return=True)
    
    
