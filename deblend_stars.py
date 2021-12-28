#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 11:05:50 2021

@author: pablo
"""

from photutils.centroids import centroid_2dg
from photutils.segmentation import detect_threshold, detect_sources
from photutils.segmentation import deblend_sources
from astropy.convolution import Gaussian2DKernel

import astropy.io.fits as fits
from scipy.optimize import curve_fit
import numpy as np
from matplotlib import pyplot as plt


def gaussian2d(vector, amplitude, x0, y0, sigmax, sigmay, rho):
    x = vector[0]
    y = vector[1]
    g = amplitude*np.exp(- 0.5*(x - x0)**2/2/sigmax**2
                         - 0.5*(y - y0)**2/2/sigmay**2
                         + rho*(x - x0)*(y - y0)/sigmax/sigmay)
    return g.ravel()


def deblend_stars(image, ew_map=None, sigma_thresh=3., fwhm_pix=3.,
                  npixels=15, nlevels=32, contrast=0.01,
                  size_thresh=2.,
                  dist_to_center_thresh=10.,
                  area_thresh=200.,
                  ew_thresh=20.,
                  output_dir='', save_fits=False):
    """
    This method estimates foreground stars over a given image.
    Additional constrains can be applied by providing an EW(Ha) map, preventing
    giant HII regions to be classified as stars.
    ---------------------------------------------------------------------------
    input params:
        - image: (2D array) Flux image used to deblend the sources
        - ew_map (optional): (2D array) EW(Ha) map used to further
        constrain the sources
        - sigma_thresh (float, default=3.0): number of stddev for estimating
        the signal threshold
        - fwhm_pix (float, default=3.0): FWHM for Gaussian smoothing prior to
        source detection
        - npixels (int, default=15): The number of connected pixels, each
        greater than threshold, that an object must have to be detected.
        - nlevels (int, default=15): The number of multi-thresholding levels
        to use
        - contrast (float, default=0.01)
        - size_thres (float, defualt=2.)
        -
        -
        - output_dir (optional, defatult=''): (str) directory where the results
        will be saved
        - save_filts (optional, default=False): Set True for saving the stellar
        masks
    """
    if ew_map is None:
        ew = np.zeros_like(image)
    rows, columns = image.shape
    centerx, centery = rows//2, columns//2
    XX, YY = np.meshgrid(np.arange(columns), np.arange(rows))
    # Estimating the image signal threshold
    threshold = detect_threshold(image, nsigma=sigma_thresh)
    # Kernel smoothing
    # FWHM_pixels / conversion_factor
    kernel_sigma = fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    kernel = Gaussian2DKernel(kernel_sigma,
                              x_size=int(fwhm_pix),
                              y_size=int(fwhm_pix))
    kernel.normalize()
    # Source detection
    segm = detect_sources(image, threshold, npixels,
                          filter_kernel=kernel)
    # Source deblending
    segm_deblend = deblend_sources(image, segm, npixels=npixels,
                                   filter_kernel=kernel, nlevels=nlevels,
                                   contrast=0.01)
    deblended_areas = segm_deblend.areas
    stellar_masks = []

    if deblended_areas.size > 1:
        mask = segm_deblend.data
        cmap = segm_deblend.make_cmap()

        plt.figure()
        plt.imshow(segm_deblend, origin='lower', cmap=cmap)
        plt.colorbar()
        plt.contour(image, levels=20, colors='k', origin='lower')
        for ii in range(deblended_areas.size):
            plt.annotate(r'Pixel area: {}'.format(deblended_areas[ii]),
                         xy=(.02, .95-ii*0.05),
                         xycoords='axes fraction', ha='left', va='top',
                         color=cmap(ii+1), fontsize=9)
        plt.savefig(output_dir+'deblending_map.png')
        plt.close()

        for ii in range(deblended_areas.size):
            source = mask == ii+1
            source_area = deblended_areas[ii]
            source_image = np.copy(image)
            source_image[~source] = 0
            pos_x, pos_y = centroid_2dg(source_image)
            dist_to_center = np.sqrt((pos_x-centerx)**2+(pos_y-centery)**2)
            amplitude_guess = source_image.max()
            initial_guess = [amplitude_guess, pos_x, pos_y, 3, 3, 0]
            try:
                popt, pcov = curve_fit(gaussian2d, [XX, YY],
                                       source_image.ravel(),
                                       p0=initial_guess,
                                       # bounds=(0, [amplitude_guess*2, 80, 80, 5, 5])
                                       )
            except:
                print('Error fitting gaussian to data')
                continue
            popt = np.abs(popt)
            sigmax, sigmay = popt[3], popt[4]
            gaussian = gaussian2d([XX, YY], *popt).reshape(image.shape)                        
            level = gaussian2d([popt[1]+3*max([popt[3], 2]),
                                popt[2]+3*max([popt[4], 2])], *popt)
            central_pixels = gaussian2d([popt[1]+sigmax, popt[2]+sigmay],
                                        *popt)
            star_mask = gaussian > level
            central_star_mask = gaussian > central_pixels
            ew_source = ew[central_star_mask]
            median_ew = np.nanmedian(ew_source)
            if np.isnan(median_ew):
                median_ew = 0.
            ellipticity = sigmax/sigmay
            # chi2 = np.sum(chi2[star_mask])
            if (sigmax < size_thresh) & (sigmay < size_thresh):
                # &(ellipticity<1.5)&(ellipticity>0.5)
                if (dist_to_center < dist_to_center_thresh) & (
                        source_area < area_thresh) & (median_ew <= ew_thresh):
                    is_star = True
                    stellar_masks.append(star_mask)
                elif (dist_to_center > dist_to_center_thresh) & (
                        median_ew <= ew_thresh):
                    is_star = True
                    stellar_masks.append(star_mask)
                else:
                    is_star = False
            else:
                is_star = False

            plt.figure(figsize=(4, 4))
            plt.subplot(221)
            plt.plot(XX[0, :], np.sum(source_image, axis=0), 'k')
            plt.plot(XX[0, :], np.sum(gaussian, axis=0), 'r')
            plt.annotate(r'$\sigma_x$={:5.3}'.format(sigmax), xy=(.1, .9),
                         xycoords='axes fraction', ha='left', va='top')
            plt.subplot(224)
            plt.plot(np.sum(source_image, axis=1), YY[:, 0], 'k')
            plt.plot(np.sum(gaussian, axis=1), YY[:, 0], 'r')
            plt.annotate(r'$\sigma_y$={:5.3}'.format(sigmay), xy=(.9, .9),
                         xycoords='axes fraction', ha='right', va='top')
            plt.subplot(223)
            plt.imshow(image, cmap='gist_earth', aspect='auto',
                       origin='lower')
            if is_star:
                plt.plot(pos_x, pos_y, '*', c='lime', markersize=10)
            else:
                plt.plot(pos_x, pos_y, '+', c='lime', markersize=10)
            plt.contour(gaussian, colors='r', levels=level,
                        linewidths=2)
            plt.annotate(r'$\sigma_x/\sigma_y$={:5.3}'.format(ellipticity),
                         xy=(.05, .99),
                         xycoords='axes fraction', ha='left', va='top',
                         fontsize=8)
            plt.annotate(r'$EW(H\alpha)_{50}$'+'={:5.3}'.format(median_ew),
                         xy=(.05, .90),
                         xycoords='axes fraction', ha='left', va='top',
                         fontsize=8)
            plt.savefig(output_dir+'detection_'+str(ii)+'.png')
            plt.close()
            # plt.xlim(pos_x-15,pos_x+15)
            # plt.ylim(pos_y-15,pos_y+15)

    stellar_masks = np.array(stellar_masks)

    if stellar_masks.size > 0:
        print('-->', stellar_masks.shape[0], ' stars detected')
        total_mask = np.zeros_like(image, dtype=bool)
        for i in range(stellar_masks.shape[0]):
            total_mask = total_mask | np.array(stellar_masks[i, :, :],
                                               dtype=bool)
        if save_fits:
            fits_path = output_dir+'stellar_mask.fits'
            hdr = fits.Header()
            hdr['COMMENT'] = "Stellar masks"
            image_list = []
            image_list.append(fits.PrimaryHDU(header=hdr))
            for ii in range(stellar_masks.shape[0]):
                image_list.append(
                    fits.ImageHDU(np.array(stellar_masks[ii, :, :],
                                           dtype=int)))
            image_list.append(np.array(total_mask, dtype=int))
            hdu = fits.HDUList(image_list)
            hdu.writeto(fits_path, overwrite=True)
            hdu.close()
            print('File saved as: ' + fits_path)
        return total_mask
    else:
        return np.zeros_like(image, dtype=bool)