#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 18:17:58 2022

@author: pablo
"""

from astropy.modeling.functional_models import Sersic2D
from photutils.centroids import centroid_com
from astropy.modeling import fitting
import warnings
import numpy as np
from scipy.special import gammaincinv
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

class SersicFit(object):
    """Fit Sersic 2D models to astronomical images."""

    def __init__(self, image, n_profiles=1, **kwargs):
        self.image = image.copy()
        self.image = np.nan_to_num(self.image, nan=0)
        # self.image /= np.nanmax(self.image)
        # Array of coordinates
        self.XX, self.YY = np.meshgrid(np.arange(self.image.shape[0]),
                                       np.arange(self.image.shape[1]),
                                       indexing='ij')
        # Set model params
        self.n_profiles = n_profiles
        self.initial_guess = kwargs.get('initial_guess',
                                        self.set_initial_guess())
        self.bounds = kwargs.get('bounds', self.set_bounds())

        print('--> Initial guess')
        for key, val in self.initial_guess.items():
            print('   ·', key, ': ', val,
                  '  ({}, {})'.format(self.bounds[key][0],
                                      self.bounds[key][1]))

    def set_initial_guess(self):
        """Estimate an initial guess for the fit."""
        # Create initial guess
        # Center of mass
        x_o_g, y_o_g = centroid_com(self.image)
        com = np.array([x_o_g, y_o_g])
        # Amplitude
        p0_amplitude = np.median(self.image[int(x_o_g)-10:int(x_o_g)+10,
                                            int(y_o_g)-10:int(y_o_g)+10])
        # Effective Radii
        p0_r_eff = (self.image.shape[0] + self.image.shape[1])/4
        # Sersic index
        p0_n = 2.
        # Ellipticity
        p0_ellip = 0.5
        # Rotation angle
        p0_theta = 0.
        initial_guess = {}
        if self.n_profiles > 1:
            for i in range(self.n_profiles):
                initial_guess['amplitude_{}'.format(i)] = p0_amplitude
                initial_guess['r_eff_{}'.format(i)] = p0_r_eff
                initial_guess['n_{}'.format(i)] = p0_n
                initial_guess['x_0_{}'.format(i)] = com[0]
                initial_guess['y_0_{}'.format(i)] = com[1]
                initial_guess['ellip_{}'.format(i)] = p0_ellip
                initial_guess['theta_{}'.format(i)] = p0_theta
        else:
            initial_guess['amplitude'] = p0_amplitude
            initial_guess['r_eff'] = p0_r_eff
            initial_guess['n'] = p0_n
            initial_guess['x_0'] = com[0]
            initial_guess['y_0'] = com[1]
            initial_guess['ellip'] = p0_ellip
            initial_guess['theta'] = p0_theta
        return initial_guess

    def set_bounds(self):
        """Set bounds for each parameter."""
        bounds = {}
        if self.n_profiles > 1:
            for i in range(self.n_profiles):
                bounds['amplitude_{}'.format(i)] = (0, None)
                bounds['r_eff_{}'.format(i)] = (0, None)
                bounds['n_{}'.format(i)] = (0, 10)
                bounds['x_0_{}'.format(i)] = (None, None)
                bounds['y_0_{}'.format(i)] = (None, None)
                bounds['ellip_{}'.format(i)] = (None, None)
                bounds['theta_{}'.format(i)] = (None, None)
        else:
            bounds['amplitude'] = (0, None)
            bounds['r_eff'] = (0, None)
            bounds['n'] = (0, 10)
            bounds['x_0'] = (None, None)
            bounds['y_0'] = (None, None)
            bounds['ellip'] = (None, None)
            bounds['theta'] = (None, None)
        return bounds

    def sersic(self, x, y, amplitude, r_eff, n, x_0, y_0, ellip, theta):
        """Astropy 2D Sersic model."""
        bn = gammaincinv(2. * n, 0.5)
        a, b = r_eff, (1 - ellip) * r_eff
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        r_maj = (x - x_0) * cos_theta + (y - y_0) * sin_theta
        r_min = -(x - x_0) * sin_theta + (y - y_0) * cos_theta
        rad = np.sqrt((r_maj / a) ** 2 + (r_min / b) ** 2)
        sersic = amplitude * np.exp(-bn * (rad ** (1 / n) - 1))
        return rad, sersic

    def fit(self):
        """Fit profile."""
        p_init = Sersic2D()
        for i in range(self.n_profiles - 1):
            p_init += Sersic2D()
        for key in p_init.bounds.keys():
            setattr(p_init, key, self.initial_guess[key])
            p_init.bounds[key] = self.bounds[key]
        fitter = fitting.LevMarLSQFitter()

        with warnings.catch_warnings():
            # Ignore model linearity warning from the fitter
            # warnings.simplefilter('ignore')
            p = fitter(p_init, self.XX, self.YY, self.image,
                       # maxiter=100000,
                       # ftol=1.4901161193847656e-08,
                       # acc=1e-09
                       )
        self.model = p
        self.model_data = self.model(self.XX, self.YY)

    def get_fit_mask(self, n_eff_rad=1):
        """..."""
        mask = np.zeros_like(self.model_data, dtype=bool)
        mask[self.model_data >= n_eff_rad * self.amplitude] = True
        return mask

    def plot_fit(self, save=False):
        """Plot results from fit."""
        im_map = 'nipy_spectral'
        vmax = np.nanpercentile(np.log10(self.image), 95)
        vmin = np.nanpercentile(np.log10(self.image), 5)
        norm = LogNorm()
        imshow_args = dict(cmap=im_map, origin='lower', aspect='auto',
                           # vmin=vmin, vmax=vmax, 
                           norm=norm)
        fig = plt.figure(figsize=(9, 3))
        ax = fig.add_subplot(131)
        mappable = ax.imshow(self.image, **imshow_args)
        plt.colorbar(mappable, ax=ax, orientation='horizontal',
                     label=r'$\log(I)$')
        ax = fig.add_subplot(132)
        mappable = ax.imshow(self.model_data, **imshow_args)
        plt.colorbar(mappable, ax=ax, orientation='horizontal',
                     label=r'$\log(\hat{I})$')
        ax = fig.add_subplot(133)
        mappable = ax.imshow(np.abs(self.image - self.model_data) / self.image,
                             origin='lower',
                             cmap='nipy_spectral', aspect='auto',
                             vmin=0, vmax=1
                             )
        plt.colorbar(mappable, ax=ax, orientation='horizontal',
                     label=r'$\frac{|I - \hat{I}|}{I}$')
        fig.subplots_adjust(wspace=0.4)
        if save:
            fig.savefig(save, bbox_inches='tight')
        return fig

