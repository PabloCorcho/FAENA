#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 17:08:29 2021

@author: pablo
"""

import numpy as np
import os
from astropy.modeling import models, fitting
from matplotlib import pyplot as plt


def fit_line(wl, spec, line_wl, fitter=fitting.LevMarLSQFitter()):
    """Fit gaussian profile to emission line."""
    init_mu = line_wl
    init_a = np.interp(np.array(init_mu), wl, spec)
    init_sigma = init_mu * 100/3e5
    max_sigma = init_mu * 300/3e5
    model_init = models.Gaussian1D(init_a, init_mu, init_sigma)
    model_init.mean.bounds = [init_mu-max_sigma, init_mu+max_sigma]
    model_init.stddev.bounds = [0, max_sigma]
    fit = fitter(model_init, wl, spec)
    return fit


def line_flux(wl, spec, line_wl, spec_err=None, wl_lims=None, plot=False,
              **plotargs):
    """Measuring raw fluxes from straight integration."""
    flux_err = np.nan
    if wl_lims is not None:
        pts = np.where((wl >= wl_lims[0]) & (wl <= wl_lims[1]))[0]
        # this accounts for resolution effects
        res_corr = (wl_lims[1] - wl_lims[0]) / (wl[pts][-1] - wl[pts][0])
    else:
        delta_lambda = line_wl * 300/3e5
        pts = np.where((wl >= line_wl - delta_lambda) &
                       (wl <= line_wl + delta_lambda))[0]
        res_corr = (2 * delta_lambda) / (wl[pts][-1] - wl[pts][0])

    if len(spec.shape) > 1:
        flux = np.trapz(spec[pts, :], wl[pts], axis=0)
        if spec_err is not None:
            flux_err = np.sqrt(np.trapz(spec_err[pts, :]**2, wl[pts], axis=0))
    else:
        flux = np.trapz(spec[pts], wl[pts])
        if spec_err is not None:
            flux_err = np.sqrt(np.trapz(spec_err[pts]**2, wl[pts]))
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.errorbar(wl[pts], spec[pts], yerr=spec_err)
        ax.annotate('F={:2.4f}, F_err={:2.4f}'.format(flux, flux_err[pts]),
                    xy=(.05, .95), va='top', ha='left')
        fig.close()
        return flux * res_corr, flux_err * res_corr, fig
    else:
        return flux * res_corr, flux_err * res_corr

realpath = os.path.abspath(__file__)
lim = realpath.find('emission_lines.py')
realpath = realpath[:lim]
emission_lines = np.loadtxt(os.path.join(realpath, 'emission_lines.txt'), 
                            usecols=(0))
emission_lines_names = np.loadtxt(os.path.join(realpath, 'emission_lines.txt'), 
                                  usecols=(1), dtype=str)

class MaskEmission(object):
        
    def mask_emission_lines(wave, window_width=30):
        emission_lines_width = [window_width]*len(emission_lines)
        
        mask = np.ones_like(wave, dtype=bool)
        lines_masked = {}
        for line, name, width in zip(emission_lines, emission_lines_names, emission_lines_width):
            line_mask = (wave > line-width/2) & (wave < line+width/2)
            if line_mask.any():
                lines_masked[name] = line
                mask[line_mask] = False
        
        return mask, lines_masked
        
    def get_continuum(wave, flux, weights=None, fit_deg='auto'):
        if fit_deg =='auto':
            fit_deg = (wave[-1]-wave[0])//200            
        mask, _ = MaskEmission.mask_emission_lines(wave)
        if weights.any():
            coeff = np.polyfit(wave[mask], flux[mask], deg=fit_deg, w=weights[mask])
        else:
            coeff = np.polyfit(wave[mask], flux[mask], deg=fit_deg)
        pol = np.poly1d(coeff)
        return pol(wave)
            
    def identify_emission_lines(wave, flux, weights=None, fit_deg='auto', thresh=0.3):        
        continuum = MaskEmission.get_continuum(wave, flux, weights, fit_deg)
        residuals = (flux-continuum)/continuum
        # Deviations *above* thresh value will be considered as potential emission lines
        emission_wl = np.where(residuals>thresh)[0] 
        
        mask, lines_masked = MaskEmission.mask_emission_lines(wave[emission_wl], window_width=10)
        return lines_masked
        
        
        
