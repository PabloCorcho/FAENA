#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 13:13:37 2021

@author: pablo
"""

import numpy as np


def equivalent_width(wl, spec, central_lims, left_lims, right_lims,
                     spec_err=None):
    """Compute the equivalent width of a spectral region.

    Parameters
    ----------
    - wl: wavelength vector in angstrom. Must be linearly sampled.
    - spec: flux density spectra per wavelength unit.
    - central_lims: wavelength window in AA to estimate the EW.
    - left_lims: wavelength window in AA to estimate pseudocontinuum.
    - right_lims: wavelength window in AA to estimate pseudocontinuum.
    - spec_err: (array) Vector containing the spectra uncertainties
    per wavelength unit
    Returns
    -------
    - flux: (float) mean flux density per wave unit within central wavelength
    window.
    - pseudocont: (float) mean pseudocontinuum flux density per wave unit
    interpolated to the central wavelength window.
    - ew: (float) Measured equivalent width.
    """
    left_wl = np.array(left_lims)
    mean_left_wl = left_wl.mean()
    right_wl = np.array(right_lims)
    mean_right_wl = right_wl.mean()
    lick_wl = np.array(central_lims)
    mean_lick_wl = lick_wl.mean()
    delta_lick_wl = lick_wl[1] - lick_wl[0]
    left_pts = np.where((wl >= left_wl[0]) & (wl <= left_wl[1]))[0]
    right_pts = np.where((wl >= right_wl[0]) & (wl <= right_wl[1]))[0]
    lick_pts = np.where((wl >= lick_wl[0]) & (wl <= lick_wl[1]))[0]
    right_weight = (mean_lick_wl - mean_left_wl
                    ) / (mean_right_wl - mean_left_wl)
    left_weight = 1 - right_weight

    if len(spec.shape) > 1:
        left_cont = np.nanmean(spec[left_pts, :], axis=0)
        right_cont = np.nanmean(spec[right_pts, :], axis=0)
        pseudocont = left_weight * left_cont + right_weight * right_cont
        flux = np.nanmean(spec[lick_pts], axis=0)
        ew = delta_lick_wl * (1 - flux/pseudocont)
        if spec_err is None:
            ew_err = np.nan
        else:
            left_cont_var = np.nanmean(spec_err[left_pts, :]**2,
                                       axis=0) / left_pts.size
            right_cont_var = np.nanmean(spec_err[right_pts, :]**2,
                                        axis=0) / right_pts.size
            pseudocont_var = (left_weight * left_cont_var
                              + right_weight * right_cont_var)
            flux_var = np.nanmean(spec_err[lick_pts]**2, axis=0)
            ew_var = ((delta_lick_wl/pseudocont)**2 * flux_var
                      + (delta_lick_wl * flux/pseudocont**2)**2 * pseudocont_var)
            ew_err = np.sqrt(ew_var)
    else:
        left_cont = np.nanmean(spec[left_pts])
        right_cont = np.nanmean(spec[right_pts])
        pseudocont = left_weight * left_cont + right_weight * right_cont
        flux = np.nanmean(spec[lick_pts])
        ew = delta_lick_wl * (1 - flux/pseudocont)

        if spec_err is None:
            ew_err = np.nan
        else:
            left_cont_var = np.nanmean(spec_err[left_pts]**2) / left_pts.size
            right_cont_var = np.nanmean(spec_err[right_pts]**2
                                        ) / right_pts.size
            pseudocont_var = (left_weight * left_cont_var
                              + right_weight * right_cont_var)
            flux_var = np.nanmean(spec_err[lick_pts]**2)
            ew_var = ((delta_lick_wl/pseudocont)**2 * flux_var
                      + (delta_lick_wl * flux/pseudocont**2)**2 * pseudocont_var)
            ew_err = np.sqrt(ew_var)
    return flux, pseudocont, ew, ew_err


def compute_balmer_break(wl, spec, spec_err=None):
    """Compute the Balmer Break D4000.

    Parameters
    ----------
    - wl: wavelength vector in AA.
    - spec: flux density per wavelength unit (erg/s/[cm^2]/AA).
    - spec_err

    Returns
    -------
    - d4000: (float) Balmber Break
    - d4000_err: (float) Balmer Break Uncertainty
    """
    if spec_err is None:
        spec_err = np.full_like(spec, fill_value=np.nan)
    left_wl = np.array([3850, 3950])
    right_wl = np.array([4050, 4250])
    left_pts = np.where((wl > left_wl[0]) & (wl < left_wl[1]))[0]
    right_pts = np.where((wl > right_wl[0]) & (wl < right_wl[1]))[0]
    # left_flux = np.trapz(spec[left_pts], wl[left_pts],
    #                      axis=0) / (left_wl[1] - left_wl[0])
    # right_flux = np.trapz(spec[right_pts], wl[right_pts],
    #                       axis=0) / (right_wl[1] - right_wl[0])
    left_flux = np.nanmean(spec[left_pts], axis=0)
    right_flux = np.nanmean(spec[right_pts], axis=0)
    left_flux_err = np.sqrt(np.nanmean(spec_err[left_pts]**2, axis=0))
    right_flux_err = np.sqrt(np.nanmean(spec_err[right_pts]**2, axis=0))

    d4000 = right_flux/left_flux
    d4000_err = np.sqrt(
        d4000**2 * ((right_flux_err/right_flux)**2
                    + (left_flux_err/left_flux)**2))
    return d4000, d4000_err


def compute_all_ew(wl, spec):
    balmer_break = compute_balmer_break(wl, spec)
    ha = equivalent_width(wl, spec, central_lims=[6550., 6575.],
                          left_lims=[6470., 6530.], right_lims=[6600., 6660.])
    hb = equivalent_width(wl, spec, central_lims=[4847.88, 4876.63],
                          left_lims=[4827.86, 4847.86],
                          right_lims=[4876.63, 4891.63])
    hg = equivalent_width(wl, spec, central_lims=[4319.750, 4363.500],
                          left_lims=[4283.500, 4319.750],
                          right_lims=[4367.250, 4419.750])
    hd = equivalent_width(wl, spec, central_lims=[4083.50, 4122.25],
                          left_lims=[4041.60, 4079.75],
                          right_lims=[4128.50, 4161.00])
    return balmer_break, ha, hb, hg, hd
