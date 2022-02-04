#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 13:13:37 2021

@author: pablo
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 16:02:57 2021

@author: pablo
"""

import numpy as np

from photutils.aperture import CircularAperture
from photutils.aperture import aperture_photometry
from photutils.centroids import centroid_2dg
from matplotlib import pyplot as plt


def equivalent_width(wl, spec, central_lims, left_lims, right_lims):
    left_wl = np.array(left_lims)
    mean_left_wl = left_wl.mean()
    right_wl = np.array(right_lims)
    mean_right_wl = right_wl.mean()
    lick_wl = np.array(central_lims)
    mean_lick_wl = lick_wl.mean()
    left_pts = np.where((wl >= left_wl[0]) & (wl <= left_wl[1]))[0]
    right_pts = np.where((wl >= right_wl[0]) & (wl <= right_wl[1]))[0]
    lick_pts = np.where((wl > lick_wl[0]) & (wl < lick_wl[1]))[0]
    right_weight = (mean_lick_wl - mean_left_wl
                    ) / (mean_right_wl - mean_left_wl)
    left_weight = 1 - right_weight
    if len(spec.shape) > 1:
        left_cont = np.nanmean(spec[left_pts, :], axis=0)
        right_cont = np.nanmean(spec[right_pts, :], axis=0)
        pseudocont = left_weight * left_cont + right_weight * right_cont
        flux = np.trapz(spec[lick_pts], wl[lick_pts], axis=0)
        ew = (lick_wl[1] - lick_wl[0]) - flux/pseudocont
    else:
        left_cont = np.nanmean(spec[left_pts])
        right_cont = np.nanmean(spec[right_pts])
        pseudocont = left_weight * left_cont + right_weight * right_cont
        flux = np.trapz(spec[lick_pts], wl[lick_pts])
        ew = (wl[lick_pts][-1] - wl[lick_pts][0]) - flux/pseudocont
        # ew = (lick_wl[1] - lick_wl[0]) - np.trapz(
        #     spec[lick_pts]/pseudocont(wl[lick_pts]), wl[lick_pts])
    return flux, pseudocont, ew


def compute_balmer_break(wl, spec):
    left_wl = np.array([3850, 3950])
    right_wl = np.array([4050, 4250])
    left_pts = np.where((wl > left_wl[0]) & (wl < left_wl[1]))[0]
    right_pts = np.where((wl > right_wl[0]) & (wl < right_wl[1]))[0]
    left_flux = np.trapz(spec[left_pts], wl[left_pts],
                         axis=0) / (left_wl[1] - left_wl[0])
    right_flux = np.trapz(spec[right_pts], wl[right_pts],
                          axis=0) / (right_wl[1] - right_wl[0])
    d4000 = right_flux/left_flux
    return d4000


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
