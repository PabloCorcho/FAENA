#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 09:40:04 2022

@author: pablo
"""

import numpy as np


# =============================================================================
# Star forming galaxies
# =============================================================================
def Kauffmann03_NII_SF(logNIIHa):
    """Kauffmann 2003 diagnostic to separate classify star-forming galaxies.

    log([O III]/Hβ) < 0.61/[log([N II ]/Hα) − 0.05] + 1.30
    """
    logOIIIHb = 0.61 / (logNIIHa - 0.05) + 1.30
    logOIIIHb[logNIIHa >= 0.05] = -2
    return logOIIIHb


def Kewley_NII_SF(logNIIHa):
    """Kewley 2001 diagnostic to separate star-forming galaxies.

    log([O III]/Hβ) < 0.72/[log([S II ]/Hα) − 0.32] + 1.30,
    """
    logOIIIHb = 0.61 / (logNIIHa - 0.47) + 1.19
    logOIIIHb[logNIIHa >= 0.47] = -2
    return logOIIIHb


def Kewley_SII_SF(logSIIHa):
    """Kewley 2001 diagnostic to separate star-forming galaxies.

    log([O III]/Hβ) < 0.72/[log([S II ]/Hα) − 0.32] + 1.30,
    """
    logOIIIHb = 0.72 / (logSIIHa - 0.32) + 1.30
    logOIIIHb[logSIIHa >= 0.32] = -2
    return logOIIIHb


def Kewley_OI_SF(logOIHa):
    """Kewley 2001 diagnostic to separate star-forming galaxies.

    log([O III]/Hβ) < 0.73/[log([O I ]/Hα) + 0.59] + 1.33,
    """
    logOIIIHb = 0.72 / (logOIHa + 0.59) + 1.32
    logOIIIHb[logOIHa >= -0.59] = -2
    return logOIIIHb


def Kewley_SII_SeyferLiner(logSIIHa):
    """Kewley 2001 diagnostic to separate star-forming galaxies.

    log([O III]/Hβ) < 0.72/[log([S II ]/Hα) − 0.32] + 1.30,
    """
    logOIIIHb = 1.89 * logSIIHa + 0.76
    logOIIIHb[logOIIIHb < Kewley_SII_SF(logSIIHa)] = np.nan
    return logOIIIHb


def Kewley_OI_SeyferLiner(logOIHa):
    """Kewley 2001 diagnostic to separate star-forming galaxies.

    log([O III]/Hβ) < 0.73/[log([O I ]/Hα) + 0.59] + 1.33,
    """
    logOIIIHb = 1.18 * logOIHa + 1.30
    return logOIIIHb

