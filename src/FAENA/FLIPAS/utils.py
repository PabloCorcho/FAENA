#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 16:26:08 2021

@author: pablo
"""

from astropy.cosmology import FlatLambdaCDM

import extinction

# =============================================================================
# Cosmological model
# =============================================================================

Cosmology = FlatLambdaCDM(H0=70, Om0=0.3)

extinction_law = extinction.ccm89
