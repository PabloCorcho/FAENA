#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 10:42:23 2023

@author: pablo
"""

import numpy as np
from astropy import units as u

from .CatalogueBase import CatalogueBase, CatProperty


__all__ = [
    'GSWLC'
    ]

DEFAULT_PATH = {"GSWLC": "/home/pablo/Research/obs_data/GWSLC/GSWLC-X2.dat"}

class GSWLC(CatalogueBase):
    
    def __init__(self, *args, **kwargs):
        super().__init__(self, name='GSWLC')
        self.path = kwargs.get("path", DEFAULT_PATH[self.name])

    
    def load_catalogue(self):
        data = np.loadtxt(self.path)
        
        self.objid = CatProperty(
            name='objid', data=data[:, 0],
            description="SDSS photometric identification number")
        
        self.glxid = CatProperty(
            name='glxid', data=data[:, 1],
            description="GALEX photometric identification number")
        
        self.plate = CatProperty(
            name='plate', data=data[:, 2],
            description="SDSS spectroscopic plate number")
        
        self.mjd = CatProperty(
            name='mjd', data=data[:, 3],
            description="SDSS spectroscopic plate date")
        
        self.fiberid = CatProperty(
            name='fiberid', data=data[:, 4],
            description="SDSS spectroscopic fiber identification number")
        
        self.ra = CatProperty(
            name='ra', data=data[:, 5], units=u.deg,
            description="Right Ascension from SDSS")
        
        self.dec = CatProperty(
            name='dec', data=data[:, 6], units=u.deg,
            description="Declination from SDSS")
        
        self.redhisft = CatProperty(
            name='redshift', data=data[:, 7],
            description="Redshift from SDSS")
        
        self.chi2 = CatProperty(
            name='chi2', data=data[:, 8],
            description="Reduced goodnes-of-fit value for the SED fitting")
        
        self.logm = CatProperty(
            name='logm', data=data[:, 9],
            description="Stellar mass")
        
        self.logm_err = CatProperty(
            name='logm_err', data=data[:, 10],
            description="Stellar mass")
        
        self.logsfr_sed = CatProperty(
            name='logsfr_sed', data=data[:, 11],
            description="Stellar mass")
        
        self.logsfr_sed_err = CatProperty(
            name='logsfr_sed_err', data=data[:, 12],
            description="Stellar mass")
        
        
        