#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 13:23:51 2020

@author: pablo
"""


from astropy.io import fits
from astropy import units as u 
import numpy as np 
from matplotlib import pyplot as plt
import os 

class Pipe3Dssp(object):
    """
    This class provides the simple stellar population models that Pipe3D 
        used with IFU data.      
        
   A total of 156 SSP's with 39 ages and 4 metallicities:
           - 2 ages x 4 metallicities in Granada+ Geneva tracks 
           - 13 ages x 4 Z in Granada+Padova track
           - 24 ages x 4 Zs in Vazdekiz/Falcon-Barroso/MILES SSP models which 
           /Users/cid match the 40 ages we are working with in the CALIFA fits.
    
    Mstars grabbed from file ssp_mass_UN_mu1.3_v9.1.ElCid, and for young ages 
    Mstars = (t/3,631)^-0.05284.
    
    Stellar Evolution = Padova 2000
    
    IMF = UN = salpeter
    
    Each SSP when born had 1 Msun.
    """
    
    ssps_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             'gsd01_156.fits')
    ssp_properties_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'fits_like_properties.dat')
        
    def __init__(self):
     
        ssp_fits = fits.open(self.ssps_path)
        
        self.header = ssp_fits[0].header
        self.wavenorm = self.header['WAVENORM']
        individual_spec_names = self.header['NAME*']
        self.ssp_age = np.zeros(len(individual_spec_names))
        self.ssp_met = np.zeros_like(self.ssp_age)
        for ii, file_i in enumerate(individual_spec_names):
            age, met = self.header[file_i].split('_')[-2:]
            self.ssp_age[ii] = float(age.split('Gyr')[0])
            self.ssp_met[ii] = float(met.split('.dat')[0].replace('z', '0.'))
            
        self.norm = np.ones(156) # Lsun/Msun (a.k.a L/M ratio)
        for i in range(156):
            self.norm[i] = ssp_fits[0].header['NORM'+str(i)]
            
        self.ssp_SED = ssp_fits[0].data # Normalized to 5500 AA        
        #*self.norm[:, np.newaxis] #Lsun/AA/Msun (L_lambda)
        # self.ssp_SED *= 3.82e33 # erg/s/AA/Msun
        wl0 = ssp_fits[0].header['CRVAL1']
        deltawl = ssp_fits[0].header['CDELT1']
        self.ssp_wl = np.arange(wl0, wl0+deltawl*self.ssp_SED.shape[1], deltawl
                                ) * u.angstrom
        
        ssp_fits.close()
        
    def ssp_ages(self, mode='individual'):
        """
        'individual' mode returns one value per ssp (156)
        'age' mode returns the age bins of all ssp's (39)

        WARNING: 'individual' ages are not ordered. 
        They are stored in the same way as the Pipe3D outputs at SFH cubes. 
        """        
        if mode == 'individual':
            return self.ssp_age * u.Gyr
        elif mode == 'age':
            return np.unique(self.ssp_age) * u.Gyr
    
    def ssp_metallicity(self,  mode='individual'):        
        """Idem ssp_ages method"""        
        if mode == 'individual':
            return self.ssp_met
        elif mode == 'age':
            return np.unique(self.ssp_met)
    
    def ssp_alive_stellar_mass(self, mode='individual'):
        """
        Returns the fraction of stellar mass within each SSP after mass loss
        for each age and metallicity.
        """
        self.ssp_alive_mass = np.loadtxt(self.ssp_properties_path,
                                                         usecols=(2), unpack=True)
        self.ssp_alive_mass = self.ssp_alive_mass.reshape(4,39)        
        if mode=='individual':
            self.ssp_alive_mass = self.ssp_alive_mass.flatten()
        elif mode=='age':
            self.ssp_alive_mass = np.mean(self.ssp_alive_mass, axis=0)
        elif mode=='metallicity':
            self.ssp_alive_mass = np.mean(self.ssp_alive_mass, axis=1)                        
        return self.ssp_alive_mass
    
    def compute_ssp_initial_mass_lum_ratio(self, mode='individual', wl=[4470, 6470]):        
        wl_pt = np.where((self.ssp_wl>wl[0])&(self.ssp_wl<wl[-1]))[0]
        lum = np.mean(self.ssp_SED[:, wl_pt], axis=1)
        initial_lum = lum.reshape(4, 39)[:, 0]                
        
        self.init_mass_to_lum = np.ones((4, 39))/initial_lum[:, np.newaxis]
                
        if mode=='individual':
            self.init_mass_to_lum = self.init_mass_to_lum.flatten()
        elif mode=='age':
            self.init_mass_to_lum = np.mean(self.init_mass_to_lum, axis=0)
        elif mode=='metallicity':
            self.init_mass_to_lum = np.mean(self.init_mass_to_lum, axis=1)            
            
        return self.init_mass_to_lum
    
    def ssp_present_mass_lum_ratio(self, mode='individual'):
        """
        The normalization provided by the models was computed such that
        models_SED * norm = L_lambda for each SSP
        Therefore, the M/L relation is expressed as M/L_Lambda
        """
        if mode == 'individual':
            normalization = self.norm * u.L_sun/u.M_sun
            self.mass_to_lum = 1/normalization
            self.mass_to_lum = self.mass_to_lum.to(u.Msun/(u.erg/u.s))
            # alive_mass = self.ssp_alive_stellar_mass()
            # self.mass_to_lum /= alive_mass
            return self.mass_to_lum  
        else: 
            raise NameError('MODE NOT DEVELOPED YET')
    
    def renormalize_models(self, normalization_wavelength):
        print('· [SSP MODELS] COMPUTING --> Renormalizing SSP models at {}'.format(normalization_wavelength))        
        pos = self.ssp_wl.searchsorted(normalization_wavelength)
        if ( pos == self.ssp_wl.size)|( pos == 0):
            print('· [SSP MODELS] WARNING --> Normalization wavelength out of the SSP wavelength array')
            return
        
        w_hi = (normalization_wavelength-self.ssp_wl[pos-1])/(self.ssp_wl[pos]-self.ssp_wl[pos-1])
        w_low = 1-w_hi        
        new_norm = self.norm * (w_hi*self.ssp_SED[:, pos] + w_low*self.ssp_SED[:, pos])
        
        self.ssp_SED = self.ssp_SED * self.norm[:, np.newaxis] / new_norm[:, np.newaxis]
        self.norm = new_norm
        self.wavenorm = normalization_wavelength
                
# Example----------------------------------------------------------------------        
if __name__ == '__main__':
    print(""" /
          Pipe3D-SSPs
          --> Self-consistency tests
          """)
    from matplotlib import pyplot as plt
    pipssp = Pipe3Dssp()
    
    plt.figure()
    plt.scatter(pipssp.ssp_age, pipssp.norm, c=pipssp.ssp_met)
    plt.yscale('log')
    plt.ylabel(r'5500 $\AA$  luminosity normalization ($L_\odot/5500\AA/M_\odot)$')
    plt.xlabel(r'SSP Age (Gyr)')
    plt.colorbar(label='Z')
    
    plt.figure()
    plt.scatter(pipssp.ssp_age, pipssp.ssp_present_mass_lum_ratio().to(u.Msun/u.Lsun), c=pipssp.ssp_met)
    plt.yscale('log')
    plt.ylabel(r'$\Upsilon~(M_\odot/L_\odot)$')
    plt.xlabel(r'SSP Age (Gyr)')
    plt.colorbar(label='Z')
    
    
    plt.figure()
    for i in np.arange(0, 156, 20):
        plt.plot(pipssp.ssp_wl, pipssp.ssp_SED[i, :],
             label='age={:.3f}, Z={:.4f}'.format(pipssp.ssp_age[i], pipssp.ssp_met[i]))
    plt.legend(bbox_to_anchor=(1, 1, .5, .0))
    
    pipssp.renormalize_models(4000*u.angstrom)
    plt.figure()    
    for i in np.arange(0, 156, 20):
        plt.plot(pipssp.ssp_wl, pipssp.ssp_SED[i, :],
             label='age={:.3f}, Z={:.4f}'.format(pipssp.ssp_age[i], pipssp.ssp_met[i]))
    plt.legend(bbox_to_anchor=(1, 1, .5, .0))
    