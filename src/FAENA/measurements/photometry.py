#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 00:56:46 2018

@author: pablo
"""

import numpy as np

import os
from astropy import constants as cts
from astropy import units as u
#import units
from scipy import interpolate


# USING PYPHOT

def AB_mags(wl, spec, filter, spec_err=None):
    if spec_err is None:
        spec_err = np.full_like(spec, fill_value=np.nan)
    T = filter.reinterp(wl)
    nu = 3e18/wl  # [c] = [AA/s]
    if len(spec.shape) > 1:
        f_nu = spec * (wl / nu)[:, np.newaxis]
        f_nu_err = spec_err * (wl / nu)[:, np.newaxis]
        int_f_nu = np.trapz(f_nu * T.transmit[:, np.newaxis], np.log(wl),
                            axis=0)
        int_f_nu_err = np.sqrt(np.trapz(
                f_nu_err**2 * T.transmit[:, np.newaxis], np.log(wl),
                axis=0))

        norm = np.trapz(T.transmit, np.log(wl))
        sq_pivot_wl = np.trapz(T.transmit * wl, wl) / np.trapz(T.transmit,
                                                               np.log(wl))
        int_f_lambda = int_f_nu / norm * 3e18 / sq_pivot_wl
        int_f_lambda_err = int_f_nu_err / norm * 3e18 / sq_pivot_wl
        AB = -2.5 * np.log10(int_f_nu / norm) - 48.60
        AB_err = (2.5/np.log(10)) * int_f_nu_err/int_f_nu
    else:
        f_nu = spec * (wl / nu)
        f_nu_err = spec_err * (wl / nu)
        int_f_nu = np.trapz(f_nu * T.transmit, np.log(wl),
                            axis=0)
        int_f_nu_err = np.sqrt(np.trapz(
                f_nu_err**2 * T.transmit, np.log(wl),
                axis=0))
        norm = np.trapz(T.transmit, np.log(wl))
        sq_pivot_wl = np.trapz(T.transmit * wl, wl) / np.trapz(T.transmit,
                                                               np.log(wl))
        int_f_lambda = int_f_nu / norm * 3e18 / sq_pivot_wl
        int_f_lambda_err = int_f_nu_err / norm * 3e18 / sq_pivot_wl
        AB = -2.5 * np.log10(int_f_nu / norm) - 48.60
        AB_err = (2.5/np.log(10)) * int_f_nu_err/int_f_nu
    return int_f_lambda, int_f_lambda_err, AB, AB_err

# =============================================================================
class Filter(object):
# =============================================================================

    def __init__(self, **kwargs):
    
        """This class provides a filter (SDSS, WISE, GALEX, 2MASS photometry) with the same 
        number of points as the given wavelength array.
        
        The *wavelength UNITS* are by default expressed in AA"""       
        
        self.wavelength = kwargs['wavelength']
        filter_name =kwargs['filter_name']
        
        if self.wavelength[5]>self.wavelength[6]:
            raise NameError('Wavelength array must be crescent') 
            
            
        self.filter_resp, self.wl_filter = Filter.get_filt(filter_name)
        self.filter = np.interp(self.wavelength, self.wl_filter, self.filter_resp)
        
        # self.filter = Filter.new_filter(self.wl_filter, 
        #                                 self.filter_resp, 
        #                                 self.wavelength)    
    def get_filt(filter_name):
       
        # absolute_path = os.path.dirname(os.path.abspath('photometry.py'))
        # absolute_path = os.path.join(absolute_path, '..','Filters')
                
        absolute_path = '/home/pablo/FAENA/measurements/Filters'        
        
#        absolute_path = os.path.join(absolute_path, 'Filters')
        filters_path = {'u':os.path.join(absolute_path, 'SDSS','u.dat'),
                       'g':os.path.join(absolute_path, 'SDSS','g.dat'),
                       'r':os.path.join(absolute_path, 'SDSS','r.dat'),
                       'i':os.path.join(absolute_path, 'SDSS','i.dat'),
                       'z':os.path.join(absolute_path, 'SDSS','z.dat'), 
                       'W1':os.path.join(absolute_path, 'WISE','W1.dat'),
                       'W2':os.path.join(absolute_path, 'WISE','W2.dat'),
                       'W3':os.path.join(absolute_path, 'WISE','W3.dat'), 
                       'W4':os.path.join(absolute_path, 'WISE','W4.dat'),
                   'GFUV':os.path.join(absolute_path, 'GALEX','GALEX_FUV.dat'), 
                   'GNUV':os.path.join(absolute_path, 'GALEX','GALEX_NUV.dat'),
                   '2MASS_J':os.path.join(absolute_path, '2MASS','2MASS_J.dat'), 
                   '2MASS_H':os.path.join(absolute_path, '2MASS','2MASS_H.dat'),
                   '2MASS_Ks':os.path.join(absolute_path, '2MASS','2MASS_Ks.dat')} 
       
       
        w_l, filt=np.loadtxt(filters_path[filter_name], 
                             usecols=(0,1), unpack=True)
#        w_l=np.loadtxt(filters_path[filter_name], usecols=0)
        
        return filt, w_l


    
    def new_filter( wl, fil, new_wl,*name, save=False):
        """ This function recieve the filter response and wavelength extension in order to interpolate it to a new set
         wavelengths.  First, it is checked if the filter starts or ends on the edges of the data, 
         if this occurs an array of zeros is added to limit the effective area. 
         Then, the filter response is differenciated seeking the limits of the curve to prevent wrong extrapolation. """
        
        f=interpolate.interp1d( wl, fil , fill_value= 'extrapolate' )
        
        new_filt=f(new_wl)
        
        bad_filter = False
        
        if  len(np.where(fil[0:5]>0.05)[0]):                           
            fil = np.concatenate((np.zeros(100), fil))
            bad_filter = True
        elif len(np.where(fil[-5:-1]>0.05)[0]):
            fil = np.concatenate((fil, np.zeros(100)))
            bad_filter = True
                                    
        
        band_init_pos = np.where(fil>0.001)[0][0]
        band_end_pos = np.where(fil[::]>0.001)[0][0]
        
        wl_init_pos = wl[band_init_pos]
        wl_end_pos = wl[-band_end_pos]
 
        new_band_init_pos = (np.abs(new_wl-wl_init_pos)).argmin()    
        new_band_end_pos = (np.abs(new_wl-wl_end_pos)).argmin()
        
        # To smooth the limits of the band, first the band width is computed (number of points inside) and then a 
        # called 'tails' to avoid erase any value close to te edge. If the filter starts at one corner of the distribution
        # obviously band_width_pos > new_band_init_pos, so the 'tails' could introduce negative positions. In order to avoid
        # this effect it is better to use the own initial position to delimitate the 'tail' of the band. But also, another 
        # problem is the possible lack of points and then the tail would be underestimated. For this reason, is estimated
        # the number of points out of the new distribution and the tail is enlarged proportionally.
        
        band_width_pos =  new_band_end_pos - new_band_init_pos
        
        band_tails_right_pos = int(band_width_pos*0.1)
        
        band_tails_left_pos  = band_tails_right_pos
        
        if band_width_pos>new_band_init_pos:
            missing_points = 0            
            if new_band_init_pos==0:
                delta_wl =  np.mean(np.ediff1d(new_wl[0:100]))
                missing_points = (new_wl[0] - wl_init_pos )/delta_wl
                                
            band_tails_left_pos = int(new_band_init_pos*0.1)
            band_tails_right_pos = int(band_width_pos*0.1)+int(missing_points*0.1)
            
        elif band_width_pos > len(new_wl)-new_band_end_pos:
            missing_points = 0            
            if new_band_end_pos==(len(new_wl)-1):
                delta_wl =  np.mean(np.ediff1d(new_wl[-100:-1]))
                missing_points = (wl_end_pos -new_wl[0] )/delta_wl
                                
            band_tails_left_pos = int(band_width_pos*0.1)
            band_tails_right_pos = int((len(new_wl)-new_band_end_pos)*0.1)+int(missing_points*0.1)
            
           
        new_filt[0:new_band_init_pos-band_tails_left_pos] = np.clip(new_filt[0:(new_band_init_pos-band_tails_left_pos)],0,0)
        new_filt[(new_band_end_pos+band_tails_right_pos):-1] = np.clip(new_filt[(new_band_end_pos+band_tails_right_pos):-1],0,0)
        
        new_filt[-1]=0     # Sometimes it is the only point which gives problems
        
        # Furthermore, the worst case is when the original filter also starst at one corner of the distrib, so probably wrong
        # values appear close to the real curve. More drastically, all the sorrounding points are set to zero.
        
        if bad_filter == True:
            new_filt[new_band_end_pos:-1]=0
            new_filt[0:new_band_init_pos]=0
        
        if save==True:
            new_filt_zip=zip(new_wl,new_filt)
            
            with open('Filters/'+str(name)+'.txt', 'w' ) as f:
                for i,j in new_filt_zip:
                    f.write('{:.4} {:.4}\n'.format(i,j))
            print('Filter'+str(name)+'saved succesfully ')        
        
        return new_filt    
    
    
    def square_filt(wl,l_i,l_e):
         
         s_filt=np.ones(len(wl))
         
         for i,j in enumerate(wl):
             if j<l_i:
                 
                 s_filt[i]=0
            
             elif j>l_i and j<l_e:
                 
                 s_filt[i]=1
            
             elif j>l_e:
                 
                 s_filt[i]=0
         return s_filt
         
         
         
# =============================================================================
class magnitude(Filter):         
# =============================================================================
    """This module computes the photmetric magnitude on a given band 
    for a given flux with UNITS expressed in erg/s.s"""
             
    def __init__(self, absolute=False, **kwargs):
        
        Filter.__init__(self, **kwargs)
        self.nu = cts.c.value/( self.wavelength*1e-10 )
        self.flux = kwargs['flux']     
        
        if 'flux_err' in kwargs.keys():
            self.flux_err = kwargs['flux_err']
            self.compute_err = True
            
        else:
            self.compute_err = False
            
            
        self.absolute = absolute
        
    def AB(self):   #photometric system  used with SDSS filters
        """ This function computes the magnitude in the AB system of a given spectrum. The spectrum units must be in erg/s for absolute
         magnitude computation or in erg/s/cm2 for apparent magnitude. """
 
        if self.absolute==True: 
            self.flux = self.flux/(4*np.pi* (10*u.pc.to('cm').value)**2)   # flux at 10 pc.
        
        F_nu = self.flux/self.nu
        
        
        integral_flux = np.trapz(F_nu*self.filter, np.log(self.nu))                
        integral_R = np.trapz(self.filter, np.log(self.nu))
        
        # integral_flux = np.trapz(F_nu*self.filter, self.nu)                
        # integral_R = np.trapz(self.filter, self.nu)
                
        mag =-2.5*np.log10(integral_flux/integral_R) - 48.60
        
        if self.compute_err:
            
            mean_flux = np.nansum(F_nu*self.filter)/np.nansum(self.filter)            
            mean_flux_err = np.nanmedian(self.flux_err/self.nu)
            
            mean_flux_err = mean_flux_err/np.sqrt(
                len(self.filter[self.filter>0.001]))                        
            rel_flux_err = np.abs(mean_flux_err/mean_flux)
                    
            mag_err = 2.5/np.log(10) * rel_flux_err            
            # print(mag_err)
            return mag, mag_err
        
        else:
            return mag
    
    def band_flux(self):   #photometric system  used with SDSS filters
        """ This function computes the magnitude in the AB system of a given spectrum. 
        The spectrum units must be in erg/s for absolute
         magnitude computation or in erg/s/cm2 for apparent magnitude. """
 
        ## Photon counts
        # if self.absolute==True: 
        #     self.flux = self.flux/(4*np.pi* (10*units.pc/units.cm)**2)   # flux at 10 pc.
        
        F_nu = self.flux/self.nu
                        
        integral_flux = np.trapz(F_nu*self.filter, np.log(self.nu))
        integral_R = np.trapz(self.filter, np.log(self.nu))
        
        
        return integral_flux/integral_R
    
         
    def Vega(self): #photometric system  used usually with Jonhson/Bessel/Coussin filters
        
        diff_wl = np.ediff1d(np.insert((self.wavelength),0,0))
            
        wl_vega=np.loadtxt('Filters/alpha_lyr.dat',usecols=0)
        diff_wl_vega = np.ediff1d(np.insert(wl_vega,0,0))
        flux_vega=np.loadtxt('Filters/alpha_lyr.dat',usecols=2)
        
        if self.absolute == True:
            flux_vega=(flux_vega * 4*np.pi*(25.30*u.lyr.to('cm').value**2 ) / (4*np.pi* (10*u.pc.to('cm').value)**2))
            self.flux = self.flux/(4*np.pi* (10*u.pc.to('cm').value)**2)   #flux at 10 pc
            
            
        vega_filter=Filter.new_filter(self.wl_filter, self.filter_resp , wl_vega)
        integral_flux_vega=np.nansum(flux_vega * vega_filter * diff_wl_vega)
        integral_flux = np.nansum(self.flux * self.filter * diff_wl ) 
        
        m=-2.5* np.log10(integral_flux/integral_flux_vega) +0.58
         
        return m
    
        
if __name__ == '__main__':
    from astropy import units as u
    from synphot import units as s_units
    from synphot import SourceSpectrum
    from synphot.models import BlackBodyNorm1D
    import pyphot
    from matplotlib import pyplot as plt
    
    temps = np.logspace(3.2, 4, 50)
    my_color = []
    py_color = []
    for temp_i in temps:
        wavelength = np.linspace(3000, 7000, 5000) #AA
        
        bb = SourceSpectrum(BlackBodyNorm1D, temperature=temp_i)
        bb_spectra = bb(wavelength, flux_unit=s_units.FLAM).value
        
        library=pyphot.get_library()
        py_g = library['SDSS_g']
        py_r = library['SDSS_r']
        py_g_flux = py_g.get_flux(wavelength, bb_spectra).value
        py_g_mag = -2.5 * np.log10(py_g_flux) - py_g.AB_zero_mag
        py_r_flux = py_r.get_flux(wavelength, bb_spectra).value
        py_r_mag = -2.5 * np.log10(py_r_flux) - py_r.AB_zero_mag
        
        # my_filter = Filter(wavelength=wavelength, filter_name='r')
        
        
        # plt.figure()        
        # plt.plot(py_filter.wavelength.value, py_filter.transmit, lw=3)
        # plt.plot(my_filter.wl_filter, my_filter.filter_resp, 'r')
        # plt.plot(wavelength, my_filter.filter, 'k--')
        # # plt.xlim(1.2e4, 1.3e4)
        
        # plt.figure()
        # plt.plot(wavelength, bb_spectra, 'k')
        # plt.plot(wavelength, bb_spectra*my_filter.filter, 'r')
        # plt.plot(wavelength, bb_spectra*np.interp(wavelength, py_filter.wavelength, 
        #                                           py_filter.transmit), 'b--')
        
        my_g_mag = magnitude(wavelength=wavelength, flux=bb_spectra*wavelength, filter_name='g')
        my_r_mag = magnitude(wavelength=wavelength, flux=bb_spectra*wavelength, filter_name='r')
        
        m_g_mag = my_g_mag.AB()
        m_r_mag = my_r_mag.AB()
        
        print('PYPHOT g-mag:', py_g_mag, 'This g-mag:', m_g_mag, '\nDifference', py_g_mag-m_g_mag)
        print('PYPHOT r-mag:', py_r_mag, 'This r-mag:', m_r_mag, '\nDifference', py_r_mag-m_r_mag)
        print(py_g_mag-py_r_mag, m_g_mag-m_r_mag)
        
        my_color.append(m_g_mag-m_r_mag)
        py_color.append(py_g_mag-py_r_mag)
    
    plt.figure()
    plt.plot(temps, my_color, 'k')
    plt.plot(temps, py_color, 'r--')
    plt.grid(b=True)
    
