#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 15:10:03 2021

@author: pablo
"""


import numpy as np

class Compute_HaEW(object):
    """ For the computation of errors:
        https://arxiv.org/pdf/astro-ph/0606341.pdf
    """
    def __init__(self, spectrum, wl, errors):
        # self.blue_band =  [6510, 6530]
        # self.red_band = [6600., 6620.]
        self.blue_band =  [6470, 6530]
        self.red_band = [6600., 6660.]
        self.central = [6550., 6575.]
        
        self.spectra = spectrum
        self.wl = wl
        self.errors = np.float64(errors)
        
        self.compute_ew()
        
    def compute_ew(self):
        # Left pseudo cont
        
        left_lamb_pos = np.where((self.wl>=self.blue_band[0]
                                  )&(self.wl<=self.blue_band[-1]))[0]                   
        
        variance_left_spectra = 1/np.nansum(1/self.errors[left_lamb_pos]**2)
        
        mean_left_spectra = np.nansum(
                self.spectra[left_lamb_pos]/self.errors[left_lamb_pos]**2
                )*variance_left_spectra
                
        
        # std_left = np.std(self.spectra[left_lamb_pos])
        std_left = np.sqrt(variance_left_spectra)
        
        self.lamb_left = (self.blue_band[0]+self.blue_band[-1])/2
                                
        # Right pseudo cont
        
        right_lamb_pos = np.where((self.wl>=self.red_band[0]
                                   )&(self.wl<=self.red_band[-1]))[0]    
        
        variance_rigth_spectra = 1/np.nansum(1/self.errors[right_lamb_pos]**2)
        
        mean_right_spectra = np.nansum(
                self.spectra[right_lamb_pos]/self.errors[right_lamb_pos]**2
                )*variance_rigth_spectra
        
        print('\n', np.nansum(1.0/self.errors[right_lamb_pos]**2),
              self.spectra[right_lamb_pos])
        # std_right = np.std(self.spectra[right_lamb_pos])
        std_right = np.sqrt(variance_rigth_spectra)
        
        self.lamb_right = (self.red_band[0]+self.red_band[-1])/2
        
        self.delta_lamb = self.lamb_right-self.lamb_left                                
        
        self.mean_left_spectra = mean_left_spectra
        self.mean_right_spectra = mean_right_spectra

        self.pseudocont = lambda lamb: mean_left_spectra*((self.lamb_right-lamb)/self.delta_lamb)\
        +mean_right_spectra*(lamb-self.lamb_left)/self.delta_lamb        
            
        self.pseudocont_err = lambda lamb: np.sqrt(
            ((self.lamb_right-lamb)/self.delta_lamb*std_right)**2 +\
            ((lamb-self.lamb_left)/self.delta_lamb*std_left)**2)
            
        
        # Central
            
        self.central_lamb_pts = np.where((self.wl>=self.central[0])&(
                self.wl<=self.central[-1]))[0]
        
        central_lamb = self.wl[self.central_lamb_pts]
        delta_central = self.central[-1]- self.central[0]
                
        central_spectra = self.spectra[self.central_lamb_pts]        
        self.mean_central_spectra = np.mean(central_spectra)
        
        central_pseudocont = self.pseudocont(central_lamb)
        
        mean_pseudocont = np.nanmean(central_pseudocont)
        mean_pseudocont_err = np.nanmean(self.pseudocont_err(central_lamb)
                                         ) / np.sqrt(len(central_pseudocont))
        
        mean_spectra = np.nanmean(central_spectra)
        mean_spectra_err = np.nanmean(self.errors[self.central_lamb_pts]
                                      ) / np.sqrt(len(central_spectra))
        
        self.mean_signal_to_noise = np.nanmean(mean_spectra/mean_spectra_err)
        
        self.central_pseudo = central_pseudocont
        self.centra_wl =central_lamb
        
        
        self.EW = np.trapz(1-central_spectra/central_pseudocont, central_lamb)
        self.EW_err = np.sqrt(
            (delta_central/mean_pseudocont*mean_spectra_err)**2+\
            ((delta_central-self.EW)/mean_pseudocont*mean_pseudocont_err)**2 
                               )        
        # self.EW_err = np.sqrt(1+mean_pseudocont/mean_spectra)*(
        #     delta_central-self.EW)/self.mean_signal_to_noise
            
    def plot_ew(self):
        from matplotlib import pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.axvline(self.blue_band[-1], color='b')
        ax.axvline(self.red_band[0], color='orange')        
        ax.axvline(self.central[0], color='lime')
        ax.axvline(self.central[-1], color='lime')
        
        ax.plot(self.wl, self.spectra, '.-', color='k')
        ###
        central_band = np.linspace(self.central[0],self.central[1], 100)
        line_flux = -self.EW*self.pseudocont(6563)        
        line_sigma = 2.5
        line_profile = np.exp(-(central_band-6563)**2/2/line_sigma**2
                              ) * 1/np.sqrt(2*np.pi)/line_sigma
        
        ax.plot(central_band, 
                self.pseudocont(central_band)+line_flux*line_profile,
                '--', color='k')
        
        line_flux_low = -(self.EW+self.EW_err)*self.pseudocont(6563)        
        line_flux_high = -(self.EW-self.EW_err)*self.pseudocont(6563)        
        
        ax.fill_between(central_band, 
                    self.pseudocont(central_band)+line_flux_low*line_profile,
                    self.pseudocont(central_band)+line_flux_high*line_profile,
                    alpha=0.5, color='blue')
        
        # ax.plot(central_band, 
        #         self.pseudocont(central_band)+line_flux*line_profile,
        #         ':', color='k')
        
        # line_flux = -(self.EW-self.EW_err)*self.pseudocont(6563)        
        # ax.plot(central_band, 
        #         self.pseudocont(central_band)+line_flux*line_profile,
        #         '--', color='k')               
        ###
        ax.fill_between(self.wl, self.spectra-self.errors,
                                 self.spectra+self.errors,
                                 color='k', alpha=0.3)
        ax.plot(self.wl, self.pseudocont(self.wl), 'r-')
        ax.fill_between(self.wl, self.pseudocont(self.wl)-self.pseudocont_err(self.wl),
                                 self.pseudocont(self.wl)+self.pseudocont_err(self.wl),
                                 color='r', alpha=0.3)
        ax.annotate(r'$EW={:.3} \pm {:.3}$'.format(self.EW, self.EW_err), xy=(.1,.9),
                    xycoords='axes fraction', ha='left')
        ax.annotate(r'$\langle S/N \rangle \sim{}$'.format(int(self.mean_signal_to_noise)),
                    xy=(.1,.85),
                    xycoords='axes fraction', ha='left')
        ax.set_xlim(self.blue_band[0], self.red_band[-1])
        ax.set_ylim(np.nanmin(
            self.spectra[(self.wl>=self.blue_band[0])&(self.wl<=self.red_band[-1])]*0.90), 
                    np.nanmax(
            self.spectra[(self.wl>=self.blue_band[0])&(self.wl<=self.red_band[-1])]*1.1)
                    )
        ax.set_yscale('log')
        
        return fig
        
        

class Compute_Haflux(object):
    def __init__(self, spectrum, wl, errors):

        self.central = [6550., 6575.]
        
        self.spectra = spectrum
        self.wl = wl
        self.errors = errors
        
        
        
    def fit_gaussian(self):                        
            
        central_lamb_pts = np.where((self.wl>=self.central[0])&(
                self.wl<=self.central[-1]))[0]
        
        central_lamb = self.wl[central_lamb_pts]
        
        central_spectra = self.spectra[central_lamb_pts]      
        central_errors = self.errors[central_lamb_pts]      
        self.mean_central_spectra = np.mean(central_spectra)
        
        from scipy.optimize import curve_fit    
        
        try:
            popt, pcov = curve_fit(self.gaussian, central_lamb, central_spectra, 
                  p0=[self.mean_central_spectra, 6563, 10],
                  # bounds=(0,[np.inf, 6570, 100]),
                  sigma=central_errors)
        
        
            self.g_popt = popt
            self.g_pcov = pcov
        
            self.ha_flux = popt[0]
        except:
            print('Not converged')
            self.ha_flux = np.nan
        
    def gaussian(self, x, A, mu, sigma):
        g = A /np.sqrt(2*np.pi)/sigma * np.exp(-(x-mu)**2/2/sigma**2)
        return g