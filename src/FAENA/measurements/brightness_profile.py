#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  2 09:20:21 2021

@author: pablo
"""

from matplotlib import pyplot as plt
import numpy as np

from photutils.centroids import centroid_com
from scipy.optimize import curve_fit
from astropy.modeling.functional_models import Sersic2D
from astropy.modeling.functional_models import Gaussian2D
from astropy.modeling import fitting
import warnings

from photutils.centroids import centroid_2dg
from photutils.aperture import CircularAperture
from photutils.aperture import aperture_photometry


def compute_eff_radii(image, mass_radius=2., plot=False):
    max_pix_rad = np.min(image.shape)//2
    radius = np.arange(3, max_pix_rad, 3)
    fake_image = np.zeros_like(image)
    fake_image[max_pix_rad//2: (3*max_pix_rad)//2,
               max_pix_rad//2: (3*max_pix_rad)//2] = image[
                   max_pix_rad//2: (3*max_pix_rad)//2,
                   max_pix_rad//2: (3*max_pix_rad)//2]
    com_x, com_y = centroid_2dg(fake_image)
    aperture_sum = []
    for rad_i in radius:
        aperture = CircularAperture((com_x, com_y), r=rad_i)
        phot_table = aperture_photometry(image, aperture)
        aperture_sum.append(phot_table['aperture_sum'].value)
    aperture_sum = np.array(aperture_sum).squeeze()
    norm_aperture_sum = aperture_sum / aperture_sum[-1]
    half_mass_rad = np.interp(0.5, norm_aperture_sum, radius)
    half_mass_aperture = CircularAperture((com_x, com_y), r=half_mass_rad)
    nhalf_mass_aperture = CircularAperture((com_x, com_y),
                                           r=mass_radius * half_mass_rad)
    half_mass_table = aperture_photometry(image, half_mass_aperture)
    nhalf_mass_table = aperture_photometry(image, nhalf_mass_aperture)
    if plot:
        fig = plt.figure()
        plt.imshow(np.log10(image))
        plt.colorbar(label='log(image)')
        plt.plot(com_x, com_y, '+', markersize=8, color='c')
        half_mass_aperture.plot(color='r', lw=2,
                                label=r'1 Re')
        nhalf_mass_aperture.plot(color='orange', lw=2,
                                 label=r'{:2.1f} Re'.format(mass_radius))
        plt.annotate('Tot mass={:.2}\nHalf mass={:.2}\nTwoHalf mass={:.2}'.
                     format(float(aperture_sum[-1]),
                            float(half_mass_table['aperture_sum'].value),
                            float(nhalf_mass_table['aperture_sum'].value)),
                     xy=(.1, 1), xycoords='axes fraction', va='bottom')
        plt.legend()
        plt.close()
        return half_mass_aperture, nhalf_mass_aperture, fig
    else:
        return half_mass_aperture, nhalf_mass_aperture
# %%
def half_mass(image):
    """
    This method takes an image as imput and returns a mask of the pixels contained 
    that account for half of the total light.         
    """
    XX, YY = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1])) 
    max_elem = np.max(image.shape)
    mask = np.zeros_like(image, dtype=bool)        
    com = np.array(centroid_com(image), dtype=int)    
    in_pos = np.array([[com[0], com[1]]])    
    mask[com[0], com[1]] = True
    
    mass = np.sum(image[mask])    
    all_light = np.sum(image)
    
    while mass < all_light/2:
        candidates_pos = []
        candidates_mass = []
        for ii, pos_i in enumerate(in_pos):
            # print(ii, pos_i)
            new_pos = pos_i+[0,1]            
            new_pos = new_pos.clip(min=0, max=max_elem)
            if (np.sum((in_pos-new_pos[np.newaxis, :])**2, axis=1).all()!=0)&(
                    new_pos[0]<max_elem)&(new_pos[1]<max_elem):             
                candidates_pos.append(new_pos)
                candidates_mass.append(image[new_pos[0], new_pos[1]])
            
            new_pos = pos_i+[1,0]                             
            new_pos = new_pos.clip(min=0, max=max_elem)                      
            if (np.sum((in_pos-new_pos[np.newaxis, :])**2, axis=1).all()!=0)&(
                    new_pos[0]<max_elem)&(new_pos[1]<max_elem):    
                candidates_pos.append(new_pos)
                candidates_mass.append(image[new_pos[0], new_pos[1]])
                
            new_pos = pos_i+[0,-1]    
            new_pos = new_pos.clip(min=0, max=max_elem)
            if (np.sum((in_pos-new_pos[np.newaxis, :])**2, axis=1).all()!=0)&(
                    new_pos[0]<max_elem)&(new_pos[1]<max_elem):    
                candidates_pos.append(new_pos)
                candidates_mass.append(image[new_pos[0], new_pos[1]])
            
            new_pos = pos_i+[-1,0]                                       
            new_pos = new_pos.clip(min=0, max=max_elem)
            if (np.sum((in_pos-new_pos[np.newaxis, :])**2, axis=1).all()!=0)&(
                    new_pos[0]<max_elem)&(new_pos[1]<max_elem):                        
                candidates_pos.append(new_pos)
                candidates_mass.append(image[new_pos[0], new_pos[1]])
                
        
        try:            
            best = np.argmax(candidates_mass)                                                           
        except:
            break
        in_pos = np.vstack((in_pos, candidates_pos[best]))
        mass += candidates_mass[best]
        mask[candidates_pos[best][0], candidates_pos[best][1]] = True
        # print(candidates_mass)
        print('% Completion... ', mass/(all_light/2))
        # if ii==3:
        #     break    
    return mask
        
        
class SersicFit(object):
    """
    This class is intended to compute accurate Sersic 2D models to astronomical
    images.        
    """
    def __init__(self, image):
        self.image = image
        self.image[np.isnan(self.image)] = 0
        # self.image /= np.nanmax(self.image)
        
        self.XX, self.YY = np.meshgrid(np.arange(self.image.shape[0]), 
                          np.arange(self.image.shape[1]))
        
        x_o_g, y_o_g = centroid_com(self.image)
        self.com = np.array([x_o_g, y_o_g])
        
        amplitude_g = np.median(self.image[int(x_o_g)-10:int(x_o_g)+10,
                                      int(y_o_g)-10:int(y_o_g)+10])
        reff_g = (self.image.shape[0]+self.image.shape[1])/4
        n_g = 2.
        ellip_g = 0.5
        theta_g = 0.
        self.initial_guess = [amplitude_g, reff_g, n_g, x_o_g, y_o_g, ellip_g, theta_g]                                
        print('Initial guess:\n - Amplitude={:.2f}\n - Reff={:.2f}\n - n_sersic={:.1f}\n - COM=({:.1f},{:.1f})\n - ellip={:.2f}\n - theta={:.2}'.\
              format(self.initial_guess[0], self.initial_guess[1], self.initial_guess[2],
                     self.initial_guess[3], self.initial_guess[4], self.initial_guess[5],
                     self.initial_guess[6])
              )
              
        
    def sersic(self, xy, amplitude, reff, n, x_0, y_0, ellip, theta):
        
        mod = Sersic2D(amplitude, reff, n, x_0, y_0, ellip, theta)
            
        return mod(xy[0], xy[1]).ravel()
        
    def fit(self):
        
        p_init = Sersic2D(
            amplitude=self.initial_guess[0], 
            r_eff=self.initial_guess[1], 
            n=self.initial_guess[2], 
            x_0=self.initial_guess[3], 
            y_0=self.initial_guess[4], 
            ellip=self.initial_guess[5], 
            theta=self.initial_guess[6])
        fit_p = fitting.LevMarLSQFitter()
        
        with warnings.catch_warnings():
            # Ignore model linearity warning from the fitter
            warnings.simplefilter('ignore')
            p = fit_p(p_init, self.XX, self.YY, self.image)
            
        self.popt = p.parameters
        self.amplitude = self.popt[0]
        self.reff = self.popt[1]
        self.n = self.popt[2]
        self.x_0, self.y_0 = self.popt[3], self.popt[4]
        self.ellip, self.theta = self.popt[5], self.popt[6]        
        self.model = self.sersic((self.XX.ravel(), self.YY.ravel()), *self.popt)
        self.model = self.model.reshape(self.image.shape)
        print('Best fit:\n - Amplitude={:.2f}\n - Reff={:.2f}\n - n_sersic={:.1f}\n - COM=({:.1f},{:.1f})\n - ellip={:.2f}\n - theta={:.2}'.\
              format(self.amplitude, self.reff, self.n, self.x_0, self.y_0, self.ellip,
                     self.theta)
              )
    def get_fit_mask(self, n_eff_rad = 1):
        mask = np.zeros_like(self.model, dtype=bool)
        mask[self.model >= n_eff_rad * self.amplitude] = True        
        return mask
    def plot_fit(self, save=False):
        # ig_model = self.gaussian((self.XX.ravel(), self.YY.ravel()), *self.ig_popt)
        # ig_model = ig_model.reshape(self.image.shape)
        
        model = self.sersic((self.XX.ravel(), self.YY.ravel()), *self.popt)
        model = model.reshape(self.image.shape)
        lvls = np.array([self.amplitude*0.5, self.amplitude, 
                           self.amplitude*2, self.amplitude*4, self.amplitude*6])
        
        fig = plt.figure(figsize=(5, 3.5))
        ax = fig.add_subplot(121)
        mappable = ax.imshow(np.log10(self.image), origin='lower', aspect='auto',
                              cmap='terrain', vmin=np.nanmax(np.log10(self.image))-3) 
        ax.contour(np.log10(self.image), colors='k', levels=np.log10(lvls),
                   linestyles=[':', '-', '--', ':'])
        plt.colorbar(mappable, ax=ax, orientation='horizontal', label=r'$\log(I)$')        
        ax.contour(np.log10(model), colors=['r', 'orange', 'r', 'r'], 
                   levels=np.log10(lvls),
                   linestyles=[':', '-', '--', ':'])    
        # ax.contour(np.log10(ig_model), colors='b', 
        #            levels=np.log10(lvls),
        #            linestyles=[':', '-', '--', ':'])
        ax.plot(self.x_0, self.y_0, 'b+')
        ax.annotate(r'$R_{eff}=$'+'{:2.1f}'.format(self.reff) +'\nn$_{Ser}$='+'{:1.1f}'.format(self.n),
                    xy=(0.05, 0.05), xycoords='axes fraction', va='bottom',
                    ha='left', bbox=dict(facecolor='white',
                                         edgecolor='green', boxstyle='round'))        
        ax.annotate(r'$e=$'+'{:4.2}\n'.format(self.ellip) +r'$\theta$='+'{:2.1f}'.format(np.rad2deg(self.theta)),
                    xy=(0.05, 0.95), xycoords='axes fraction', va='top',
                    ha='left', bbox=dict(facecolor='white',
                                         edgecolor='green', boxstyle='round'))        
        # ax.annotate(r'$n=$'+'{:4.2}'.format(self.n),
        #             xy=(0.05, 0.1), xycoords='axes fraction', va='bottom',
        #             ha='left', weight='bold')
        ax = fig.add_subplot(122)
        mappable = ax.imshow(np.log10(self.image/model), origin='lower', 
                             cmap='nipy_spectral', aspect='auto',
                             vmin=-.3, vmax=.3)
        plt.colorbar(mappable, ax=ax, orientation='horizontal', 
                     label=r'$\log_{10}(image/model)$')
        plt.show()        
        if save:
            fig.savefig(save, bbox_inches='tight')
        plt.close()
        
        
if __name__ == '__main__':
    from astroquery.sdss import SDSS
    from astropy.coordinates import SkyCoord
    from astropy.wcs import WCS
    pos = SkyCoord(236.677624895, 43.3724791101, unit='deg', frame='icrs')
    xid = SDSS.query_region(pos, spectro=True)
    
    im = SDSS.get_images(matches=xid, band='r')
    
    wcs = WCS(im[0][0].header)
    
    pixel_pos = wcs.world_to_array_index(pos)
    data = np.copy(im[0][0].data)
    cutout = data[pixel_pos[0]-50:pixel_pos[0]+50,
                    pixel_pos[1]-50:pixel_pos[1]+50]
    com = centroid_com(cutout)
    
    # a = half_mass(cutout)
    # %%
    SFit = SersicFit(cutout)
    SFit.fit()
    SFit.plot_fit()
        
    plt.figure()
    plt.imshow(np.log10(cutout), cmap='Greys', origin='lower')
    plt.contour(np.log10(SFit.model), colors='r', levels=[np.log10(SFit.amplitude)])
    plt.plot(com[0], com[1], 'r+')
    # plt.contour(a, colors='b', levels=1)
    
    
