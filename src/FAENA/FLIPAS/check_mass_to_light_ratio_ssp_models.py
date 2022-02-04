#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 12:15:07 2021

@author: pablo
"""

from pst import SSP
from astropy import units as u
import numpy as np
from matplotlib import pyplot as plt
granada = SSP.BaseGM()
mass_to_light_ratio = 1/granada.norm.to((u.erg/u.s)/(u.angstrom * u.Msun))
mean_granada = np.mean(mass_to_light_ratio, axis=0)
std_granada = np.std(mass_to_light_ratio, axis=0)

popstar = SSP.PopStar(IMF='kro_0.15_100')
mass_to_light_popstar = np.zeros(shape=(popstar.metallicities.size, 
                                        popstar.ages.size)) * mass_to_light_ratio.unit
for i in range(popstar.metallicities.size):
    for j in range(popstar.ages.size):
        l5500 = np.interp(5500*u.angstrom,
                          popstar.L_lambda[i, j].spectral_axis,
                          popstar.L_lambda[i, j].flux)
        mass_to_light_popstar[i, j] = 1/l5500.to((u.erg/u.s)/(u.angstrom * u.Msun))
mean_popstar = np.mean(mass_to_light_popstar, axis=0)
std_popstar = np.std(mass_to_light_popstar, axis=0)
# %%
plt.figure()
for i, met_i in enumerate(granada.metallicities):
    plt.plot(granada.ages, mass_to_light_ratio[i, :], label='{:.4f}'.format(met_i))
for i, met_i in enumerate(popstar.metallicities):
    plt.plot(popstar.ages, mass_to_light_popstar[i, :], label='{:.4f}'.format(met_i))
plt.xscale('log')
plt.yscale('log')
plt.legend()
# %%
plt.figure()

plt.plot(granada.ages, mean_granada, c='b', label='BaseGM')
plt.fill_between(granada.ages, mean_granada-std_granada,
                 mean_granada+std_granada, color='blue', alpha=.3)
plt.plot(popstar.ages, mean_popstar, c='r', label='Popstar')
plt.fill_between(popstar.ages, mean_popstar-std_popstar,
                 mean_popstar+std_popstar, color='red', alpha=.3)
plt.xscale('log')
plt.yscale('log')
plt.legend()