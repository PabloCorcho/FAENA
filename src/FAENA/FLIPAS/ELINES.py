#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 15:30:20 2021

@author: pablo
"""

from astropy import units as u

flux_norm = u.def_unit('10^{-16} erg/s/cm^2', 1e-16 * u.erg/u.s/u.cm**2)


class ELINES(object):
    """
    This class recieves as input the ELINES fits extension (output from Pipe3D)
    and builds a table of emission line properties.
    """

    def __init__(self, ELINES_extension, verbose=True):
        self.verbose = verbose
        self.ELINES_data = ELINES_extension
        self.read_elines_data()

    def read_elines_data(self):
        if self.verbose:
            print('· [ELINES MODULE] READING --> Header extensions')
        hdr = self.ELINES_data.header
        hdr_keys = list(hdr.keys())
        self.ELINES_variables = {}
        for ii, key in enumerate(hdr_keys):
            npos = key.find('NAME')
            if npos == 0:
                number = int(key[npos+4:])
                line_name = hdr[key].split(' ')[-1]
                if hdr[key].find('flux') == 0:
                    self.ELINES_variables[line_name] = {}
                    self.ELINES_variables[line_name]['flux'] = number
                elif hdr[key].find('vel') == 0:
                    self.ELINES_variables[line_name]['vel'] = number
                elif hdr[key].find('disp') == 0:
                    self.ELINES_variables[line_name]['disp'] = number
                elif hdr[key].find('EW') == 0:
                    self.ELINES_variables[line_name]['EW'] = number
                elif hdr[key].find('e_flux') == 0:
                    self.ELINES_variables[line_name]['e_flux'] = number
                elif hdr[key].find('e_vel') == 0:
                    self.ELINES_variables[line_name]['e_vel'] = number
                elif hdr[key].find('e_disp') == 0:
                    self.ELINES_variables[line_name]['e_disp'] = number
                elif hdr[key].find('e_EW') == 0:
                    self.ELINES_variables[line_name]['e_EW'] = number

    def get_line_flux(self, line_name):
        if self.verbose:
            print('· [ELINES MODULE] READING --> Line {}'.format(line_name))
        line_flux = self.ELINES_data.data[
            self.ELINES_variables[line_name]['flux'], :, :] * flux_norm
        line_flux_error = self.ELINES_data.data[
            self.ELINES_variables[line_name]['e_flux'], :, :] * flux_norm
        return line_flux, line_flux_error


if __name__ == '__main__':
    pass

# Mr. Krtxo
