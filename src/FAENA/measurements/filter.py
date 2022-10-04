"""Created on Mon Aug 15 08:59:35 2022.

@author: pablo
"""
import os
import numpy as np


def list_filter_dir():
    """Load all default filters."""
    filter_dir = os.listdir(os.path.join(os.path.dirname(__file__),
                                         'Filters/'))
    filters = {}
    for dir_ in filter_dir:
        path = os.path.join(os.path.dirname(__file__),
                            'Filters', dir_)
        if os.path.isdir(path):
            filters[dir_] = os.listdir(path)
    return filters


class Filter(object):
    """Generic class for photometric filters.

    The *wavelength UNITS* are by default expressed in AA

    Attributes
    ----------
    - database_dir
    - name
    - path
    - wavelength
    - transmision
    Methods
    -------
    - load_data
    Example
    -------
    """
    database_dir = os.path.join(os.path.dirname(__file__), 'Filters/')

    def __init__(self, name=None, path=None, wavelength=None,
                 transmision=None):
        """# TODO."""
        self.name = name
        self.path = path
        self.wavelength = wavelength
        self.transmision = transmision

        if self.path is not None:
            self.load_data_from_path(self.path)
        elif self.name is not None and self.path is None:
            self.load_data_from_database()

    def load_data_from_path(self, path):
        """Load filter wavelength and response from ascii file."""
        self.wavelength, self.transmision = np.loadtxt(path, unpack=True)

    def load_data_from_database(self):
        """Load filter wavelength and response from ascii file."""
        database = list_filter_dir()
        found = False
        for set_, filters_ in database.items():
            for filter_ in filters_:
                if filter_.find(self.name) >= 0:
                    found = True
                    self.path = os.path.join(self.database_dir,
                                             set_, filter_)
                    self.load_data_from_path(self.path)
            if found:
                break

    def get_mean_wl(self):
        """Return the arithmetic mean wavelength of the filter."""
        mean_wl = (
            np.trapz(self.wavelength * self.transmision, self.wavelength)
            / np.trapz(self.transmision, self.wavelength))
        return mean_wl

    def get_pivot_wl(self):
        """Return the pivot wavelenght of the filter."""
        pvt_wl = (
            np.trapz(self.transmision, self.wavelength)
            / np.trapz(self.transmision / self.wavelength**2, self.wavelength))
        pvt_wl = np.sqrt(pvt_wl)
        return pvt_wl

    def get_eff_wl(self, f, wl):
        """Return the effective wavelength computed from spectra f.

        Params
        ------
        - f: (array) Flux array in FLAM units.
        - wl: (array) Wavelength array in AA.
        """
        T = np.interp(wl, self.wavelength, self.transmision, left=0, right=0)
        eff_wl = np.trapz(f * T * wl**2, wl) / np.trapz(f * T * wl, wl)
        return eff_wl

# Mr Krtxo \(ﾟ▽ﾟ)/

if __name__ == '__main__':
    filter_ = Filter(name='JPLUS_J0660_FullEfficiency')
