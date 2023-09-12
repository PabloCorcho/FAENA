from abc import ABCMeta, abstractclassmethod, abstractproperty
import os
from astropy import units as u

__all__ = ['CatalogueBase']


class CatalogueBase(metaclass=ABCMeta):
    """Catalogue Base class"""
    
    def __init__(self, *args, **kwargs):
        self.name = kwargs.get("name", "unknown")
        self.name = kwargs.get("name", "unknown")

    @property
    def path(self):
        return self._path
    
    @path.setter
    def path(self, value):
        if os.path.isfile(value):
            self._path = value
        else:
            raise FileNotFoundError(f"Path to catalogue: {value} not found")

    @abstractclassmethod
    def load_catalogue(self):
        raise NotImplementedError(
            "Base method needs to be overriden by child Catalogue")

    def save_catalogue(self):
        raise NotImplementedError("TBI")

class CatProperty(object):
    """..."""

    def __init__(self, name, data,
                 units=u.dimensionless_unscaled,
                 description=""):
        self.name = name
        self.data = data
        self.units = units
        self.description = description
        
    @property
    def name(self):
        return self._name
    
    @name.setter
    def name(self, value):
        self._name = value.lower()
    
    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @property
    def units(self):
        return self._units
    
    @units.setter
    def units(self, value):
        if type(value) is u.Unit or type(value) is u.CompositeUnit:
            self._units = value
        else:
            raise TypeError("Input units must be astropy.units.core.Unit")
    
    @property
    def description(self):
        return self._description

    @description.setter
    def description(self, value):
        self._description = value    
    
    