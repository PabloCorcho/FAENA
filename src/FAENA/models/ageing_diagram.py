#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 10:45:25 2022

@author: pablo
"""

import numpy as np


def exp_sequence(xx, ew_ifty, k, alpha, ew_offset):
    """..."""
    ew = k * 10**(alpha * xx) + ew_ifty
    logew = np.log10(ew + ew_offset)
    return logew


class DemarcationLines(object):
    """..."""

    def __init__(self):
        self.ew_offset = 7
        self.dummy_d4000 = np.linspace(0.9, 2.5)
        self.dummy_gr = np.linspace(0.7, 1.2)
        self.ageing_sequence_params = (-4.5, 45, -1.7, self.ew_offset)
        self.quenched_sequence_params = (1.8, -7, -0.9, self.ew_offset)
        # D4000 FIT
        self.ageing_sequence_params_d4000 = (-4.3, 250, -1.2, self.ew_offset)
        self.quenched_sequence_params_d4000 = (1.8, -12, -0.5, self.ew_offset)
        self.params = {
            'ageing': {
                'gr': (-4.5, 45, -1.7, self.ew_offset),
                'd4000': (-4.3, 250, -1.2, self.ew_offset)},
            'quenched': {
                'gr': (1.8, -7, -0.9, self.ew_offset),
                'd4000': (1.8, -12, -0.5, self.ew_offset)},
                }

    def get_lines(self, x, proxy='gr'):
        """..."""
        ageing = exp_sequence(x, *self.params['ageing'][proxy])
        quenched = exp_sequence(x, *self.params['quenched'][proxy])
        return ageing, quenched

    def get_class(self, x, logew, proxy='gr'):
        """..."""
        ageing, quenched = self.get_lines(x, proxy)

        classes = np.zeros_like(x, dtype=int)

        ageing_galaxies = (logew > ageing) & (logew > quenched)
        quenched_galaxies = (logew < ageing) & (logew < quenched)
        retired_galaxies = (logew > ageing) & (logew < quenched)
        undetermined_galaxies = (logew < ageing) & (logew > quenched)
        classes[ageing_galaxies] = 1
        classes[undetermined_galaxies] = 2
        classes[quenched_galaxies] = 3
        classes[retired_galaxies] = 4
        return classes

