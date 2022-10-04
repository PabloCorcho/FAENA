"""Created on Sun Aug 14 18:39:02 2022.

@author: pablo

Module containing methods to resample images.
"""

import numpy as np

from FAENA.math.interpolate import regular_bilin_interp

# __all__ = ['flux_conserving_resampling']


def flux_conserving_resampling(new_x, new_y, x, y, image):
    """Flux-conserving image interpolation.

    Params
    ------
    - new_x: (1D or 2D array)
    - new_y: (1D or 2D array)
    - x: (1D array)
    - y: (1D array)
    - image: (2D array)
    """
    invert_0 = False
    invert_1 = False
    if (new_x[:, 1:] < new_x[:, :-1]).all():
        invert_0 = True
    if (new_y[1:, :] < new_x[:-1, :]).all():
        invert_1 = True
        print('Invert 1')

    cum0 = np.nancumsum(image, axis=0)
    cum1 = np.nancumsum(cum0, axis=1)
    interp_cum = regular_bilin_interp(x, y, cum1, new_x, new_y)
    if invert_0:
        new_cum0 = interp_cum[:, :-1] - interp_cum[:, 1:]
    else:
        new_cum0 = interp_cum[:, 1:] - interp_cum[:, :-1]
    if invert_1:
        new_image = new_cum0[:-1, :] - new_cum0[1:, :]
    else:
        new_image = new_cum0[1:, :] - new_cum0[:-1, :]
    return new_image

# Mr Krtxo \(ﾟ▽ﾟ)/
