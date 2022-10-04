import numpy as np


def get_zeros(f, x):
    """Get all the roots of a function."""
    x_zeros = []

    zeros_negative = np.where(f <= 0)[0]
    zeros_positive = np.where(f >= 0)[0]
    if (len(zeros_negative) < 1) | (len(zeros_positive) < 1):
        return x_zeros

    left_sign_change = np.where(
        zeros_negative[1:] - zeros_negative[:-1] > 1)[0]
    right_sign_change = np.where(
        zeros_positive[1:] - zeros_positive[:-1] > 1)[0]

    if len(left_sign_change) > 0:
        # Multiple zeros
        for i in left_sign_change:
            zero_n = zeros_negative[i]
            zero_p = zeros_negative[i] + 1
            x_zero = np.interp(0, [f[zero_n], f[zero_p]],
                               [x[zero_n], x[zero_p]])
            x_zeros.append(x_zero)
    else:
        i = 0
    # Account for the last part of the function where presented negative values
    zero_n = zeros_negative[i:][-1]
    zero_p = np.where(zeros_positive > zero_n)[0]
    if len(zero_p) > 0:
        zero_p = zeros_positive[zero_p[0]]
        x_zero = np.interp(0, [f[zero_n], f[zero_p]],
                           [x[zero_n], x[zero_p]])
        x_zeros.append(x_zero)

    if len(right_sign_change) > 0:
        for i in right_sign_change:
            zero_n = zeros_positive[i] + 1
            zero_p = zeros_positive[i]
            x_zero = np.interp(0, [f[zero_n], f[zero_p]],
                               [x[zero_n], x[zero_p]])
            x_zeros.append(x_zero)
    else:
        i = 0
    zero_p = zeros_positive[i:][-1]
    zero_n = np.where(zeros_negative > zero_p)[0]
    if len(zero_n) > 0:
        zero_n = zeros_negative[zero_n[0]]
        x_zero = np.interp(0, [f[zero_n], f[zero_p]],
                           [x[zero_n], x[zero_p]])
        x_zeros.append(x_zero)

    return np.unique(x_zeros)

# Mr Krtxo \(ﾟ▽ﾟ)/
