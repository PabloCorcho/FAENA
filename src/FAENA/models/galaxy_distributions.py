import numpy as np


def schechter(mag, mag_star, phi, alpha):
    s = 0.4 * np.log(10) * phi * (10**(0.4 * (mag_star-mag)))**(1+alpha) *\
        np.exp(-10**(0.4 * (mag_star-mag)))
    return s


class LuminosityFunction:
    """Container class for galaxy luminosity distribution functions.
    
    TODO
    """

    def blanton03(r_abs, H0=70, redshift=0):
        """..."""
        mag_star = - 20.44 - 5 * np.log10( 1 / H0 * 100)
        alpha = - 1.05
        phi = 1.49e-2 * (H0/100)**3
        # z-evolution parameters (Lin+99)
        q = 1.62
        mag_star_corr = mag_star + q * (0.1 - redshift)
        p = 0.18
        phi = (phi * 10**(0.4 * p * redshift) / 10**(0.4 * p * 0.1))
        return schechter(r_abs, mag_star_corr, phi, alpha)
        


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    import pandas as pd    
    min_lum = -18.6
    dr = 0.1
    r_binedges = np.arange(-24, -15, dr)
    r_bins = r_binedges[:-1] + dr / 2

    corcho20 = pd.read_csv('/home/pablo/Research/obs_data/SDSS/galaxy_bimodality_sample_absvalues.csv')
    corcho_r = corcho20['r_abs'].values
    mask = corcho_r < min_lum
    corchoVmax = corcho20['Vmax'].values[mask]
    corcho_g = corcho20['g_abs'].values[mask]
    corcho_r = corcho_r[mask]
    
    corcho20_g = pd.read_csv('/home/pablo/Research/obs_data/GAMA/gama_sample.csv')
    corcho_r_g = corcho20_g['r_abs'].values
    mask_g = corcho_r_g < min_lum
    corchoVmax_g = corcho20_g['vmax'].values[mask_g]
    corcho_g_g = corcho20_g['g_abs'].values[mask_g]
    corcho_r_g = corcho_r_g[mask_g]
    
    h_corcho, _ = np.histogram(corcho_r, bins=r_binedges,
                               weights=1/corchoVmax)
    h_corcho_g, _ = np.histogram(corcho_r_g, bins=r_binedges,
                                 weights=1/corchoVmax_g)
    
    phi_corcho = h_corcho/dr
    phi_corcho_g = h_corcho_g/dr
    
    dndr = LuminosityFunction.blanton03(r_bins)
    
    plt.figure()
    plt.plot(r_bins, phi_corcho, '--', label='Corcho+20 SDSS')
    plt.plot(r_bins, phi_corcho_g, '--', label='Corcho0 GAMA')
    plt.plot(r_bins, dndr)
    plt.axvline(-18.6, ls='--', c='k')
    plt.axvline(-23.1, ls='--', c='k')
    plt.legend()
    plt.ylabel(r'$\Phi$ (Mpc$^{-3}$ mag$^{-1}$)')
    plt.xlabel(r'$M_r$')
    plt.grid(b=True)
    plt.yscale('log')
    plt.ylim(1e-6, 1e-1)
    plt.xlim(-16, -24)