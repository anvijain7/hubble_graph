import os
import numpy as np
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit
from scipy.stats import norm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "Pantheon+SH0ES.dat")

# Column names from the file header
# CID IDSURVEY zHD zHDERR zCMB zCMBERR zHEL zHELERR m_b_corr m_b_corr_err_DIAG MU_SH0ES MU_SH0ES_ERR_DIAG ...

def load_data(path):
    data = np.genfromtxt(path, names=True, dtype=None, encoding=None)
    return data


def model(z, H0):
    c_kms = 299792.458
    d_L_Mpc = c_kms * z / H0
    return 5 * np.log10(d_L_Mpc) + 25


def main():
    data = load_data(DATA_PATH)

    z = data['zHD']
    mu = data['MU_SH0ES']
    mu_err = data['MU_SH0ES_ERR_DIAG']

    mask = (z >= 0.01) & (z <= 0.1) & (mu > 0) & (mu_err > 0) & (mu_err < 5)
    z_fit = z[mask]
    mu_fit = mu[mask]
    mu_err_fit = mu_err[mask]

    initial_guess = [70.0]
    popt, pcov = curve_fit(model, z_fit, mu_fit, sigma=mu_err_fit, absolute_sigma=True, p0=initial_guess)
    H0_fit = popt[0]
    H0_err = np.sqrt(np.diag(pcov))[0]

    print(f"Fitted local Hubble constant: H0 = {H0_fit:.2f} +/- {H0_err:.2f} km s^-1 Mpc^-1")

    # Plot data with error bars and fitted line
    z_plot = np.linspace(0.01, 0.1, 200)
    mu_model = model(z_plot, H0_fit)

    plt.errorbar(z_fit, mu_fit, yerr=mu_err_fit, fmt="o", markersize=4, alpha=0.7, label="Data (z <= 0.1)")
    plt.plot(z_plot, mu_model, color="red", lw=2, label=f"Fit: H0={H0_fit:.2f} km s^-1 Mpc^-1")
    plt.xlabel("Redshift z")
    plt.ylabel("Distance Modulus $\mu$")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("local_H0_fit.png", dpi=200)
    plt.show()

    # Residuals histogram
    residuals = (mu_fit - model(z_fit, H0_fit)) / mu_err_fit
    plt.figure()
    plt.hist(residuals, bins=30, density=True, alpha=0.6, color='C0', edgecolor='black')
    x = np.linspace(-5, 5, 400)
    plt.plot(x, norm.pdf(x, 0, 1), 'r-', lw=2, label='Standard Normal')
    plt.xlabel('Normalised Residuals (σ)')
    plt.ylabel('Density')
    plt.title('Residuals Distribution')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig('residuals_histogram.png', dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
