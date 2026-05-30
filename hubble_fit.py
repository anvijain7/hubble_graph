import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "Pantheon+SH0ES.dat")

# Column names from the file header
# CID IDSURVEY zHD zHDERR zCMB zCMBERR zHEL zHELERR m_b_corr m_b_corr_err_DIAG MU_SH0ES MU_SH0ES_ERR_DIAG ...

def load_data(path):
    # use comment parsing to skip the header line(s) if needed
    with open(path, "r") as f:
        header = f.readline().strip().split()

    # We want zHD (column 2) and MU_SH0ES_ERR_DIAG (column 12 from the header after MU_?)
    # The header shows zHD at index 2 and MU_SH0ES_ERR_DIAG at index 12 (0-based index 13).
    # Also use m_b_corr or MU_ if desired. Here we use zHD and MU_SH0ES_ERR_DIAG.
    data = np.loadtxt(path, comments="#", skiprows=1)
    return data


def luminosity_distance_modulus(z, H0, M):
    # For local Hubble flow, approximate distance modulus using d_L = c z / H0 with c in km/s.
    c_kms = 299792.458
    d_L_Mpc = c_kms * z / H0
    mu = 5 * np.log10(d_L_Mpc) + 25 + M
    return mu


def main():
    data = load_data(DATA_PATH)

    z = data[:, 2]  # zHD
    mu = data[:, 10]  # MU_SH0ES
    mu_err = data[:, 11]  # MU_SH0ES_ERR_DIAG

    # Filter to z <= 0.1
    mask = z <= 0.1
    z_fit = z[mask]
    mu_fit = mu[mask]
    mu_err_fit = mu_err[mask]

    # Fit a straight line in log space or directly using the Hubble law approximation
    # Use a simple linear model for mu(z) in the local regime via the Hubble flow.
    def model(z_val, H0, M):
        return luminosity_distance_modulus(z_val, H0, M)

    initial_guess = [70.0, 0.0]
    popt, pcov = curve_fit(model, z_fit, mu_fit, sigma=mu_err_fit, absolute_sigma=True, p0=initial_guess)
    H0_fit, M_fit = popt
    H0_err, M_err = np.sqrt(np.diag(pcov))

    print(f"Fitted local Hubble constant: H0 = {H0_fit:.2f} +/- {H0_err:.2f} km/s/Mpc")
    print(f"Fitted nuisance offset M = {M_fit:.3f} +/- {M_err:.3f}")

    # Plot data with error bars and fitted line
    z_plot = np.linspace(0, 0.1, 200)
    mu_model = model(z_plot, H0_fit, M_fit)

    plt.errorbar(z_fit, mu_fit, yerr=mu_err_fit, fmt="o", markersize=4, alpha=0.7, label="Data (z <= 0.1)")
    plt.plot(z_plot, mu_model, color="red", lw=2, label=f"Fit: H0={H0_fit:.1f} km/s/Mpc")
    plt.xlabel("Redshift z")
    plt.ylabel("Distance Modulus $\mu$")
    plt.title("Local Hubble Flow Fit to SH0ES Pantheon+ Data")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("/Users/Anvii/project/hubble_research/local_H0_fit.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
