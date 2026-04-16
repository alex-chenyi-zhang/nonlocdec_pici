import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern 
from scipy.special import xlogy
from scipy.optimize import minimize

def samples_from_gp_exponential_profile(x_positions, covariance_kernel, corr_length, delta_x, sigma_ext, sigma_int, mu_S, lamb, n_embryos=1000):
    """
    Generate samples from a GP with an exponential mean profile.

    Parameters:
    x_positions: The positions where the GP is evaluated.
    mu_S: height of the exponential mean profile.
    lamb: decay length of the exponential mean profile.

    covariance_kernel: The covariance kernel to use for the GP.
    corr_length: The correlation length of the GP, expressedn in number of neighbors (at typical spacing)
    delta_x: The spacing between the points in the GP.
    sigma_int: Standard deviation of the spatially independent component of the noise.
    sigma_ext: Standard deviation of the spatially correlated component of the noise.
    n_embryos: The number of samples to generate.

    Returns:
    np.ndarray: The samples from the GP.
    """

    n_cells = len(x_positions)

    mean_values = np.exp(-x_positions/lamb)

    if covariance_kernel == 'SquaredExponential':
        kernel = sigma_ext**2 * RBF(length_scale=corr_length * delta_x)
    elif covariance_kernel == 'SimpleExponential':
        kernel = sigma_ext**2 * Matern(length_scale=corr_length * delta_x, nu=0.5)
    else:
        raise ValueError("Invalid covariance kernel. Choose 'SquaredExponential' or 'SimpleExponential'.")
    
    gp = GaussianProcessRegressor(kernel=kernel)
    samples = gp.sample_y(x_positions, n_samples=n_embryos, random_state=0)

    samples *= mean_values
    if sigma_int != 0:
        samples_with_mean = mu_S * mean_values + samples + np.random.normal(0, sigma_int, (n_cells, n_embryos)) * mean_values
    else:
        samples_with_mean = mu_S * mean_values + samples

    return samples_with_mean.T


def compute_PI(G, c_bins):
    """
    Compute the PI of the ensemble of morphogen profiles contained in G (shape: n_embryos x n_cells).
    
    c_bins: The number of bins to use for the histogram.
    PI is the mutual information between the morphogen level and the cell positions (actually cell index for how this is implemented)
    """
    n_cells = G.shape[1]
    c_max = np.max(G) *1.2
    c_min = np.min(G)#0.
    delta_c = (c_max-c_min)/c_bins
    
    hist, bin_edges = np.histogram(G, bins=c_bins, range=(c_min, c_max), density=True)
    p_g = hist * np.diff(bin_edges)
    
    p_gx = np.zeros((n_cells, c_bins))
    
    for i_cell in range(n_cells):
        hist, bin_edges = np.histogram(G[:,i_cell], bins=c_bins, range=(c_min, c_max), density=True)
        p_gx[i_cell,:] = hist* np.diff(bin_edges)
    
    
    S_pg = -np.sum(xlogy(p_g, p_g)) / np.log(2)
    S_pgx = -np.sum(xlogy(p_gx, p_gx)) / (n_cells * np.log(2))
    
    PI = S_pg - S_pgx
    
    return S_pg, S_pgx, PI

def compute_CI(G, cov = None):
    """
    Compute the CI for the ensemble of morphogen profiles contained in G (shape: n_embryos x n_cells).

    cov: The covariance matrix of the morphogen profiles. If None, it is computed from G.
    """
    n_embryos, n_cells = G.shape
    if cov is None:
        cov = np.cov(G.T, ddof=1)
    var_g = np.diag(cov)
    CI = (np.sum(np.log(var_g)) - np.linalg.slogdet(cov)[1])*np.log2(np.e)/ (2*n_cells)
    return cov, CI
