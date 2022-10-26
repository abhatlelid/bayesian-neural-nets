
# Imports 
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

#variables
D = np.array(
    [[1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, -1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0,  1, -1, 0, 0, 0, 0, 0, 0],
    [0, 0,  0, 1, -1, 0, 0, 0, 0, 0],
    [0, 0,  0, 0, 1, -1, 0, 0, 0, 0],
    [0, 0,  0, 0, 0, 1, -1, 0, 0, 0],
    [0, 0,  0, 0, 0, 0, 1, -1, 0, 0],
    [0, 0,  0, 0, 0, 0, 0, 1, -1, 0],
    [0, 0,  0, 0, 0, 0, 0, 0, 1, -1],
    [-1, 0, 0, 0, 0, 0, 0, 0, 0, 1],]
)
sigma2_eps = 0.001
n_param = D.shape[0]
Q_m = D.T@D
mu_m = np.zeros(n_param)
Q_m_modified = Q_m + np.identity(10)*0.001
Sigma_m = np.linalg.inv(Q_m_modified)
Sigma_eps = np.identity(n_param)*sigma2_eps
Sigma_d = Sigma_m + Sigma_eps


def get_sample_and_variables(batch_size, n_batches):
    N = batch_size*n_batches
    d_sample = np.random.multivariate_normal(mu_m.flatten(), Sigma_d, size=N)
    combined_sample = np.zeros((N, 2*n_param))
    mask = np.zeros((N, n_param))
    range_vector = np.zeros((N, n_param)) 
    range_vector[:,] = np.arange((n_param))
    n_masked = np.random.randint(n_param, size=N)
    n_masked = n_masked.reshape((N, 1))
    bool_vector = range_vector[:,] < n_masked
    mask[bool_vector] = 1
    np.random.shuffle(mask.T)
    d_sample = d_sample*mask
    combined_sample[:,:n_param] = d_sample
    combined_sample[:,n_param:] = mask

    return combined_sample, Sigma_m, Q_m, sigma2_eps


def get_input_tensor(d, x, n_param=10):
    input_tensor = torch.zeros(n_param*2)
    for d_, x_ in zip(d, x):
        input_tensor[x_] = d_
        input_tensor[n_param+x_] = 1
    return input_tensor


def get_posteior(d, x):
    n_data = len(d)
    Sigma_eps = np.identity(n_data)*sigma2_eps

    G = np.zeros((n_data,n_param))
    for idx, elem in enumerate(x):
        G[idx, elem] = 1
    mu_eps = G@mu_m

    # Matrix calculations
    Sigma_mm = Sigma_m
    Sigma_dm = G@Sigma_mm
    Sigma_md = Sigma_mm@(G.T)
    Sigma_dd = G@Sigma_mm@(G.T) + Sigma_eps
    Sigma_dd_inv = np.linalg.inv(Sigma_dd)
    mu_m_d = mu_m + Sigma_md@Sigma_dd_inv@(d-mu_eps)
    Sigma_m_d = Sigma_mm - Sigma_md@Sigma_dd_inv@Sigma_dm 
    return mu_m_d, Sigma_m_d


def plot_posterior(mu_m_d, Sigma_m_d):
    # Plotting the posterior of m given d
    sigma_m_d_ii = np.diagonal(Sigma_m_d)
    m = range(0, 10)
    plt.errorbar(
        x = m,
        y = mu_m_d, 
        yerr=sigma_m_d_ii,
        xerr=None,
        fmt='o',
        ecolor='k',
        elinewidth=1.2, 
        capsize=2,
        linestyle='-',
    )
    plt.xlabel('m')
    plt.ylabel('m|d')
    plt.grid(color='silver', linestyle='-', linewidth=1)
    plt.xticks(np.arange(len(m)))
    plt.title('Posterior mean with posterior standard deviation')
    plt.show()


if __name__ == "__main__":
    get_sample_and_variables(batch_size=64, n_batches=10000)