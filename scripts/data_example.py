
# Imports 
import numpy as np
import matplotlib.pyplot as plt

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
data_indices = [2, 5, 8]
n_param = D.shape[0]
n_data = len(data_indices)
Q_m = D.T@D
mu_m = np.zeros(n_param)
sigma2_eps = 0.001
Sigma_eps = np.identity(n_data)*sigma2_eps
G = np.zeros((n_data,n_param))
for idx, elem in enumerate(data_indices):
    G[idx, elem] = 1
mu_eps = G@mu_m
Q_m_modified = Q_m + np.identity(10)*0.0001
Sigma_m = np.linalg.inv(Q_m_modified)
Sigma_d = G@Sigma_m@G.T + Sigma_eps

def get_example_variables():
    return data_indices, mu_m, Sigma_eps, G, mu_eps, Sigma_m, Q_m, Sigma_d, sigma2_eps


def get_posteior(d):
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