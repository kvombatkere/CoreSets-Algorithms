# coreset_gmm_analysis.py
# Evaluate GMM coreset effect on speed and accuracy. Produces analysis outputs and plots 
# for presentation. 
#
# Andrew Roberts
#
# Working Directory: CoreSets-Algorithms/

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os


# --------------
# General Setup
# --------------

# Working directory 
base_path = '.'

# Random Number Generator object to be used for all tests
rng_seed = 2
rng = np.random.default_rng(rng_seed)

# Load GMM Coreset module
sys.path.insert(0, os.path.join(base_path, 'algorithms'))
import coreset_gmm as gmm
import weighted_gmm as wgmm
import helper_functions as hf

# Where to save outputs
output_path = os.path.join(base_path, 'output')


# ------------------
# Simulate GMM Data
# ------------------

# Number of clusters (i.e. number of Gaussians in the mixture)
k = 5

# Mixture Weights
w_true = np.array([.07, .05, .5, .2, .18])

# Cluster means
means_true = rng.integers(low = -50, high = 50, size = (5, 2))

# Cluster variances
covs_true = np.array([np.diag([10.0, 10.0]), 
					  np.diag([7.0, 1.0]),
					  np.diag([20.0, 20.0]), 
					  [[3.0, 2.0], [2.0, 5.0]], 
					  [[11.0, 10.0], [10.0, 10.0]]])

# Number of samples
N = 10000

# Generate synthetic data
arr = hf.simulate_gmm_data(rng, N, k, means_true, covs_true, w_true)
hf.scatter_2D([arr, means_true], title = 'Simulated GMM Data, N = ' + str(N))
plt.savefig(os.path.join(output_path, 'simulated_gmm_data_rng_' + str(rng_seed) + '.png'))
plt.close()


# ------------------
# Construct Coreset
# ------------------

# Setup 
eps = .01
delta = .01
coreset_frac = .01
spectrum_bound = 1/100

# Verify all covariance matrices satisfy spectrum bound
for i in range(len(covs_true)):
	cov_eig_values = np.linalg.eigh(covs_true[i])[0]
	cov_passes = np.logical_and(cov_eig_values >= spectrum_bound, cov_eig_values <= (1 / spectrum_bound))
	print("Cov", i, "satisfies spectrum bound: ", str(cov_passes))

# Instantiate coreset object
coreset = gmm.Coreset_GMM(rng, arr, k, eps, spectrum_bound, delta)

# Generate coreset
C, C_weights = coreset.generate_coreset(m = int(np.ceil(coreset_frac * N)))

hf.scatter_2D([arr, C, means_true], title = 'Coreset (size 1%), N = ' + str(N), s = [None, C_weights, None])
plt.savefig(os.path.join(output_path, 'gmm_coreset_rng_' + str(rng_seed) + '.png'))












