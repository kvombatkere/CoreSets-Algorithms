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
import time
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
coreset_frac = .05
spectrum_bound = 1/100

# Coreset size
m = int(np.ceil(coreset_frac * N))

# Verify all covariance matrices satisfy spectrum bound
print("\n\nVerifying spectrum bound assumption holds:")
for i in range(len(covs_true)):
	cov_eig_values = np.linalg.eigh(covs_true[i])[0]
	cov_passes = np.logical_and(cov_eig_values >= spectrum_bound, cov_eig_values <= (1 / spectrum_bound))
	print("Cov", i, "satisfies spectrum bound: ", str(cov_passes))

# Instantiate coreset object
coreset = gmm.Coreset_GMM(rng, arr, k, eps, spectrum_bound, delta)

# Generate coreset
C, C_weights = coreset.generate_coreset(m = m)

hf.scatter_2D([arr, C, means_true], title = 'Coreset (size 1%), N = ' + str(N), s = [None, C_weights, None])
plt.savefig(os.path.join(output_path, 'gmm_coreset_rng_' + str(rng_seed) + '.png'))


# --------------------------
# Fit GMM on entire dataset
# --------------------------

# Setup
prior_threshold = .001

# Instantiate Weighted GMM object
gmm_model = wgmm.Weighted_GMM(rng, k, prior_threshold)

# Fit GMM model on all data
print("\n\nFitting GMM on Entire Dataset")
t0 = time.time()
w_all, means_all, covs_all = gmm_model.fit(arr)
t1 = time.time()
print("Time in fit() method:", t1 - t0)


# -------------------
# Fit GMM on coreset
# -------------------

# Fit GMM model on coreset
print("\n\nFitting GMM on Coreset")
t0 = time.time()
w_C, means_C, covs_C = gmm_model.fit(C, C_weights)
t1 = time.time()
print("Time in fit() method:", t1 - t0)


# -----------------------------
# Fit GMM on uniform subsample
# -----------------------------

# Collect uniform subsample
arr_subsample = rng.choice(arr, size = m)

# Fit GMM model on uniform subsample
print("\n\nFitting GMM on Uniform Subsample")
t0 = time.time()
w_uniform, means_uniform, covs_uniform = gmm_model.fit(arr_subsample)
t1 = time.time()
print("Time in fit() method:", t1 - t0)


# ---------------------------
# Compare the three GMM fits
# ---------------------------

















