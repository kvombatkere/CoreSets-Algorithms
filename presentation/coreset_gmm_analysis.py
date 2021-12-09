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

# Number of samples
N = 20000

# Integer indicator to determine which dataset is used 
dataset_num = 2

if dataset_num == 1:

		#
		# Dataset 1
		#

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

elif dataset_num == 2: 

		#
		# Dataset 2
		#

		# Number of clusters (i.e. number of Gaussians in the mixture)
		k = 7

		# Mixture Weights
		w_true = np.array([.05, .01, .001, .001, .7, .1, .138])

		# Cluster means
		means_true = rng.integers(low = -100, high = 100, size = (7, 2))

		# Cluster variances
		covs_true = np.array([np.diag([50.0, 50.0]), 
							  np.diag([14.0, 7.0]),
							  np.diag([20.0, 20.0]), 
							  np.diag([35.0, 70.0]),
							  np.diag([40.0, 40.0]),
							  np.diag([10.0, 10.0]), 
							  np.diag([5.0, 7.0])])


# Generate synthetic data
arr = hf.simulate_gmm_data(rng, N, k, means_true, covs_true, w_true)
hf.scatter_2D([arr, means_true], title = 'Simulated GMM Data, N = ' + str(N))
plt.savefig(os.path.join(output_path, 'simulated_gmm_data_rng_' + str(rng_seed) + '_dataset_' + str(dataset_num) + '.png'))
plt.close()

print("True Means:")
print(means_true)

print("True Weights:")
print(w_true)

print("True Covs:")
print(covs_true)


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

hf.scatter_2D([arr, C, means_true], title = 'Coreset, N = ' + str(N) + ', m = ' + str(m), s = [None, C_weights, None])
plt.savefig(os.path.join(output_path, 'gmm_coreset_rng_' + str(rng_seed) + '.png'))
plt.close()


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


# Evaluate log likelihood
L_all = gmm_model.evaluate_log_likelihood(arr, w_all, means_all, covs_all)
print("Log Likelihood, All Data:", L_all)



# -------------------
# Fit GMM on coreset
# -------------------

# Fit GMM model on coreset
print("\n\nFitting GMM on Coreset")
t0 = time.time()
w_C, means_C, covs_C = gmm_model.fit(C, C_weights)
t1 = time.time()
print("Time in fit() method:", t1 - t0)

print("Coreset Means:")
print(means_C)

print("Coreset Weights:")
print(w_C)

print("Coreset Covs:")
print(covs_C)

# Evaluate log likelihood
L_coreset = gmm_model.evaluate_log_likelihood(arr, w_C, means_C, covs_C)
print("Log Likelihood, Coreset:", L_coreset)

# -----------------------------
# Fit GMM on uniform subsample
# -----------------------------

# Collect uniform subsample
arr_subsample = rng.choice(arr, size = m)

hf.scatter_2D([arr, arr_subsample, means_true], title = 'Uniform Subsample, N = ' + str(N) + ', m = ' + str(m))
plt.savefig(os.path.join(output_path, 'gmm_subsample_rng_' + str(rng_seed) + '.png'))
plt.close()

# Fit GMM model on uniform subsample
print("\n\nFitting GMM on Uniform Subsample")
t0 = time.time()
w_uniform, means_uniform, covs_uniform = gmm_model.fit(arr_subsample)
t1 = time.time()
print("Time in fit() method:", t1 - t0)

print("Uniform Subsample Means:")
print(means_uniform)

print("Uniform Subsample Weights:")
print(w_uniform)

print("Uniform Subsample Covs:")
print(covs_uniform)

# Evaluate log likelihood
L_uniform = gmm_model.evaluate_log_likelihood(arr, w_uniform, means_uniform, covs_uniform)
print("Log Likelihood, Uniform Subsampling:", L_uniform)


# ------------------------------
# Compare the different GMM fits
# ------------------------------

# Data generated via coreset estimates
arr_coreset_generated = hf.simulate_gmm_data(rng, N, k, means_C, covs_C, w_C)
hf.scatter_2D([arr_coreset_generated, means_true, means_C], title = 'GMM Data Simulated with Coreset Estimates, N = ' + str(N))
plt.savefig(os.path.join(output_path, 'gmm_coreset_generation_rng_' + str(rng_seed) + '.png'))
plt.close()

# Data generated via uniform subsample estimates
arr_uniform_generated = hf.simulate_gmm_data(rng, N, k, means_uniform, covs_uniform, w_uniform)
hf.scatter_2D([arr_uniform_generated, means_true, means_uniform], title = 'GMM Data Simulated with Uniform Subsample  Estimates, N = ' + str(N))
plt.savefig(os.path.join(output_path, 'gmm_subsample_generation_rng_' + str(rng_seed) + '.png'))
plt.close()






