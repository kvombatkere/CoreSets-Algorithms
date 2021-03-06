# gmm_test.py
# Test script for Coreset_GMM class, which produces a coreset to be used for parameter
# estimation in Gaussian Mixture Models. 
#
# Andrew Roberts
#
# Working Directory: CoreSets-Algorithms/test

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os


#
# General Setup
#

# Random Number Generator object to be used for all tests
rng = np.random.default_rng(5)

# Load GMM Coreset module
sys.path.insert(1, os.path.join(sys.path[0], '../algorithms'))
import coreset_gmm as gmm
import weighted_gmm as wgmm
import helper_functions as hf

# Data sets to be used for many of the tests:

# 1.) Gaussian, uniform mixture weights, Bivariate, 3 cluster, N = 10000
means1 = [[5, 5], [-5, -5], [0, 0]]
covs1 = [np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]]), np.array([[7, 0], [0, 1]])]
arr1 = hf.simulate_gaussian_clusters(rng, [3000, 2000, 5000], 3, means1, covs1)


#
# Test GMM data simulation
#

n = 1000


# Bivariate, spherical, k = 1
mean1 = [np.array([0, 0])]
cov1 = [np.array([[1, 0], [0, 1]])]
x1 = hf.simulate_gmm_data(rng, n, 1, mean1, cov1, [1])
plt.hist(x1, 100)
plt.title('Spherical, k = 1, Projection of 2 Components onto 2D Plane')
plt.show()

# Bivariate, non-spherical, k = 1
mean2 = [np.array([0, 0])]
cov2 = [np.array([[3, 0], [0, 1]])]
x2 = hf.simulate_gmm_data(rng, n, 1, mean2, cov2, [1])
plt.hist(x2, 100)
plt.title('Non-Spherical, k = 1, Projection of 2 Components onto 2D Plane')
plt.show()

# Bivariate, spherical, k = 2
mean3 = [np.array([-5, -2]), np.array([5, 2])]
cov3 = [np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]])]
x3 = hf.simulate_gmm_data(rng, n, 2, mean3, cov3, [.5, .5])
plt.hist(x3, 100)
plt.title('Spherical, Symmetric Mixture, k = 2, Projection of 2 Components onto 2D Plane')
plt.show()

#
# Test Gaussian cluster data simulation
#

# Bivariate, k = 3
means = [[5, 5], [-5, -5], [0, 0]]
covs = [np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]]), np.array([[3, 0], [0, 1]])]
x = hf.simulate_gaussian_clusters(rng, [30, 20, 50], 3, means, covs)
plt.scatter([x[i][0] for i in range(len(x))], [x[i][1] for i in range(len(x))])
plt.title("3 Gaussian Clusters")
plt.show()


#
# Test k-means++ algorithm
#

# Run k-means++
B, B_cost = hf.kmeans_pp(arr1, k, rng)
ax = hf.scatter_2D([arr1, B], title = 'kmeans++')
plt.show()


#
# Test coreset generation
#

# Setup 
k = 3
eps = .01
spectrum_bound = 1/100
delta = .01

# Instantiate coreset object
coreset = gmm.Coreset_GMM(rng, arr1, k, eps, spectrum_bound, delta)

# Obtain bicriteria approximation by running k-means++ multiple times and selecting the 
# best set of k initialization points. 
B_star, B_star_cost = coreset.get_bicriteria_kmeans_approx() 
print("B_star_cost = ", B_star_cost)
ax = hf.scatter_2D([arr1, B_star], title = 'Bicriteria Approximation')
plt.show()

# Generate coreset
C, C_weights = coreset.generate_coreset(m = 100)

ax = hf.scatter_2D([arr1, C], title = 'Coreset of size 100')
plt.show()


#
# Test Weighted KMeans
#

# Setup 
k = 3

# Run Kmeans, initialized with kmeans++ (uniform weights)
centers = hf.weighted_kmeans(arr1, k, rng, centers_init = 'kmeans++')
ax = hf.scatter_2D([arr1, centers], title = 'Weighted kmeans: Uniform Weights')
plt.show()

# Run Kmeans, initialized with kmeans++, weighting points by their contribution to within cluster variance
w1 = np.array([hf.dist(arr1[i], means1[0]) for i in range(3000)])
w2 = np.array([hf.dist(arr1[i + 3000], means1[1]) for i in range(2000)])
w3 = np.array([hf.dist(arr1[i + 5000], means1[2]) for i in range(5000)])
w = np.concatenate([w1, w2, w3])

centers = hf.weighted_kmeans(arr1, k, rng, w = w, centers_init = 'kmeans++')
ax = hf.scatter_2D([arr1, centers], title = 'Weighted kmeans: Weights Proportional to Within-Cluster Variance Contribution')
plt.show()


#
# Test Weighted GMM 
#

# Setup
k = 3
prior_threshold = .001
spectrum_bound = .001
delta = .01
eps = .01

# Instantiate GMM Coreset and Weighted GMM objects
coreset = gmm.Coreset_GMM(rng, arr1, k, eps, spectrum_bound, delta)
gmm_model = wgmm.Weighted_GMM(rng, k, prior_threshold)

# Compute coreset of size 1% of data
C, w_C = coreset.generate_coreset(m = int(np.floor(.01 * len(arr1))))

# Fit GMM model on all data
w, m, c = gmm_model.fit(arr1)
print('\n\nGMM on all data:')
print('\nMixture Weights: ', w)
print('\nMeans: ', m)
print('\nCovariances:')
for cov in c:
	print(cov)

# Fit GMM model on coreset
w, m, c = gmm_model.fit(C, w_C)
print('\n\nGMM on coreset:')
print('\nMixture Weights: ', w)
print('\nMeans: ', m)
print('\nCovariances:')
for cov in c:
	print(cov)

#
# EM Algorithm Test: Simple Test 
#

# Simple univariate data
arr_test = np.array([0, 0, 10, 11, 15, 1, 2, 2, 3, 10])

# Instantiate Weighted GMM object
gmm_test = wgmm.Weighted_GMM(rng, 2, .001)
centers_init = hf.weighted_kmeans(arr_test, gmm_test.k, gmm_test.rng, centers_init = 'kmeans++')

# Fit GMM
w_clusters, means, covs = gmm_test.fit(arr_test, centers_init = centers_init)

print(w_clusters)
print(means)
print(covs)












