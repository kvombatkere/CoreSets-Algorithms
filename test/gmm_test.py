# gmm_test.py
# Test script for Coreset_GMM class, which produces a coreset to be used for parameter
# estimation in Gaussian Mixture Models. 
#
# Andrew Roberts

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os

def simulate_gmm_data(rng, n, k, means, covs, weights):
	# Returns a random sample of size n from a mixture of k multivariate normal 
	# distributions in d-dimensional space (where d is implied by the dimension of the elements of 
	# 'means' and 'covs')  with mean vectors given in 'mean', covariance matrices given 
	# in 'cov', and mixture weights given in 'weights'. 
	#
	# Args:
	#	rng: numpy random number generator object. 
	#	n: int, the number of samples. 
	#	k: int, the number of Gaussians in the mixture. 
	#	means: numpy ndarray of shape (k, d), where 'mean[i]' gives the mean vector of the ith Gaussian. 	
	#	covs: numpy ndarray of shape (k, d, d), where 'cov[i]' gives the covariance matrix of the ith Gaussian. 
	#	weights: list or numpy ndarray of shape k, where 'weights[i]' gives the mixture weight of the ith Gaussian. 
	#
	# Returns: 
	#	numpy ndarray of shape (n, d), where each row is a d-dimensional point sampled from the GMM. 

	weights = weights / np.sum(weights)		
	mixture_samples = rng.choice(k, size = n, p = weights)
	gmm_samples = np.array([rng.multivariate_normal(means[j], covs[j]) for j in mixture_samples])
	
	return(gmm_samples)

def simulate_gaussian_clusters(rng, n, k, means, covs):
	# Returns a random sample of size (at most) n comprised of points sampled from k different multivariate normal 
	# distributions (note: not a mixture of the distributions, different from GMM). The dimension of the points is  
	# implied by the dimension of the mean vectors and covariance matrices. Denote this dimension by d.  
	#
	# Args: 
	#	rng: numpy random number generator object. 
	#	n: list or int. If list, ith element contains number of samples to be drawn from the ith Gaussian. If int, 
	#	   floor(n/k) samples will be drawn from each Gaussian. 
	#	k: int, the number of Gaussians.  
	#	means: numpy ndarray of shape (k, d), where 'mean[i]' gives the mean vector of the ith Gaussian. 	
	#	covs: numpy ndarray of shape (k, d, d), where 'cov[i]' gives the covariance matrix of the ith Gaussian. 
	#
	# Returns:
	#	numpy ndarray of shape (m, d), where m <= n. Each row is a d-dimensional point sampled from one of the k 
	#	Gaussians.  

	# n interpreted as total number of samples; equally weight clusters
	if len(n) == 1:
		n_cluster = int(np.floor(n / k))
		n = [int(n_cluster for i in range(k))]

	samples = np.concatenate([rng.multivariate_normal(means[j], covs[j], size = n[j]) for j in range(k)])

	return samples

#
# General Setup
#

# Random Number Generator object to be used for all tests
rng = np.random.default_rng(5)

# Load GMM Coreset module
sys.path.insert(1, os.path.join(sys.path[0], '../algorithms'))
import coreset_gmm as gmm


#
# Test GMM data simulation
#

n = 1000

'''

# Bivariate, spherical, k = 1
mean1 = [np.array([0, 0])]
cov1 = [np.array([[1, 0], [0, 1]])]
x1 = simulate_gmm_data(rng, n, 1, mean1, cov1, [1])
plt.hist(x1, 100)
plt.title('Spherical, k = 1, Projection of 2 Components onto 2D Plane')
plt.show()

# Bivariate, non-spherical, k = 1
mean2 = [np.array([0, 0])]
cov2 = [np.array([[3, 0], [0, 1]])]
x2 = simulate_gmm_data(rng, n, 1, mean2, cov2, [1])
plt.hist(x2, 100)
plt.title('Non-Spherical, k = 1, Projection of 2 Components onto 2D Plane')
plt.show()

# Bivariate, spherical, k = 2
mean3 = [np.array([-5, -2]), np.array([5, 2])]
cov3 = [np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]])]
x3 = simulate_gmm_data(rng, n, 2, mean3, cov3, [.5, .5])
plt.hist(x3, 100)
plt.title('Spherical, Symmetric Mixture, k = 2, Projection of 2 Components onto 2D Plane')
plt.show()

#
# Test Gaussian cluster data simulation
#

# Bivariate, k = 3
means = [[5, 5], [-5, -5], [0, 0]]
covs = [np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]]), np.array([[3, 0], [0, 1]])]
x = simulate_gaussian_clusters(rng, [30, 20, 50], 3, means, covs)
plt.scatter([x[i][0] for i in range(len(x))], [x[i][1] for i in range(len(x))])
plt.show()


#
# Test k-means++ algorithm
#

# Setup 
k = 3
eps = .01
spectrum_bound = 1/100
delta = .01

# Simulate bivariate data with 3 clusters
means = [[5, 5], [-5, -5], [0, 0]]
covs = [np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]]), np.array([[3, 0], [0, 1]])]
arr = simulate_gaussian_clusters(rng, [30, 20, 50], k, means, covs)


# Run k-means++
coreset = gmm.Coreset_GMM(rng, arr, k, eps, spectrum_bound, delta) 
B, B_cost = coreset.kmeans_pp()
ax = coreset.scatter_2D()
ax = coreset.scatter_2D(B, ax)

# Obtain bicriteria approximation by running k-means++ multiple times and selecting the 
# best set of k initialization points. 
B_star, B_star_cost = coreset.get_bicriteria_kmeans_approx() 
print("B_star_cost = ", B_star_cost)
ax = coreset.scatter_2D(B_star, ax)
plt.show()

'''

#
# Test coreset generation
#

# Setup 
k = 3
eps = .01
spectrum_bound = 1/100
delta = .01

# Simulate bivariate data with 3 clusters
means = [[5, 5], [-5, -5], [0, 0]]
covs = [np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]]), np.array([[3, 0], [0, 1]])]
arr = simulate_gaussian_clusters(rng, [3000, 2000, 5000], k, means, covs)

# Generate coreset
coreset = gmm.Coreset_GMM(rng, arr, k, eps, spectrum_bound, delta)
C, C_weights = coreset.generate_coreset(m = 100)

ax = coreset.scatter_2D()
ax = coreset.scatter_2D(C, ax)
plt.show()






