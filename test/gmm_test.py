# gmm_test.py
# Test script for Coreset_GMM class, which produces a coreset to be used for parameter
# estimation in Gaussian Mixture Models. 
#
# Andrew Roberts

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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
	#	numpy ndarray of shape (n, d), where each row is an n-dimensional point sampled from the GMM. 

	weights = weights / np.sum(weights)		
	mixture_samples = rng.choice(k, size = n, p = weights)
	gmm_samples = np.array([rng.multivariate_normal(means[j], covs[j]) for j in mixture_samples])
	
	return(gmm_samples)


#
# Test GMM data simulation
#

rng = np.random.default_rng(5)
n = 1000

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
