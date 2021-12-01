# gmm_test.py
# Test script for Coreset_GMM class, which produces a coreset to be used for parameter
# estimation in Gaussian Mixture Models. 
#
# Andrew Roberts

import pandas as pd
import numpy as np

def simulate_gmm_data(rng, n, k, d, means, covs, weights):
	# Returns a random sample of size n from a mixture of k multivariate normal 
	# distributions in d-dimensional space with mean vectors given in 'mean', covariance matrices given 
	# in 'cov', and mixture weights given in 'weights'. 
	#
	# Args:
	#	rng: numpy random number generator object. 
	#	n: int, the number of samples. 
	#	k: int, the number of Gaussians in the mixture. 
	#	d: int, the dimension of the space. 
	#	means: numpy ndarray of shape (k, d), where 'mean[i]' gives the mean vector of the ith Gaussian. 	
	#	covs: numpy ndarray of shape (k, d, d), where 'cov[i]' gives the covariance matrix of the ith Gaussian. 
	#	weights: list or numpy ndarray of shape k, where 'weights[i]' gives the mixture weight of the ith Gaussian. 
	#
	# Returns: 
	#	numpy ndarray of shape (n, d), where each row is an n-dimensional point sampled from the GMM. 

	weights = weights / np.sum(weights)		
	mixture_samples = np.choice(k, size = n, p = weights)
	gmm_samples = np.array([rng.multivariate_normal(means[j], covs[j]) for j in mixture_samples])


#
# Test
#

rng = np.random.default_rng(5)
