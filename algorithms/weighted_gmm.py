# weighted_gmm.py
# Class implementation for Gaussian Mixture Model (GMM), generlized to allow weighted input data. 
#
# Andrew Roberts

import matplotlib.pyplot as plt
import helper_functions as hf
import numpy as np
import warnings

class Weighted_GMM: 

	def __init__(self, rng, k, prior_threshold): 
		self.rng = rng
		self.k = k
		self.prior_threshold = prior_threshold

	def fit(self, x_arr, w_points = None):

		# If not provided, uniform weights are used.
		if w_points is None:
			w_points = np.ones(shape = x_arr.shape[0])
		w_points = np.array(w_points) / np.sum(w_points)

		n = x_arr.shape[0]

		# Initialize matrix of cluster responsibilities; R_ij = p(point i in cluster j|x_i, mean_j, cov_j)
		# Initial values come from kmeans "hard clustering"; all probability mass assigned to one cluster for each point.
		means_init = hf.weighted_kmeans(x_arr, self.k, self.rng, w_points, centers_init = 'kmeans++')
		r = hf.get_point_assignments(x_arr, means_init)		
		R = np.zeros(shape = (n, self.k))
		R[np.arange(n), r] = 1

		print(R[:20, :])


		'''

		# Optimize parameters given initial cluster responsibilities
		w_clusters, means, covs = self.maximization(x_arr, R)

		while True: 
			# E Step: Compute responsibilities given parameters
			R = self.expectation(x_arr, w_points, w_clusters, means, covs)
		
			# Store old parameters for comparison
	
			# M Step: Optimize parameters given responsibilities
			w_clusters, means, covs = self.maximization(x_arr, R)

		'''

	def auxiliary_function(self, R, w_points, w_clusters, P):
		# Computes the standard EM for GMM auxiliary function Q(theta, theta^(t - 1)), modified so each term 
		# is weighted by point weight. Let N be the number of observations in the data.  
		#
		# Args: 
		#	R: Numpy array of shape (N, k). r_ij = responsibility of cluster j for point i = p(z_i = j|x_i, mean_j, cov_j).
		#	w_points: Numpy array of shape N. Point weights. 
		#	w_clusters: Numpy array of shape k. Gaussian mixture weights. 
		#	P: Numpy array of shape (N, k). p_ij = likelihood point i is from cluster j = p(x_i|mean_j, cov_j).

		return w_points.dot(R.dot(w_clusters)) + w_points.dot((R * P).sum(1))
