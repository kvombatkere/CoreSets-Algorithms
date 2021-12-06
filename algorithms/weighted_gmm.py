# weighted_gmm.py
# Class implementation for Gaussian Mixture Model (GMM), generlized to allow weighted input data. 
#
# Andrew Roberts


import matplotlib.pyplot as plt
import numpy as np
import warnings

class Weighted_GMM: 

	def __init__(self, rng, k, prior_threshold): 
		self.rng = rng
		self.k = k
		self.prior_threshold = prior_threshold

	def fit(self, x_array, w_points):

		# Initialize matrix of cluster responsibilities; r_ij = p(point i in cluster j|x_i, mean_j, cov_j)
		# TODO: need to turn this into matrix
		r = self.weighted_kmeans(w_points, centers_init = 'kmeans++')
		
		# Optimize parameters given initial cluster responsibilities
		w_clusters, means, covs = self.maximization(x_array, r)

		while True: 
			# E Step: Compute responsibilities given parameters
			r = self.expectation(X, w_points, w_clusters, means, covs)
		
			# Store old parameters for comparison
	
			# M Step: Optimize parameters given responsibilities
			w_clusters, means, covs = self.maximization(x_array, r)

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
