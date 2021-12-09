# weighted_gmm.py
# Class implementation for Gaussian Mixture Model (GMM), generalized to allow weighted input data. 
#
# Andrew Roberts

from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import helper_functions as hf
import numpy as np
import warnings

class Weighted_GMM: 
	"""
	Class to fit a GMM model on weighted input data via a modified EM algorithm that accounts 
	for the point weights. 
	----------
	rng: numpy random number generator object. 
	k: int, the number of Gaussians in the mixture (i.e. number of "clusters"). 
	prior_threshold: float, a small number that is added to each diagonal entry of the covariance 
					 matrices to prevent singular matrices. 
	----------

	References: 
	Mario Lucic, Matthew Faulkner, Andreas Krause, and Dan Feldman. "Training Gaussian Mixture 
	Models at Scale via Coresets." Journal of Machine Learning Research. 2018."
	"""

	def __init__(self, rng, k, prior_threshold): 
		self.rng = rng
		self.k = k
		self.prior_threshold = prior_threshold

	def fit(self, x_arr, w_points = None, centers_init = 'kmeans', tol = 1e-8, max_itr = 10000):
		# Estimate the parameter values of a weighted GMM model (i.e. GMM that takes point weights into 
		# account) via a weighted generalization of the EM algorithm. 
		#
		# Args: 
		#	x_arr: numpy array of shape (# observations, # features). 
		#   w_points: numpy array of shape '# observations'; the point weights.
		#	centers_init: Determines the initialization of the parameter values for the first iteration 
		#			      of the EM algorithm. If 'kmeans' initialization is determined by running a weighted
		#				  version of kmeans. If 'uniform_subsample', initializes the k centers by randomly 
		#				  sampling from 'x_arr'. Otherwise, must be a numpy array of shape (k, # features), 
		#				  in which case this array is used as the initial Gaussian means. 
		#	tol: float, the EM algorithm will terminate when the negative log-likelihood decreases by less than 
		#		 'tol' or 'max_iter' iterations are reached. 
		#	max_itr: integer, see 'tol' above. 
		#
		# Returns:
		#	3-tuple containing: 
		#		- mixture weights: numpy array of shape k. 
		#		- means: numpy array of shape (k, # features).
		#		- covariance matrices: numpy array of shape (k, # features, # features)

		# For univariate data, convert to # observations x 1 array
		if len(x_arr.shape) == 1:
			x_arr.shape = (len(x_arr), 1)

		# If not provided, uniform weights are used.
		if w_points is None:
			w_points = np.ones(shape = x_arr.shape[0])
		w_points = np.array(w_points) / np.sum(w_points)

		n = x_arr.shape[0]

		# Initialize matrix of cluster responsibilities; R_ij = p(point i in cluster j|x_i, mean_j, cov_j)
		# Initial values come from kmeans "hard clustering"; all probability mass assigned to one cluster for each point.
		if hasattr(centers_init, '__len__') and (not isinstance(centers_init, str)):
			if len(centers_init) == self.k:
				means_init = centers_init
			else:
				raise ValueError('If manually specifying initial centers, <centers_init> must have length k')
		elif centers_init == 'kmeans':
			means_init = hf.weighted_kmeans(x_arr, self.k, self.rng, w_points, centers_init = 'kmeans++')
		elif centers_init == 'uniform_subsample':	
			means_init = rng.choice(x_arr, size = self.k, replace = False)
		else:			
			raise ValueError('centers_init argument is invalid.')

		r = hf.get_point_assignments(x_arr, means_init)		
		R = np.zeros(shape = (n, self.k))
		R[np.arange(n), r] = 1

		# Optimize parameters given initial cluster responsibilities
		w_clusters, means, covs = self.maximization(x_arr, R)

		# Initialize the stopping condition value (negative log likelihood) 
		value = np.inf 
		converged = False

		for i in range(max_itr): 
			# E Step: Compute responsibilities given parameters
			R, P = self.expectation(x_arr, w_points, w_clusters, means, covs)
	
			# Compute relative change in objective (i.e. auxiliary or "Q" function)
			value_prev = value
			value = self.neg_log_likelihood(w_points, w_clusters, P)
			# value = self.auxiliary_function(R, w_points, w_clusters, P)
			if value_prev < value: 
				raise ValueError('Objective function increased, indicating error occurred.')

			# M Step: Optimize parameters given responsibilities
			w_clusters, means, covs = self.maximization(x_arr, R)

			# Consider stopping condition: absolute change in auxiliary function
			if not np.isinf(value_prev):
				if value_prev - value < tol:
					converged = True

			if converged:
				break

		if not converged:
			warnings.warn("Weighted GMM did not converge")

		return (w_clusters, means, covs)
			

	def maximization(self, x_arr, R):
		# Conducts the standard "maximzation" step for EM algorithm for GMM and returns the optimized
		# mixture weights, means, and covariances. 
		#
		# Args:
		#	x_arr: numpy array of shape (# observations, # features). 
		#	R: numpy array of shape (# observations, k); the "responsibility" matrix. 
		#	   Element r_ij is the responsibility cluster j has for observation i. Note that 
		#	   if the input data x_arr is weighted, then the point weights should be baked into 
		#	   R already. 
		#
		# Returns:
		#	3-tuple containing: 
		#		- Updated mixture weights: numpy array of shape k. 
		#		- Updated means: numpy array of shape (k, # features).
		#		- Updated covariance matrices: numpy array of shape (k, # features, # features).
		
		n = x_arr.shape[0]
		d = x_arr.shape[1]

		# Compute updated means, covariance matrices, and mixture weights via EM update rules
		w_clusters = R.sum(0)
		means = R.T.dot(x_arr) / w_clusters[:, None]

		covs = np.zeros(shape = (self.k, d, d))
		for j in range(self.k):
			covs[j] = sum([R[i, j] * np.outer(x_arr[i] - means[j], x_arr[i] - means[j]) for i in range(n)]) / w_clusters[j]	
			covs[j] = covs[j] + np.diag(np.ones(d) * self.prior_threshold)

		# Properly normalize weights
		w_clusters = w_clusters / w_clusters.sum()

		return (w_clusters, means, covs)

	
	def expectation(self, x_arr, w_points, w_clusters, means, covs):
		# Performs the standard "expectation" step for EM for GMMs. Computes the "responsibility" matrix
		# R, where r_ij is the responsibility cluster j has for observation i. The only deviation from the 
		# standard algorithm is that the responsibilities r_ij are each scaled by ther respective point weight 
		# w_points[i].  
		#
		# Args:
		#	x_arr: numpy array of shape (# observations, # features). 
		#   w_points: numpy array of shape '# observations'; the point weights.
		#	w_clusters: numpy array of shape 'k'; the mixture weights. 
		# 	means: numpy array of shape (k, # features), the cluster means. 
		# 	covs: numpy array of shape (k, # features, # features), the covariance matrices for each cluster. 
		#
		# Returns:
		# 	2-tuple, containing following elements:
		#		- numpy array of shape (# observations, k), the (weighted) responsibility matrix.  	
		#		- numpy array of shape (# observations, k), the unweighted responsibility matrix; used for 
		#		  evaluating the auxiliary function. 

		n = x_arr.shape[0]
		P = np.zeros(shape = (n, self.k))	

		# Compute multivariate normal densities
		for j in range(self.k):
			P[:, j] = multivariate_normal(means[j], covs[j]).pdf(x_arr)
			
		# Weight columns by mixture weights
		R = P * w_clusters[None, :]

		# If any rows cntain all zeros, purturb the values
		# to be a small positive number to avoid dividing by 0 below. 
		R[np.where(~R.any(axis = 1))[0], :] = 1e-16

		# Normalize rows: divide by row sums
		R = R / R.sum(1)[:, None]
		
		# Scale rows by point weights
		R = R * w_points[:, None]

		return R, P

	def neg_log_likelihood(self, w_points, w_clusters, P):
		# Computes the negative log likelihood of a GMM model, given the parameters and data. 
		# This function is defined to conveniently work with the EM algorithm loop as a backend method. For a more general 
		# function that calculates the log-liklihood, see 'evaluate_log_likelihood()'. 
		#
		# Args:
		#	w_points: numpy array of shape '# observations', the point weights. If None, uses uniform weights. 
		#	w_clusters: numpy array of shape k. The mixture weights. 
		#	P: numpy array of shape (# observations, k), where the entry p_ij := p(x_i|mean_j, cov_j), where p(.) denotes the 
		#	   multivariate normal density. 
		#
		# Returns:
		#	float, the negative of the log-likelihood function. 

		log_likelihood = np.sum([w_points[i] * np.log(np.max([w_clusters.dot(P[i, :]), 1e-16])) for i in range(len(w_points))])
	
		return -1 * log_likelihood

	def evaluate_log_likelihood(self, x_arr, w_clusters, means, covs, w_points = None):
		# Given data and the parameter values, computes the log-likelihood of the GMM model. 
		#
		# Args:
		#	x_arr: numpy array of shape (# observations, # features). 
		#	w_clusters: numpy array of shape k. The mixture weights. 
		#	means: numpy array of shape (k, # features). The Gaussian means. 
		#	covs: numpy array of shape (k, # features, # features). The Gaussian covariance matrices. 
		#	w_points: numpy array of shape '# observations', the point weights. If None, uses uniform weights. 
		# 
		# Returns:
		#	float, the log-likelihood given the input data and parameter values. 

		# If not provided, uniform weights are used.
		if w_points is None:
			w_points = np.ones(shape = x_arr.shape[0])
		w_points = np.array(w_points) / np.sum(w_points)

		log_likelihood = 0

		for i in range(x_arr.shape[0]):
			x_i_contribution = 0 
			for j in range(x_arr.shape[1]):
				x_i_contribution = x_i_contribution + (w_clusters[j] * multivariate_normal(means[j], covs[j]).pdf(x_arr[i]))
			
			log_likelihood = log_likelihood + w_points[i] + np.log(x_i_contribution)

		return log_likelihood


	def auxiliary_function(self, R, w_points, w_clusters, P):
		# Computes the standard EM for GMM auxiliary function Q(theta, theta^(t - 1)), modified so each term 
		# is weighted by point weight. Let N be the number of observations in the data. Note that this function 
		# technically returns the negative of the Q function, corresponding to the negative log-likelihood. 
		#
		# Args: 
		#	R: Numpy array of shape (N, k). r_ij = responsibility of cluster j for point i = p(z_i = j|x_i, mean_j, cov_j).
		#	w_points: Numpy array of shape N. Point weights. 
		#	w_clusters: Numpy array of shape k. Gaussian mixture weights. 
		#	P: Numpy array of shape (N, k). p_ij = likelihood point i is from cluster j = p(x_i|mean_j, cov_j).
		#
		# Returns: 
		#	float, the negative of the Q/auxiliary function. 

		return -1 * (w_points.dot(R.dot(np.log(np.maximum(w_clusters, 1e-16)))) + w_points.dot((R * np.log(np.maximum(P, 1e-16))).sum(1)))



