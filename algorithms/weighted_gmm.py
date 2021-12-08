# weighted_gmm.py
# Class implementation for Gaussian Mixture Model (GMM), generlized to allow weighted input data. 
#
# Andrew Roberts

from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import helper_functions as hf
import numpy as np
import warnings

class Weighted_GMM: 

	def __init__(self, rng, k, prior_threshold): 
		self.rng = rng
		self.k = k
		self.prior_threshold = prior_threshold

	def fit(self, x_arr, w_points = None, centers_init = 'kmeans', tol = 1e-8, max_itr = 10000):

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
		covs = [cov + np.diag(np.ones(x_arr.shape[1]) * self.prior_threshold) for cov in covs]

		# Initialize auxiliary function value 
		Q = np.inf 
		converged = False

		for i in range(max_itr): 
			# E Step: Compute responsibilities given parameters
			R, P = self.expectation(x_arr, w_points, w_clusters, means, covs)
			
			# Compute relative change in objective (i.e. auxiliary or "Q" function)
			Q_prev = Q
			Q = self.auxiliary_function(R, w_points, w_clusters, P)
	
			# M Step: Optimize parameters given responsibilities
			w_clusters, means, covs = self.maximization(x_arr, R)

			# Consider stopping condition: relative change in auxiliary function
			if not np.isinf(Q_prev):
				print("(", Q_prev, ", ", Q, ", ", (Q_prev - Q) / Q_prev, ")")
				if (Q_prev - Q) / Q_prev < tol:
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
			covs[j] = np.sum([R[i, j] * np.outer(x_arr[i], means[j]) for i in range(n)]) / w_clusters[j]	
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

		# Normalize rows: divide by row sums
		R = R / R.sum(1)[:, None]
		
		# Scale rows by point weights
		R = R * w_points[:, None]

		return R, P



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

		return -1 * (w_points.dot(R.dot(np.log(w_clusters))) + w_points.dot((R * np.log(P)).sum(1)))
