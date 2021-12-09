# coreset_gmm.py 
# Implementation for class that constructs a coreset to be used for GMM estimation. 
#
# Andrew Roberts


import matplotlib.pyplot as plt
import helper_functions as hf
import numpy as np
import warnings

class Coreset_GMM:
	"""
	Class to compute a coreset (i.e. a weighted subsample) of given input data to be used for Gaussian
	Mixture Model (GMM) estimation. 
	----------
	rng: numpy random number generator object. 
	x_arr: numpy array of shape (# observations, # features). 
	k: integer, the number of Gaussians in the mixture (i.e. number of "clusters"). 
	epsilon: float, desired approximation factor in (epsilon, delta) approximation. 	
	spectrum_bound: float, eigenvalues of covariance matrices of the GMM model should be bounded in 
					[spectrum_bound, 1 / spectrum_bound]. 
	delta: float, desired probability of approximation in (epsilon, delta) approximation. 	

	----------

	References: 
	Mario Lucic, Matthew Faulkner, Andreas Krause, and Dan Feldman. "Training Gaussian Mixture 
	Models at Scale via Coresets." Journal of Machine Learning Research. 2018."

	"""

	def __init__(self, rng, x_arr, k, epsilon, spectrum_bound, delta): 
		
		# For univariate data, convert to # observations x 1 array
		if len(x_arr.shape) == 1:
			x_arr.shape = (len(x_arr), 1)

		self.rng = rng
		self.x_arr = x_arr
		self.k = k
		self.epsilon = epsilon
		self.spectrum_bound = spectrum_bound
		self.delta = delta
		self.d = self.x_arr.shape[1]

	def get_bicriteria_kmeans_approx(self, N = None):
		# Runs kmeans++ specified number of times, then returns the best set of initial values for the k means
		# in the sense that minimizes squared pairwise distances. 
		#
		# Args:
		#	N: int, the number of kmeans++ iterations to run. Defaults to the value determined by self.delta. 
		#
		# Returns:
		#	numpy array of shape (k, d), the k centers. 

		if N is None:
			N = int(np.ceil(np.log2(1 / self.delta)))
			print("Running ", N, " iterations of kmeans++")
		else:
			warnings.warn("Using manually set number of runs for kmeans initialization algorithm; theoretical bounds not guaranteed.")

		k_means_init = [hf.kmeans_pp(self.x_arr, self.k, self.rng) for i in range(N)]
		argmin = np.argmin([b[1] for b in k_means_init])

		return k_means_init[argmin]

	def calc_point_weight(self, alpha, x, x_cost, cluster_size, cluster_cost, B_cost):
		# Calculates weight for each point that determines sampling probability in coreset construction (see reference paper). 
		#
		# Args:
		#	alpha: float, approximation factor determined by generate_coreset(). 
		#	x: float, numpy array of shape d. A single input point.  
		#	x_cost: float, distance from x to B (see generate_coreset()). 
		#	cluster_size: int, the number of points in x's cluster (defined as the set of points that are closest
		#				  to b, where b is in element in b that is closest to x) (see generate_coreset()). 
		#	cluster_cost: float, within-cluster variance of x's cluster (see generate_coreset()). 
		#	B_cost: float, sum of squared distances from self.x_arr to B (see generate_coreset()). 
		#
		# Returns:
		#	float, the (unnormalized) weight to determine the sampling probability of point x for coreset generation. 

		return (alpha * x_cost) + (2 * alpha * cluster_cost / cluster_size) + (2 * B_cost / cluster_size)


	def generate_coreset(self, B = None, B_cost = None, N_bicriteria_runs = None, m = None):
		# Generates the coreset (weighted subsample) of self.x_arr, to be used in fitting a 
		# GMM model. 
		#
		# Args:
		#	B: numpy array of shape (k', d), where k' >= k. A bicriteria approximation to kmeans
		#	   run on self.x_arr. If None, get_bicriteria_kmeans_approx() will be run. 
		#	B_cost: float, the sum of squared distances from self.x_array to B. If None, 
		#			computes this quantity. 
		#	N_bicriteria_runs: If B is None, then this is passed to get_bicriteria_kmeans_approx(). 
		#	m: int, the coreset size. If None, set to 1% of the number of observations in self.x_arr. 
		#
		# Returns:
		# 	2-tuple, (the coreset) containing following elements:
		#		- numpy array of shape (m, d), the subsample of points. 	
		#		- numpy array of shape m, the weights for each point in the coreset.  

		# Set algorithm parameters		
		if B is None: 
			B, B_cost = self.get_bicriteria_kmeans_approx(N = N_bicriteria_runs)
		elif B_cost is None:
			B_cost = np.sum([hf.dist_set(x, B)**2 for x in self.x_arr])
	
		alpha = 16 * (np.log2(self.k) + 2)
		n = len(self.x_arr)
		B_len = len(B)

		# Determine coreset size
		if m is None: 
			m = int(np.ceil(.01 * n))

		print("alpha = ", alpha)
		print("m = coreset size = ", m)

		# Create partition of data into clusters based on proximity to B
		point_costs = np.zeros(shape = n)
		closest_centers = np.zeros(shape = n, dtype = np.int16)

		for i in range(n):
			dists_to_centers = [hf.dist(self.x_arr[i], b)**2 for b in B]
			closest_centers[i] = int(np.argmin(dists_to_centers))
			point_costs[i] = dists_to_centers[closest_centers[i]]

		# closest_centers = np.array([np.argmin([hf.dist(x, b) for b in B]) for x in self.x_arr])
		B_cluster_sizes = [int(np.sum(closest_centers == j)) for j in range(B_len)]
		B_cluster_costs = [hf.cluster_cost(self.x_arr[closest_centers == j], B) for j in range(B_len)]

		# Weight points and sample based on weighting to create coreset	
		sampling_weights = np.array([self.calc_point_weight(alpha, self.x_arr[i], point_costs[i], 
									                   	    B_cluster_sizes[closest_centers[i]], 
													   		B_cluster_costs[closest_centers[i]], B_cost) for i in range(n)])	
		sampling_weights = sampling_weights / np.sum(sampling_weights)

		# Calculate weights for each point in coreset	
		coreset_selector = self.rng.choice(np.arange(self.x_arr.shape[0]), size = m, p = sampling_weights)
		coreset_weights = np.array([1 / (sampling_weights[i] * m) for i in coreset_selector])
	
		return(self.x_arr[coreset_selector], coreset_weights)
		
		
		





