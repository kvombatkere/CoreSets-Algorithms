import matplotlib.pyplot as plt
import helper_functions as hf
import numpy as np
import warnings

class Coreset_GMM:
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

	# Runs kmeans++ specified number of times, then returns the best set of initial values for the k means
	# in the sense that minimizes squared pairwise distances. 
	def get_bicriteria_kmeans_approx(self, N = None):
		if N is None:
			N = int(np.ceil(np.log2(1 / self.delta)))
			print("Running ", N, " iterations of kmeans++")
		else:
			warnings.warn("Using manually set number of runs for kmeans initialization algorithm; theoretical bounds not guaranteed.")

		k_means_init = [hf.kmeans_pp(self.x_arr, self.k, self.rng) for i in range(N)]
		argmin = np.argmin([b[1] for b in k_means_init])

		return k_means_init[argmin]

	# Calculates weight for each point that determines sampling probability in coreset construction
	def calc_point_weight(self, alpha, x, x_cost, cluster_size, cluster_cost, B_cost):
		return alpha * x_cost + 2 * alpha * cluster_cost / cluster_size + 2 * B_cost / cluster_size


	def generate_coreset(self, B = None, B_cost = None, N_bicriteria_runs = None, m = None):

		# Set algorithm parameters		
		if B is None: 
			B, B_cost = self.get_bicriteria_kmeans_approx(N = N_bicriteria_runs)
		elif B_cost is None:
			B_cost = np.sum([hf.dist_set(x, B)**2 for x in self.x_arr])

		# TODO: which constant to use? 
		if m is None: 
			m = .01 * (self.d**4 * self.k**6 + self.k**2 * np.log(1/self.delta)) / (self.spectrum_bound**4 * self.epsilon**2)
		else:
			warnings.warn("Using manually set coreset size; theoretical bounds not guaranteed.")
			
		alpha = 16 * (np.log2(self.k) + 2)
		n = len(self.x_arr)
		B_len = len(B)

		print("alpha = ", alpha)
		print("m = coreset size = ", m)

		# Create partition of data into clusters based on proximity to B
		point_costs = np.zeros(shape = n)
		closest_centers = np.zeros(shape = n, dtype = np.int16)

		for i in range(n):
			dists_to_centers = [hf.dist(self.x_arr[i], b)**2 for b in B]
			closest_centers[i] = int(np.argmin(dists_to_centers))
			point_costs[i] = dists_to_centers[closest_centers[i]]

		closest_centers = np.array([np.argmin([hf.dist(x, b) for b in B]) for x in self.x_arr])
		B_cluster_sizes = [int(np.sum(closest_centers == j)) for j in range(B_len)]
		B_cluster_costs = [hf.cluster_cost(self.x_arr[self.x_arr == j], B) for j in range(B_len)]

		# Weight points and sample based on weighting to create coreset	
		sampling_weights = np.array([self.calc_point_weight(alpha, self.x_arr[i], point_costs[i], 
									                   	    B_cluster_sizes[closest_centers[i]], 
													   		B_cluster_costs[closest_centers[i]], B_cost) for i in range(n)])	
		sampling_weights = sampling_weights / np.sum(sampling_weights)

		# Calculate weights for each point in coreset	
		coreset_selector = self.rng.choice(np.arange(self.x_arr.shape[0]), size = m, p = sampling_weights)
		coreset_weights = np.array([1 / sampling_weights[i] for i in coreset_selector])
	
		return(self.x_arr[coreset_selector], coreset_weights)
		
		
		





