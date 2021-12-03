import matplotlib.pyplot as plt
import numpy as np
import warnings

class Coreset_GMM:
	def __init__(self, rng, x_arr, k, epsilon, spectrum_bound, delta): 
		self.rng = rng
		self.x_array = x_arr
		self.k = k
		self.epsilon = epsilon
		self.spectrum_bound = spectrum_bound
		self.delta = delta
		self.d = self.x_array.shape[1]

	# Return l2 distance between 2 1D arrays.	
	def dist(self, x, y): 
		return(np.linalg.norm(x - y))

	# Return l2 distance between a point (array) and set (of arrays)
	# Note: If Y is a 2D ndarray, the rows of Y will be considered the "points". 
	def dist_set(self, x, Y):
		return min([self.dist(x, y) for y in Y])

	# Plot points if x_array has dimension 1 or 2
	# arr = None means default to plotting self.x_array.
	# ax = None means create new axis, otherwise adds to existing one
	def scatter_2D(self, arr = None, ax = None):
		if self.d > 2: 
			warnings.warn("Cannot plot data of dimension > 2")
		else: 	
			if arr is None:
				arr = self.x_array

			n = len(arr)
			x = [arr[i][0] for i in range(n)]
			
			if self.d == 1:
				y = np.zeros(shape = n)
			else:
				y = [arr[i][1] for i in range(n)]
	
			if ax is None:
				f, ax = plt.subplots()	
			ax.scatter(x, y)

			return ax
		
	# Computes sum of squared distances between a set of "centers" B and a set of points X. 
	# If X is None, then X := self.x_array
	def cluster_cost(self, B, X = None):
		if X is None:
			X = self.x_array

		return np.sum([self.dist_set(x, B)**2 for x in X])
	
		
	# Implementation of k-Means++ algorithm, which initializes the k means to be used
	# in naive k-Means. 
	def kmeans_pp(self):
		B = np.zeros(shape = (self.k, self.d))
		B[0] = self.rng.choice(self.x_array)

		for j in range(1, self.k):
			dists_to_B = [self.dist_set(x, B[:j])**2 for x in self.x_array]
			probs = dists_to_B / np.sum(dists_to_B)
			B[j] = self.rng.choice(self.x_array, p = probs)
			
		return (B, self.cluster_cost(B))

	# Runs kmeans++ specified number of times, then returns the best set of initial values for the k means
	# in the sense that minimizes squared pairwise distances. 
	def get_bicriteria_kmeans_approx(self, N = None):
		if N is None:
			N = int(np.ceil(np.log2(1 / self.delta)))
			print("Running ", N, " iterations of kmeans++")
		else:
			warnings.warn("Using manually set number of runs for kmeans initialization algorithm; theoretical bounds not guaranteed.")

		k_means_init = [self.kmeans_pp() for i in range(N)]
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
			B_cost = np.sum([self.dist_set(x, B)**2 for x in self.x_array])

		# TODO: which constant to use? 
		if m is None: 
			m = .01 * (self.d**4 * self.k**6 + self.k**2 * np.log(1/self.delta)) / (self.spectrum_bound**4 * self.epsilon**2)
		else:
			warnings.warn("Using manually set coreset size; theoretical bounds not guaranteed.")
			
		alpha = 16 * (np.log2(self.k) + 2)
		n = len(self.x_array)
		B_len = len(B)

		print("alpha = ", alpha)
		print("m = coreset size = ", m)

		# Create partition of data into clusters based on proximity to B
		point_costs = np.zeros(shape = n)
		closest_centers = np.zeros(shape = n, dtype = np.int16)

		for i in range(n):
			dists_to_centers = [self.dist(self.x_array[i], b)**2 for b in B]
			closest_centers[i] = int(np.argmin(dists_to_centers))
			point_costs[i] = dists_to_centers[closest_centers[i]]

		closest_centers = np.array([np.argmin([self.dist(x, b) for b in B]) for x in self.x_array])
		B_cluster_sizes = [int(np.sum(closest_centers == j)) for j in range(B_len)]
		B_cluster_costs = [self.cluster_cost(B, self.x_array[self.x_array == j]) for j in range(B_len)]

		# Weight points and sample based on weighting to create coreset	
		sampling_weights = np.array([self.calc_point_weight(alpha, self.x_array[i], point_costs[i], 
									                   	    B_cluster_sizes[closest_centers[i]], 
													   		B_cluster_costs[closest_centers[i]], B_cost) for i in range(n)])	
		sampling_weights = sampling_weights / np.sum(sampling_weights)

		# Calculate weights for each point in coreset	
		coreset = self.rng.choice(self.x_array, size = m, p = sampling_weights)
		coreset_weights = np.array([1 / x for x in sampling_weights]) / m	
			
		return(coreset, coreset_weights)
		
