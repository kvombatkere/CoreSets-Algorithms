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
		return(np.sqrt(np.sum((x - y)**2)))

	# Return l2 distance between a point (array) and set (of arrays)
	# Note: If Y is a 2D ndarray, the rows of Y will be considered the "points". 
	def dist_set(self, x, Y):
		return min([self.dist(x, y) for y in Y])

	# Plot points if x_array has dimension 1 or 2
	# arr = None means default to plotting self.x_array
	def scatter_2D(self, arr = None, ax = None):
		if self.d > 2: 
			warnings.warn('Cannot plot data of dimension > 2')
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
			
		
	# Implementation of k-Means++ algorithm, which initializes the k means to be used
	# in naive k-Means. 
	def kmeans_pp(self):
		B = np.zeros(shape = (self.k, self.d))
		B[0] = self.rng.choice(self.x_array)

		for j in range(1, self.k):
			dists_to_B = [self.dist_set(x, B[:j])**2 for x in self.x_array]
			probs = dists_to_B / np.sum(dists_to_B)
			B[j] = self.rng.choice(self.x_array, p = probs)
			
		return B

				
			
		
		
