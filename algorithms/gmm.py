
class Coreset_GMM:
	def __init__(self, x_arr, k, epsilon, spectrum_bound, delta): 
		self.x_array = x_arr
		self.k = k
		self.epsilon = epsilon
		self.spectrum_bound = spectrum_bound
		self.delta = delta

	# Implementation of k-Means++ algorithm, which initializes the k means 
	def kmeans_pp(self):
		
