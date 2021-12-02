
class Coreset_GMM:
	def __init__(self, rng, x_arr, k, epsilon, spectrum_bound, delta): 
		self.rng = rng
		self.x_array = x_arr
		self.k = k
		self.epsilon = epsilon
		self.spectrum_bound = spectrum_bound
		self.delta = delta

	# Return l2 distance between 2 1D arrays.	
	def d(x, y): 
		return(np.sqrt(np.sum((x - y)**2)))

	# Return l2 distance between a point (array) and set (of arrays)
	# Note: If Y is a 2D ndarray, the rows of Y will be considered the "points". 
	def d_set(self, x, Y):
		return min([self.d(x, y) for y in Y])
		

	# Implementation of k-Means++ algorithm, which initializes the k means to be used
	# in naive k-Means. 
	def kmeans_pp(self):
		B = np.zeros(shape = self.k)
		B[0] = self.rng.choice(self.x_array)

		for j in range(1, self.k):
			dists_to_B = [self.d_set(x, B[:j])**2 for x in self.x_arr]
			probs = dists_to_b / np.sum(dists_to_B)
			B[j] = self.rng.choice(self.x_array, p = probs)
			
		return B

				
			
		
		
