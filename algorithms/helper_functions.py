# helper_functions.py
# Assorted helper functions useful in running clustering analyses, including distance metrics and
# kmeans algorithms.
#
# Andrew Roberts

import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------------------
# Distance Metric Functions
# ------------------------------------------------------------------------

def dist_manhattan(x, y):
	# Return Manhattan (l1)  distance between 2 1D arrays.
	#
	# Args:
	#	x: 1d numpy array.
	#	y: 1d numpy array of same shape as x.
	#
	# Returns:
	#	||x - y||_1, the Manhattan distance between the vectors.
	return(np.linalg.norm(x - y), 1)

def dist(x, y):
	# Return Euclidean (l2)  distance between 2 1D arrays.
	#
	# Args:
	#	x: 1d numpy array.
	#	y: 1d numpy array of same shape as x.
	#
	# Returns:
	#	||x - y||_2, the Euclidean distance between the vectors.

	return(np.linalg.norm(x - y))

def dist_set_manhattan(x, Y):
	# Return l1 distance between a point (array) and set (of arrays).
	#
	# Args:
	#	x: 1d numpy array of length '# features' (the "point").
	#	Y: numpy array of length (# points in set, # features) (the "set").
	#
	# Returns:
	#	d(x, Y) = min_{y in Y} ||x - y||_1; i.e. the standard Manhattan distance between
	# 	a point and a set. Note that the lower case 'y' values in the distance expression
	# 	are the rows of the numpy array Y.

	return min([dist_manhattan(x, y) for y in Y])

def dist_set(x, Y):
	# Return l2 distance between a point (array) and set (of arrays).
	#
	# Args:
	#	x: 1d numpy array of length '# features' (the "point").
	#	Y: numpy array of length (# points in set, # features) (the "set").
	#
	# Returns:
	#	d(x, Y) = min_{y in Y} ||x - y||_2; i.e. the standard Euclidean distance between
	# 	a point and a set. Note that the lower case 'y' values in the distance expression
	# 	are the rows of the numpy array Y.

	return min([dist(x, y) for y in Y])

def cluster_cost_manhattan(x_arr, B):
	# Computes sum of distances between a set of "centroids" B and a set of points x_arr.
	#
	# Args:
	#	x_arr: numpy array of shape (# observations, # features).
	#	B: numpy array of shape (# centroids, # features).
	#
	# Returns:
	#	float, the sum of the squared Euclidean distance between each point and its closest centroid
	#	(summed over all observations).

	return np.sum([dist_set_manhattan(x, B) for x in x_arr])

def cluster_cost(x_arr, B):
	# Computes sum of squared distances between a set of "centroids" B and a set of points x_arr.
	#
	# Args:
	#	x_arr: numpy array of shape (# observations, # features).
	#	B: numpy array of shape (# centroids, # features).
	#
	# Returns:
	#	float, the sum of the squared Euclidean distance between each point and its closest centroid
	#	(summed over all observations).

	return np.sum([dist_set(x, B)**2 for x in x_arr])


def max_distance(x_arr, C):
	# Returns the farthest point in x_arr from C
	#
	# Args:
	#	x_arr: array of points in R^d
	#	C: Array of centers chosen in R^d
	#
	# Returns:
	#	Point (array) farthest from C

	max_dist = 0
	max_point = None

	for p in x_arr:

		total_center_distance = 0
		for c in C:

			distance_val_c = dist(p,c)
			total_center_distance += distance_val_c

		if total_center_distance > max_dist:
			max_dist = total_center_distance
			max_point = p

	#print(max_dist, max_point)
	return max_point


# ------------------------------------------------------------------------
# k-means Functions
# ------------------------------------------------------------------------

def get_point_assignments(x_arr, centers):
	# Given data set and a set of "centers", return a vector that assigns each point to its
	# closest center (in Euclidean/l2 distance).
	#
	# Args:
	#	x_arr: numpy array of shape (# observations, # features).
	#   centers: numpy array of shape (# centers, # features).
	#
	# Returns:
	#	numpy array of shape '# observations'. The ith entry of this array is an integer
	#	in {0, ..., # centers - 1}, indicating the index of 'centers' containing the center
	#	closest to point i in Euclidean distance.

	return np.array([int(np.argmin([dist(x, c) for c in centers])) for x in x_arr], dtype = np.int16)




def kmeans_pp(x_arr, k, rng):
	# Implementation of k-Means++ algorithm, which initializes the k means to be used
	# in naive k-Means.
	#
	# Args:
	#	x_arr: numpy array of shape (# observations, # features).
	#	k: int, number of clusters.
	#	rng: numpy random number generator object.
	#
	# Returns:
	#	tuple, containing following elements:
	#		- numpy array of shape (k, # features), the k means returned by the algorithm.
	#		- float, the "cost" of the clustering; i.e. the sum of squared errors to the k centroids.

	# For univariate data, convert to # observations x 1 array
	if len(x_arr.shape) == 1:
		x_arr.shape = (len(x_arr), 1)

	B = np.zeros(shape = (k, x_arr.shape[1]))
	B[0] = rng.choice(x_arr)

	for j in range(1, k):
		dists_to_B = [dist_set(x, B[:j])**2 for x in x_arr]
		probs = dists_to_B / np.sum(dists_to_B)
		B[j] = rng.choice(x_arr, p = probs)

	return (B, cluster_cost(x_arr, B))


def weighted_kmeans(x_arr, k, rng, w = None, centers_init = None, tol = 1e-4):
	# Standard K-Means Algorithm with option to use non-uniform point weights.
	#
	# Args:
	#	x_arr: numpy array of shape (# observations, # features).
	#	k: int, number of clusters.
	#	rng: numpy random number generator object.
	#	w: numpy array of shape '# observations'. The point weights. If None, uses standard uniform weights.
	#	centers_init: If numpy array of shape (k, # features) then interpreted as the k centroids used to
	#				  initialize the algorihth. If 'kmeans++' then initializes the centers via the k-means++
	#				  algorithm. If None, uniformly generates the k centroids.
	#	tol: float, stopping condition. Algorithm terminates when relative change in centers is smaller than
	#		 'tol' (in 2-norm/Frobenius norm).
	#
	# Returns:
	#	numpy array of shape (k, # features), the k centroids returned by the algorithm.

	n = len(x_arr)

	# For univariate data, convert to # observations x 1 array
	if len(x_arr.shape) == 1:
		x_arr.shape = (len(x_arr), 1)

	# Determine points weights
	if w is None:
		w = np.repeat([1 / n], [n])
	else:
		w = np.array(w) / np.sum(w)

	# Initialize k centers
	if centers_init is None:
		centers = rng.choice(x_arr, size = k, p = w)
	elif centers_init == 'kmeans++':
		centers = kmeans_pp(x_arr, k, rng)[0]

	while True:
		# Assign points to clusters
		point_assignments = get_point_assignments(x_arr, centers)

		# Compute new centers
		centers_prev = np.array(centers)
		centers = np.array([np.average(x_arr[point_assignments == j], axis = 0, weights = w[point_assignments == j]) for j in range(k)])

		# Stop when relative change in clusters is small (in Frobenius norm)
		if np.linalg.norm(centers - centers_prev) / np.linalg.norm(centers_prev) < tol:
			print("weighted_kmeans() converged")
			break

	return centers


# ------------------------------------------------------------------------
# Data Simulation Functions
# ------------------------------------------------------------------------


def simulate_gmm_data(rng, n, k, means, covs, weights):
	# Returns a random sample of size n from a mixture of k multivariate normal
	# distributions in d-dimensional space (where d is implied by the dimension of the elements of
	# 'means' and 'covs')  with mean vectors given in 'mean', covariance matrices given
	# in 'cov', and mixture weights given in 'weights'.
	#
	# Args:
	#	rng: numpy random number generator object.
	#	n: int, the number of samples.
	#	k: int, the number of Gaussians in the mixture.
	#	means: numpy ndarray of shape (k, d), where 'mean[i]' gives the mean vector of the ith Gaussian.
	#	covs: numpy ndarray of shape (k, d, d), where 'cov[i]' gives the covariance matrix of the ith Gaussian.
	#	weights: list or numpy ndarray of shape k, where 'weights[i]' gives the mixture weight of the ith Gaussian.
	#
	# Returns:
	#	numpy ndarray of shape (n, d), where each row is a d-dimensional point sampled from the GMM.

	weights = weights / np.sum(weights)
	mixture_samples = rng.choice(k, size = n, p = weights)
	gmm_samples = np.array([rng.multivariate_normal(means[j], covs[j]) for j in mixture_samples])

	return(gmm_samples)

def simulate_gaussian_clusters(rng, n, k, means, covs):
	# Returns a random sample of size (at most) n comprised of points sampled from k different multivariate normal
	# distributions.  The dimension of the points is implied by the dimension of the mean vectors and covariance matrices.
	# Denote this dimension by d. Note that this differs from 'simulate_gmm_data()' only in that the user can specify
	# exactly how many points will be in each cluster (as opposed to probabalistically doing so via setting the mixture
	# weights). This can be useful for testing purposes.
	#
	# Args:
	#	rng: numpy random number generator object.
	#	n: list or int. If list, ith element contains number of samples to be drawn from the ith Gaussian. If int,
	#	   floor(n/k) samples will be drawn from each Gaussian.
	#	k: int, the number of Gaussians.
	#	means: numpy ndarray of shape (k, d), where 'mean[i]' gives the mean vector of the ith Gaussian.
	#	covs: numpy ndarray of shape (k, d, d), where 'cov[i]' gives the covariance matrix of the ith Gaussian.
	#
	# Returns:
	#	numpy ndarray of shape (m, d), where m <= n. Each row is a d-dimensional point sampled from one of the k
	#	Gaussians.

	# n interpreted as total number of samples; equally weight clusters
	if len(n) == 1:
		n_cluster = int(np.floor(n / k))
		n = [int(n_cluster for i in range(k))]

	samples = np.concatenate([rng.multivariate_normal(means[j], covs[j], size = n[j]) for j in range(k)])

	return samples


# ------------------------------------------------------------------------
# Plotting Functions
# ------------------------------------------------------------------------


def scatter_2D(x_arr_list, ax = None, title = None):
	# Generate scatter plot of points if 'x_arr' has dimension 1 or 2.
	#
	# Args:
	#	x_arr_list: list of numpy arrays of shape (# observations_i, # features_i), where '# features' <= 2.
	#				The subscript 'i' indicates that these quantities can differ for each element in the list.
	#				Each element is treated as a different set of data to be plotted on the same plot.
	#	ax: pyplot axis object. If None, creates new axis, otherwise appends
	#		to existing axis.
	#
	# Returns:
	#	pyplot axis containing the plot.

	for x_arr in x_arr_list:
		d = x_arr.shape[1]

		if d > 2:
			warnings.warn("Cannot plot data of dimension > 2")
		else:
			n = x_arr.shape[0]
			x = [x_arr[i][0] for i in range(n)]

			if d == 1:
				y = np.zeros(shape = n)
			else:
				y = [x_arr[i][1] for i in range(n)]

			if ax is None:
				f, ax = plt.subplots()
			ax.scatter(x, y)

	# Add title
	if title is not None:
		plt.title(title)

	return ax
