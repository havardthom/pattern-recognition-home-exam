import numpy as np

# k Nearest Neighbors function
###############################################################################
# INPUT
# X:		  N x d array with data
# p:	   	  Data point to find neighbors for
# k:  		  Number of neighbors to find
###############################################################################
# OUTPUT
# idx: 		  Indices of k nearest neighbors to p
# eucdst:	  Array with distances from p to all other data points
###############################################################################
def kNN(X, p, k):
 N = X.shape[0] # Number of data vectors

 # Check if k is larger or equal to the number of data points
 if k >= N:
	 k = N-1

 # Array of euclidean distances between data points in X and data point p
 eucdst = (X - p)**2
 eucdst = np.sum(eucdst, axis=1)
 eucdst = np.sqrt(eucdst)

 # Get array of vector indices sorted by smallest values in distance array
 idx = np.argsort(eucdst)

 # Return the indexes of k nearest neighbours (ignore first, since its data point p itself) and distance array
 return idx[1:k+1], eucdst
