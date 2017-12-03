import numpy as np

# Principal component analysis function
###############################################################################
# INPUT
# X:		  N x d array with data
# ndim:	      Dimensions to reduce too (default=2)
###############################################################################
# OUTPUT
# transformed: Reduced N x ndim array
###############################################################################
def PCA(X, ndim=2):
	N = X.shape[0] # Number of data vectors
	d = X.shape[1] # Number of dimensions

	# Check if ndim is larger than current dimensions
	if ndim > d:
		ndim = d

	# Create mean vector containing the mean of each dimension in X
	mean_vector = np.mean(X, axis=0)

	# Calculate deviations from mean
	B = X - mean_vector

	# Find covariance matrix
	Cov = np.dot(B.T, B)/(N - 1)
	# Cov = np.matrix(np.cov(XT)) # Alternative way

	# Get eigenvalues and eigenvectors of the covariance matrix
	eigval, eigvec = np.linalg.eig(Cov)
	eigval = np.real(eigval)
	eigvec = np.real(eigvec)

	# Get array of eigenvalue indices sorted by smallest values in eigenvalue array
	index = eigval.argsort()
	# Reverse indices to get largest first
	index = index[::-1]

	# Create subset of ndim eigenvectors corresponding to ndim largest eigenvalues
	W = eigvec[:,index[:ndim]]

	# Return reduced array
	transformed = np.dot(X, W)
	return transformed
