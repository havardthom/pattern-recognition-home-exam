import numpy as np
from kNN import kNN

# Laplacian Eigenmaps function
###############################################################################
# INPUT
# X:		  N x d array with data
# ndim:	      Dimensions to reduce too (default=2)
# k:  		  Number of neighbors to use in kNN algorithm (default=4)
###############################################################################
# OUTPUT
# transformed: Reduced N x ndim array
###############################################################################
def LEM(X, ndim=2, k=4):
	N = X.shape[0] # Number of data vectors
	d = X.shape[1] # Number of dimensions

	# Check if ndim is larger or equal to current dimensions
	if ndim >= d:
		ndim = d - 1

	# Create adjacencey matrix with weights
	W = np.zeros((N, N))
	for i in range(N):
		 idx, eucdst = kNN(X[i], X, k) # Get k nearest neighbours and distance array
		 for j in range(k):
			 # Weight with heat e**(||x1-x2||**2/t) where t = 200
			 heat = np.exp(-eucdst[idx[j]]**2/200)
			 W[i, idx[j]] = heat
			 W[idx[j], i] = heat
			# #  Alternative: weight with {0,1}
			#  W[i, idx[k]] = 1
			#  W[idx[k], i] = 1

	# Create diagonal weight matrix (with column sums of W)
	D = np.diag(W.sum(axis=1))

	# Create laplacian matrix
	L = D - W

	# Get eigenvalues and eigenvectors of laplacian matrix (use linalg.eigh since L is symmetric)
	eigval, eigvec = np.linalg.eigh(L)
	eigval = np.real(eigval)
	eigvec = np.real(eigvec)

	# Get array of eigenvalue indices sorted by smallest values in eigenvalue array
	index = eigval.argsort()

	# Return embedded matrix (ignore first eigenvector since its constant)
	transformed = eigvec[:,index[1:ndim+1]]
	return transformed
