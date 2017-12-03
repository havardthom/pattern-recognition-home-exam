import numpy as np

# Least Square Sum classifier
class Least_Square_Classifier:
# Training function
###############################################################################
# INPUT
# X:		  N x d array with training data
# Y:	   	  N x 1 array with training data labels
###############################################################################
	def train(self, X, y):
		# Add column with ones to X for bias
		X = np.c_[X, np.ones(len(X))]
		# Get pseudoinverse of X
		PinvX = np.linalg.pinv(X)
		# Find optimal weight vector by solving equation w = PinvX*y
		self.w = np.dot(PinvX, y)


# Classify function
###############################################################################
# INPUT
# X:		  N x d array with test data
###############################################################################
# OUTPUT
# Y:		  N x 1 array with predicted data labels
###############################################################################
	def classify(self, X):
		# Add column with ones to X for bias
		X = np.c_[X, np.ones(len(X))]
		# Get predicted labels
		Y = np.sign(np.dot(X, self.w))
		return Y
