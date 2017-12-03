import numpy as np

# Activation function for 2LP
def tanh(x):
	return np.tanh(x)

# Derivative of activation function
def tanh_deriv(x):
	return 1.0 - np.tanh(x)**2

# Two Layer Perceptron classifier
class Two_Layer_Perceptron_Classifier:
# Initialize
###############################################################################
# INPUT
# size_h:	   	  Units in hidden layer (default=2)
# alpha:		  Momentum factor (default=0.1)
# learning_rate:  Learning rate (default=0.1)
# epochs:	   	  Max epochs when training (default=5000)
# error_rate:	  Accepted error rate to stop training (default=0.01)
# ###############################################################################
	def __init__(self, size_h=2, alpha=0.1, learning_rate=0.1, epochs=5000, error_rate=0.01):
		self.size_h = size_h
		self.alpha = alpha
		self.learning_rate = learning_rate
		self.epochs = epochs
		self.error_rate = error_rate

# Training function
###############################################################################
# INPUT
# X:		  N x d array with training data
# Y:	   	  N x 1 array with training data labels
# ###############################################################################
	def train(self, X, y):
		N = X.shape[0]
		# Add column for bias
		X = np.c_[X, np.ones(N)]
		d = X.shape[1]

		# Initialize hidden layer weight array
		self.hl_weight = np.random.uniform(size=(d, self.size_h))
		# Initialize hidden layer output array
		self.hl_output = np.empty((N, self.size_h))

		# Initialize output layer weight array
		self.ol_weight = np.random.uniform(size=(self.size_h, y.shape[1]))
		# Initialize output layer output array
		self.ol_output = np.empty_like(y)

		# Initialize momentum to zero
		self.hl_momentum = 0
		self.ol_momentum = 0

		# Start backpropagation
		for i in range(self.epochs):
			self.propagate_forward(X)
			self.propagate_backward(X, y)
			self.update_weights(X)
			# Stop training if accepted error rate is reached
			if self.sqe <= self.error_rate:
				break

	# Backpropagation functions
	def propagate_forward(self, X):
		# Compute output vector of hidden layer
		self.hl_output = tanh(np.dot(X, self.hl_weight))

		# Compute output vector of output layer
		self.ol_output = tanh(np.dot(self.hl_output, self.ol_weight))

	def propagate_backward(self, X, y):
		# Compute delta for output layer
		self.ol_delta = (y - self.ol_output) * tanh_deriv(self.ol_output)

		# Compute delta for hidden layer
		self.hl_delta = np.dot(self.ol_delta, self.ol_weight.T) * tanh_deriv(self.hl_output)

		# Sum of squared errors
		self.sqe = (0.5*(y - self.ol_output)**2).sum()
		print "Error rate: " + str(self.sqe)

	def update_weights(self, X):
		# Calculate the hidden layer gradient
		hl_Delta_w = self.learning_rate * np.dot(X.T, self.hl_delta) + (self.alpha * self.hl_momentum)
		# Update hidden layer weight
		self.hl_weight += hl_Delta_w
		# Set hidden layer momentum to gradient
		self.hl_momentum = hl_Delta_w

		# Caculate the output layer gradient
		ol_Delta_w = self.learning_rate * np.dot(self.hl_output.T, self.ol_delta) + (self.alpha * self.ol_momentum)
		# Update output layer weight
		self.ol_weight += ol_Delta_w
		# Set output layer momentum to gradient
		self.ol_momentum = ol_Delta_w

# Classify function
###############################################################################
# INPUT
# X:		  N x d array with test data
###############################################################################
# OUTPUT
# Y:		  N x 1 array with predicted data labels
###############################################################################
	def classify(self, X):
		N = X.shape[0]
		# Add column for bias
		X = np.c_[X, np.ones(N)]
		# Get predicted labels
		self.propagate_forward(X)
		return np.sign(self.ol_output)


# Perceptron classifier
class Perceptron_Classifier:
# Initialize
###############################################################################
# INPUT
# learning_rate:  Learning rate (default=0.1)
# n_iter:	   	  Max iterations when training (default=5000)
# ###############################################################################
	def __init__(self, learning_rate=0.1, n_iter=5000):
		self.learning_rate = learning_rate
		self.n_iter = n_iter
# Training function
###############################################################################
# INPUT
# X:		  N x d array with training data
# Y:	   	  N x 1 array with training data labels
###############################################################################
	def train(self, X, y):
		N = X.shape[0]
		# Add column for bias
		X = np.c_[X, np.ones(N)]
		d = X.shape[1]
		# Initalize random weight vector
		self.w = np.random.uniform(size=(d, 1))
		# Start iteration count
		t = 0
		while True:
			# Start error count
			error = 0
			# Create new gradient with zeros
			gradient = np.zeros((1,d))
			# Check for misclassified points in X
			for i in range(N):
				if (-y[i])*np.dot(X[i], self.w) >= 0:
					# If misclassified update error count and gradient
					error += 1
					gradient = gradient + self.learning_rate*(X[i]*-y[i])
			# Update weight vector
			self.w = self.w - (self.learning_rate*gradient.T)

			print "Missclassified: " + str(error)
			# Update iteration count
			t += 1
			# Stop training if there are 0 missclassified or iteration limit is reached
			if error == 0 or t >= self.n_iter:
				break


# Classify function
###############################################################################
# INPUT
# X:		  N x d array with test data
###############################################################################
# OUTPUT
# Y:		  N x 1 array with predicted data labels
###############################################################################
	def classify(self, X):
		N = X.shape[0]
		# Add column for bias
		X = np.c_[X, np.ones(N)]
		# Get predicted labels
		Y = np.sign(np.dot(X, self.w))
		return Y
