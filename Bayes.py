import numpy as np

# Gaussian Naive Bayes classifier
class Gaussian_Naive_Bayes_classifier:
# Training function
###############################################################################
# INPUT
# X:		  N x d array with training data
# Y:	   	  N x 1 array with training data labels
###############################################################################
	def train(self, X, y):
		# Split training data in classes
		class1 = X[np.where(y == 1)[0]]
		class2 = X[np.where(y == -1)[0]]

		# Get prior probability for each class
		self.prior1 = np.mean(y == 1)
		self.prior2 = np.mean(y == -1)

		# Get mean array for each class containing the mean of each attribute
		self.theta1 = np.mean(class1, axis=0)
		self.theta2 = np.mean(class2, axis=0)

		# Get variance array for each class containing the variance of each attribute
		self.sigma1 = np.var(class1, axis=0)
		self.sigma2 = np.var(class2, axis=0)

# Classify function
###############################################################################
# INPUT
# X:		  N x d array with test data
###############################################################################
# OUTPUT
# Y:		  N x 1 array with predicted data labels
###############################################################################
	def classify(self, X):
		# Logarithmic likelihood: ln( p(X|class) )

		# Get likelihood that data points x in X is put in class 1
		gaus_pdf_1 = -0.5 * np.sum(np.log(2 * np.pi * self.sigma1))
		gaus_pdf_1 -= 0.5 * np.sum(((X - self.theta1)**2) / (self.sigma1), axis=1)
		prior1 = np.log(self.prior1)

		# Get likelihood that data points x in X is put in class 2
		gaus_pdf_2 = -0.5 * np.sum(np.log(2 * np.pi * self.sigma2))
		gaus_pdf_2 -= 0.5 * np.sum(((X - self.theta2)**2) / (self.sigma2), axis=1)
		prior2 = np.log(self.prior2)

		# g(X) = ln( p(X|class) ) + ln( P(class) )
		prob_est1 = gaus_pdf_1 + prior1
		prob_est2 = gaus_pdf_2 + prior2

		# Stack the two arrays next to eachother vertically
		joint_prob = np.vstack((prob_est1, prob_est2))

		# Get an array of class indices with highest likelihood
		Y = np.argmax(joint_prob, axis=0)

		# Fix class labels
		Y[Y == 1] = -1
		Y[Y == 0] = 1
		Y = Y[np.newaxis]
		# Return predicted labels
		return Y.T
