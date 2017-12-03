import scipy.io as sio
import numpy as np
import time

from Bayes import Gaussian_Naive_Bayes_classifier
from LS import Least_Square_Classifier
from Perceptron import Perceptron_Classifier, Two_Layer_Perceptron_Classifier
from PCA import PCA
from LEM import LEM
from sklearn.svm import LinearSVC

from ASCII import ASCII

use_small = True
use_large = False

# Timer functions to measure PCA and LEM performance
def start_time():
    global TICK
    TICK = time.time()

def stop_time(prefix):
    global TICK
    old = TICK
    TICK = time.time()
    print(prefix + " used " + str(TICK-old) + " seconds")

class BinaryRecognitionSystem:
	def __init__(self):
		digitdata = sio.loadmat('data/Xtr_digits.mat')
		self.Xtr_digits = digitdata['Xtr_digits'].T

		digitdata = sio.loadmat('data/ytr_digits.mat')
		self.ytr_digits = digitdata['ytr_digits'].T

		digitdata = sio.loadmat('data/Xtr_digits_larger.mat')
		self.Xtr_digits_larger = digitdata['Xtr_digits_larger'].T

		digitdata = sio.loadmat('data/ytr_digits_larger.mat')
		self.ytr_digits_larger = digitdata['ytr_digits_larger'].T

		digitdata = sio.loadmat('data/Xte_digits_2015.mat')
		self.Xte_digits_2015 = digitdata['Xte_digits_2015'].T

		# Solution from using 2LP
		digitdata = sio.loadmat('data/yte_digits_2015.mat')
		self.solution = digitdata['yte_digits_2015']

		self.classifier = Least_Square_Classifier()

if __name__ == "__main__":
	sys = BinaryRecognitionSystem()

	# # Testing with linear Support Vector Machine from sklearn
	# svm = LinearSVC()
	# sys.Xtr_digits = LEM(sys.Xtr_digits)
	# sys.Xtr_digits_larger = LEM(sys.Xtr_digits_larger)
	# sys.Xte_digits_2015 = LEM(sys.Xte_digits_2015)
	# svm.fit(sys.Xtr_digits, sys.ytr_digits.T[0])
	# result = svm.predict(sys.Xte_digits_2015)
	# print("Number of misclassified songs out of a total %d songs: %d" % (sys.Xte_digits_2015.shape[0],(result[np.newaxis].T != sys.solution).sum()))

	print 'Welcome to Binary digit recognition system! (type help for commands)'
	while True:
			userinput = raw_input('Command: ').split()
			try:
				if userinput[0] == "help":
					fmt = ' {0:25} # {1:}'
					print fmt.format("decode", "Decode test data with chosen classifier and DTDR method")
					print fmt.format("LS", "Use Least Sum of Squares classifier (default)")
					print fmt.format("2LP ", "Use Two-Layer Perceptron classifier (NOTE: parameters must be changed manually)")
					print fmt.format("GNB", "Use Gaussian Naive Bayes classifier")
					print fmt.format("P", "Use Perceptron classifier (NOTE: parameters must be changed manually)")
					print fmt.format("ORG", "Use original data (default)")
					print fmt.format("PCA", "Reduce data with Principal Component Analysis (NOTE: parameters must be changed manually)")
					print fmt.format("LEM", "Reduce data with Laplacian Eigenmaps (NOTE: parameters must be changed manually)")
					print fmt.format("small", "Use small training dataset (default)")
					print fmt.format("large", "Use large training dataset")
					print fmt.format("help", "Show help")
					print fmt.format("exit", "Exit Binary digit recognition system")
				elif userinput[0] == "exit":
					print "Exiting Binary digit recognition system..."
					break
				elif userinput[0] == "LS":
					sys.classifier = Least_Square_Classifier()
					print "Using Least Sum of Squares classifier!"
				elif userinput[0] == "2LP":
					sys.classifier = Two_Layer_Perceptron_Classifier()
					print "Using Two-Layer Perceptron classifier!"
				elif userinput[0] == "GNB":
					sys.classifier = Gaussian_Naive_Bayes_classifier()
					print "Using Gaussian Naive Bayes classifier!"
				elif userinput[0] == "P":
					sys.classifier = Perceptron_Classifier()
					print "Using Perceptron classifier!"
				elif userinput[0] == "ORG":
					sys = BinaryRecognitionSystem()
					print "Using original data! (classifier is reset to LS)"
				elif userinput[0] == "PCA":
					sys = BinaryRecognitionSystem()
					print "Reducing data..."
					start_time()
					sys.Xtr_digits = PCA(sys.Xtr_digits)
					sys.Xtr_digits_larger = PCA(sys.Xtr_digits_larger)
					sys.Xte_digits_2015 = PCA(sys.Xte_digits_2015)
					stop_time("PCA")
					print "Using PCA reduced data! (classifier is reset to LS)"
				elif userinput[0] == "LEM":
					sys = BinaryRecognitionSystem()
					print "Reducing data..."
					start_time()
					sys.Xtr_digits = LEM(sys.Xtr_digits)
					sys.Xtr_digits_larger = LEM(sys.Xtr_digits_larger)
					sys.Xte_digits_2015 = LEM(sys.Xte_digits_2015)
					stop_time("LEM")
					print "Using LEM reduced data! (classifier is reset to LS)"
				elif userinput[0] == "small":
					use_small = True
					use_large = False
					print "Using small training dataset!"
				elif userinput[0] == "large":
					use_small = False
					use_large = True
					print "Using large training dataset!"
				elif userinput[0] == "decode":
					# Train with training data
					if (use_small == True):
						sys.classifier.train(sys.Xtr_digits, sys.ytr_digits)
					elif (use_large == True):
						sys.classifier.train(sys.Xtr_digits_larger, sys.ytr_digits_larger)

					# Classify test data
					result = sys.classifier.classify(sys.Xte_digits_2015)
					print("Number of misclassified binary digits out of a total %d binary digits: %d" % (sys.Xte_digits_2015.shape[0],(result != sys.solution).sum()))
					# Print result
					result = result.T
					ASCII(result[0])
				else:
					print "Command not found"
			except IndexError:
				print "Please input a command"
