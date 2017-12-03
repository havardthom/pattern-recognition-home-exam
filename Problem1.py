import scipy.io as sio
import numpy as np

from matplotlib import pyplot as plt
# Including this to avoid UnicodeEncodeError when printing norwegian letters

from Bayes import Gaussian_Naive_Bayes_classifier
from LS import Least_Square_Classifier
from Perceptron import Perceptron_Classifier, Two_Layer_Perceptron_Classifier
from sklearn.svm import LinearSVC

class ClassificationSystem:
	def __init__(self, filename):
		songdata = sio.loadmat(filename)
		self.Xtr = songdata['Xtr']
		self.ytr = songdata['ytr']
		self.song_info_tr = np.array(songdata['song_info_tr'])
		self.Xte = songdata['Xte']
		self.yte = songdata['yte']
		self.song_info_te = np.array(songdata['song_info_te'])

		self.classifier = Least_Square_Classifier()

if __name__ == "__main__":
	sys = ClassificationSystem('data/songdata_small.mat')

	# # Testing with linear Support Vector Machine from sklearn
	# svm = LinearSVC()
	# svm.fit(sys.Xtr, sys.ytr.T[0])
	# result = svm.predict(sys.Xte)
	# print("Number of misclassified songs out of a total %d songs: %d" % (sys.Xte.shape[0],(result[np.newaxis].T != sys.yte).sum()))

	print 'Welcome to Music Genre classification system! (type help for commands)'
	while True:
			userinput = raw_input('Command: ').split()
			try:
				if userinput[0] == "help":
					fmt = ' {0:25} # {1:}'
					print fmt.format("classify", "Categorize music into genres with chosen classifier")
					print fmt.format("LS", "Use Least Sum of Squares classifier (default)")
					print fmt.format("2LP ", "Use Two-Layer Perceptron classifier (NOTE: parameters must be changed manually)")
					print fmt.format("GNB", "Use Gaussian Naive Bayes classifier")
					print fmt.format("P", "Use Perceptron classifier (NOTE: parameters must be changed manually)")
					print fmt.format("songdata_small", "Use small dataset (default)")
					print fmt.format("songdata_big", "Use big dataset")
					print fmt.format("help", "Show help")
					print fmt.format("exit", "Exit music genre classification system")
				elif userinput[0] == "exit":
					print "Exiting Music Genre classification system..."
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
				elif userinput[0] == "songdata_small":
					sys = ClassificationSystem('data/songdata_small.mat')
					print "Using small dataset! (classifier is reset to LS)"
				elif userinput[0] == "songdata_big":
					sys = ClassificationSystem('data/songdata_big.mat')
					print "Using big dataset! (classifier is reset to LS)"
				elif userinput[0] == "classify":
					# Train with training data and classify test data
					sys.classifier.train(sys.Xtr, sys.ytr)
					result = sys.classifier.classify(sys.Xte)

					# Print results
					fmt = '# {0:13} # {1:30} # {2:}'
					print fmt.format("GENRE [wrong]", "ARTIST", "TITLE")
					for i in range(len(sys.Xte)):
						artist = sys.song_info_te[i][0][0]
						title = sys.song_info_te[i][1][0]
						# Get correct genre
						if result[i] == 1:
							genre = 'Rap'
						elif result[i] == -1:
							genre = 'Classical'
						if result[i] != sys.yte[i]:
							print fmt.format("["+str(genre)+"]", str(artist), str(title))
						else:
							print fmt.format(str(genre), str(artist), str(title))
					print("Number of misclassified songs out of a total %d songs: %d" % (sys.Xte.shape[0],(result != sys.yte).sum()))
				else:
					print "Command not found"
			except IndexError:
				print "Please input a command"
