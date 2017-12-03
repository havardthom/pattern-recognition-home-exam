import scipy.io as sio
import numpy as np
import time
from matplotlib import pyplot as plt

from PCA import PCA
from LEM import LEM
from kNN import kNN

use_pca = True
use_lem = False

def start_time():
    global TICK
    TICK = time.time()

def stop_time(prefix):
    global TICK
    old = TICK
    TICK = time.time()
    print(prefix + " used " + str(TICK-old) + " seconds")

class RecommenderSystem:
	def __init__(self, filename):
		# Initialize class with song data
		data = sio.loadmat(filename)
		self.X = data['X']
		self.song_info = data['song_info']

	def recommend_songs(self, songnumber):
		if use_pca:
			# Get reduced (2 dimensions) data using PCA
			# start_time()
			self.transformed = PCA(self.X)
			# stop_time("PCA")
		elif use_lem:
			# Get reduced (2 dimensions) data using LEM
			# start_time()
			self.transformed = LEM(self.X)
			# stop_time("LEM")

		# Get seed data point
		self.p = self.transformed[songnumber]
		# Get 20 nearest neighbors of seed
		self.idx = kNN(self.transformed, self.p, 20)[0]

	def visualize(self):
		try:
			x1 = self.transformed[:,0]
			x2 = self.transformed[:,1]
		except AttributeError:
			print "No ranked list to visualize, run the recommender first"
			return

		# Plot all data points
		plt.figure(1)
		plt.plot(x1[:42], x2[:42], '.b', label='class1')
		plt.plot(x1[42:], x2[42:], '.r', label='class2')

		# Plot seed with different marker
		if songnumber < 42:
			plt.plot(self.p[0],self.p[1], '^b', label='seed')
		else:
			plt.plot(self.p[0],self.p[1], '^r', label='seed')

		# Plot markers for neighbors of seed
		plt.plot(x1[self.idx],x2[self.idx],'.k',
		  markerfacecolor='None', markersize=15, markeredgewidth=1, label='neighbors')

		plt.xlabel('x1')
		plt.ylabel('x2')
		plt.legend(loc='lower left')
		plt.title('Visualization of latest ranked list')
		plt.show()


if __name__ == "__main__":
	rec = RecommenderSystem('data/songdata_full.mat')

	print 'Welcome to RecoMusic! (type help for commands)'
	while True:
			userinput = raw_input('Command: ').split()
			try:
				if userinput[0] == "help":
					fmt = ' {0:25} # {1:}'
					print fmt.format("'songnumber'(eg. 81)", "Output a ranked list of recommended songs based on the song listed on songnumber")
					print fmt.format("visualize", "Visualize the latest ranked list")
					print fmt.format("list", "List available songs")
					print fmt.format("PCA", "Use Principal Component Analysis for reduction (default)")
					print fmt.format("LEM", "Use Laplacian Eigenmaps for reduction (NOTE: parameters must be changed manually)")
					print fmt.format("help", "Show help")
					print fmt.format("exit", "Exit music recommender")
				elif userinput[0] == "list":
					for i in range(len(rec.song_info)):
						print str(i+1) + ': ' + str(rec.song_info[i][0][0]) + ' - ' + str(rec.song_info[i][1][0])
				elif userinput[0] == "exit":
					print "Exiting music recommender..."
					break
				elif userinput[0] == "visualize":
					rec.visualize()
				elif userinput[0] == "PCA":
					use_pca = True
					use_lem = False
					print "Using PCA!"
				elif userinput[0] == "LEM":
					use_pca = False
					use_lem = True
					print "Using LEM!"
				else:
					try:
                        # Get songnumber
						songnumber = int(userinput[0])
						songnumber -= 1
                        # Check if its in range
						if 0 <= songnumber < len(rec.song_info):
                            # Get indices of recommended songs in rec.idx
							rec.recommend_songs(songnumber)
                            # Print results
							print "Top 20 recommended songs based on '" + str(rec.song_info[songnumber][0][0]) + " - " + str(rec.song_info[songnumber][1][0]) + "'"
							fmt = '# {0:4} # {1:30} # {2:}'
							print fmt.format("\033[1mRank", "Artist", "Title\033[0m")
							for i in range(len(rec.idx)):
								artist = rec.song_info[rec.idx[i]][0][0]
								title = rec.song_info[rec.idx[i]][1][0]
								print fmt.format(str(i+1), str(artist), str(title))
						else:
							print "Songnumber out of range, type list to get available songs"
					except ValueError:
						print "Command not found"
			except IndexError:
				print "Please input a command"
