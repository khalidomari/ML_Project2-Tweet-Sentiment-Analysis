###############################################################
########################## IMPORTS ############################
###############################################################
import numpy as np
import string
import pickle

###############################################################
############################ MAIN #############################
###############################################################
def main():
	BASE_PATH = './glove.twitter.27B/'
	sizes = [25, 50, 100, 200]

	for s in sizes:
		file = BASE_PATH + 'glove.twitter.27B.{}d.txt'.format(s)
		print('Start for ', file)
		glove_embeddings = {}
		f = open(file, encoding='utf8')
		for line in f:
			tokens = line.split()
			word = tokens[0]
			vector = np.asarray(tokens[1:], dtype='float32')
			glove_embeddings[word] = vector
		f.close()

		#Pickle glove embeddings
		pickle.dump(glove_embeddings, open( file.replace('txt','p'), 'wb'))
		print('File stored in pickle file.')

if __name__ == "__main__":
	main()