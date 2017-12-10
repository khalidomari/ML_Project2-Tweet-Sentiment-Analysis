###############################################################
########################## IMPORTS ############################
###############################################################
import numpy as np
import string
from nltk.tokenize import TweetTokenizer
from sklearn.model_selection import train_test_split
import pickle

###############################################################
########################## FUNCTIONS ##########################
###############################################################
def glove_vectorize_tweet(tweet_tokens, glove_dict, keys):
    vec = np.zeros_like(glove_dict[keys[0]])
    for word in tweet_tokens: 
        vec += glove_dict[word]
    return vec

def glove_vectorize(tweet_tokens_list, glove_dict, keys):
	list_vec = []
	for tweet_tokens in tweet_tokens_list:
		list_vec.append(glove_vectorize_tweet(tweet_tokens, glove_dict, keys))
	return list_vec


###############################################################
############################ MAIN #############################
###############################################################
def main():
	#Loading data
	print('Loading data .... ')
	DATA_PATH = '../../data/corrected_data/corrected_datasets_stopwords_pos_neg_test.p'
	[pos, neg, test] = pickle.load(open(DATA_PATH,'rb'))
	pos = list(set(pos))
	neg = list(set(neg))

	#Tokenize
	print('Tokenizing ...')

	print('~~~ positive ...')
	tknzr = TweetTokenizer()
	pos = [tknzr.tokenize(tweet) for tweet in pos]

	print('~~~ negative ...')
	neg = [tknzr.tokenize(tweet) for tweet in neg]

	print('~~~ test ...')
	test = [tknzr.tokenize(tweet) for tweet in test] 

	# Load pretrained Glove embedding
	print('Loading Stanford Tweets Glove ...')
	file = 200
	GLOVE_FILE = './glove.twitter.27B/glove.twitter.27B.{}d.p'.format(file)
	glove_embeddings = pickle.load(open(GLOVE_FILE, 'rb'))
	keys = list(glove_embeddings.keys())

	# Reduce glove_embeddings
	all_tokens = list(set([w for tokens in pos+neg+test for w in tokens]))
	DEFAULT = np.zeros_like(glove_embeddings[keys[0]])
	glove_reduced = {}
	for key in all_tokens:
		glove_reduced[key] = DEFAULT
	print('here')
	for key in set(all_tokens).intersection(keys):
		glove_reduced[key] = glove_embeddings[key]
	print('{} ==> {}'.format(len(glove_embeddings), len(glove_reduced)))

	print('Tweets to vectors ...')

	print('~~~ positive ...')
	keys = list(glove_reduced.keys())
	pos = glove_vectorize(pos, glove_reduced, keys)

	print('~~~ negative ...')
	neg = glove_vectorize(neg, glove_reduced, keys)

	print('~~~ test ...')
	test = glove_vectorize(test, glove_reduced, keys)

	#print('store embeddings ...')
	#PATH = '../../data/embeddings/glove/glove_data.{}d.p'.format(file)
	#pickle.dump(glove_data, open(PATH, 'wb'))

	#Concatenate pos and neg vectors
	print('Preparing final data ...')
	X = np.vstack((pos, neg))
	y = [1 for i in range(len(pos))] + [-1 for i in range(len(neg))]

	#Separate data to train and test
	print('train test split ...')
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=0)

	#Store final data
	print('Storing final data ...')
	pickle.dump([X_train, X_test, y_train, y_test, test], open('../../data/embeddings/glove/train_test_splited_glove_data.{}d.p'.format(file), 'wb'))

	print('Done')



if __name__ == "__main__":
	main()