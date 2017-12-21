###############################################################
########################## IMPORTS ############################
###############################################################
import numpy as np
import string
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
import pickle
from collections import Counter

###############################################################
########################## FUNCTIONS ##########################
###############################################################
def count_word(word, c_pos, c_neg):
	'''
	counts the occurence of a given word in two counter respectively
	:param c_pos: counter, positive counter
	:param c_neg: counter, negative counter
	:return: int, int
	'''
	count_pos = 0
	count_neg = 0
	if word in c_pos:
		count_pos = c_pos[word]
	if word in c_neg:
		count_neg = c_neg[word]
	return count_pos, count_neg

def remove_stopwords(tweet_list_tokens, stopwords, unique=True):
	'''
	Remove stopwords and remove repeated tweets
	param tweet_list_tokens: list of tokens lists, list of tokenized tweets
	param stopwords: list of strings, stopwords
	param unique: bool
	return: list of tokens lists, without stopwords, and without repeated items if unique
	'''
	new_list = []
	for tweet_tokens in tweet_list_tokens:
		words = [word for word in tweet_tokens if not word in stopwords]
		new_tweet = ' '.join(word for word in words)
		new_list.append(new_tweet)

	if unique:
		new_list = list(set(new_list))
	print('before ==> {} tweets'.format(len(tweet_list_tokens)))
	print('after  ==> {} tweets'.format(len(new_list)))
	return new_list


###############################################################
############################ MAIN #############################
###############################################################
def main():
	print('loading data ...')
	DATA_PATH = '../data/corrected_data/'
	pos, neg, test = pickle.load(open(DATA_PATH+'corrected_datasets_pos_neg_test.p', 'rb'))

	print('tokenizing and lemmatizing data ...')
	tknzr = TweetTokenizer()
	lemmatizer = WordNetLemmatizer()
	pos = list([[lemmatizer.lemmatize(w) for w in tknzr.tokenize(tweet)] for tweet in pos])
	neg = list([[lemmatizer.lemmatize(w) for w in tknzr.tokenize(tweet)] for tweet in neg])
	test = list([[lemmatizer.lemmatize(w) for w in tknzr.tokenize(tweet)] for tweet in test])

	# Counters
	print('counting stopwords ...')
	c_pos = Counter([word for tokens in pos for word in tokens])
	c_neg = Counter([word for tokens in neg for word in tokens])

	#Import Stopwords
	stop_words = [line.rstrip('\n').lower() for line in open('../data/dictionaries/stopwords.txt')]
	stop_words = [w for w in stop_words if w in c_pos+c_neg]
	print('{} stopwords in the list'.format(len(stop_words)))

	THRESHOLD = 1 #.3
	del_stopwords = []
	for word in stop_words:
		count_pos, count_neg = count_word(word, c_pos, c_neg)
		count_total = count_pos + count_neg
		if  count_total > 0 and np.abs(count_pos - count_neg)/(count_total) <= THRESHOLD:
			del_stopwords.append(word)
	print('{} stopwords will be removed'.format(len(del_stopwords)))
	print('the remaining words :', set(stop_words)-set(del_stopwords))

	#remove stopwords
	print('removing stopwords ...')
	new_data = [remove_stopwords(pos, del_stopwords), remove_stopwords(neg, del_stopwords), remove_stopwords(test, del_stopwords, unique=False)]

	# Store processed tweets
	print('storing data ...')
	BASE = '../data/corrected_data/'
	pickle.dump(new_data, open( BASE + "corrected_datasets_stopwords_pos_neg_test.p", "wb" ))

if __name__ == "__main__":
	main()