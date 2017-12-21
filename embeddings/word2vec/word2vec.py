#import
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from collections import Counter
import numpy as np
import nltk
import string
import gensim
import pickle
import time
import csv

def check(tweet_list_tokens):
	'''
	replace empty tweets with empty
	:param tweet_list_tokens: list of tokens lists, list of tokenized tweets
	:return: list of list of tokens lists
	'''
	return [t if t != [] else ['empty'] for t in tweet_list_tokens]

def tweet2vector(tweet_tokens, model, tf_idf_dic = None):
	'''
	compute a vector representation of a tweet based on the average word2vec embeddings of each token if 
	tf_idf_dic is None, otherwise using tf_idf weights
	:param tweet_tokens: list of tokens, tokenized tweet
	:param model: word2vec model
	:param tf_idf_dic: dic, tf_idf scores of each word
	:return: array, vector representation of tweet
	'''
	default = np.zeros_like(model['happy'])
	if len(tweet_tokens)==0: return default
	if tf_idf_dic != None:
		return sum([tf_idf_dict[word]*model[word] if word in model.wv.vocab and word in tf_idf_dict.keys() else default
				for word in tweet_tokens ])/len(tweet_tokens)
	return sum([model[word] if word in model.wv.vocab else default for word in tweet_tokens ])/len(tweet_tokens)

def main():
	#Loading Data
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

	load = False
	if not load:
		#Generate word2vec model
		print('Generating word2vec model ...')
		size = 150
		window = 3
		model = gensim.models.Word2Vec(pos + neg + test, size=size, window=window, min_count=1, workers=4)

		#Store w2v model
		print('Storing word2vec model ...')
		pickle.dump(model, open('word2vec_model_ s{}_w{}.p'.format(size, window), 'wb'))

		'''
		#Calculate tf-idf scores
		print('Generating tf_idf dict ...')
		corpus = [' '.join(tokens) for tokens in pos_tokens + neg_tokens + test_tokens]
		vectorizer = TfidfVectorizer(min_df=1)
		X = vectorizer.fit_transform(corpus)
		idf = vectorizer.idf_
		tf_idf_dict = dict(zip(vectorizer.get_feature_names(), idf))

		#Store tfidf dict
		print('Storing tf_idf dict ...')
		pickle.dump(tf_idf_dict, open('tf_idf.p', 'wb'))
		'''
	else:
		model_path = 'word2vec_model.p'
		model = pickle.load(open(model_path, 'rb'))

	tf_idf_dict = None

	#Tweet2vec
	print('Tweets to vectors ...')
	pos_vec = np.asarray([tweet2vector(tweet, model, tf_idf_dict) for tweet in pos])
	neg_vec = np.asarray([tweet2vector(tweet, model, tf_idf_dict) for tweet in neg])
	test_vec = np.asarray([tweet2vector(tweet, model, tf_idf_dict) for tweet in test])

	#Concatenate pos and neg vectors
	print('Preparing final data ...')
	X = np.vstack((pos_vec, neg_vec))
	y = [1 for i in range(len(pos_vec))] + [-1 for i in range(len(neg_vec))]

	#Separate data to train and test
	print('train test split ...')
	test_size = 0.05
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

	#Store final data
	print('Storing final data ...')
	pickle.dump([X_train, X_test, y_train, y_test, test_vec], open('../../data/embeddings/word2vec/train_test_splited_word2Vec_ s{}_w{}_tests{}.p'.format(size, window, test_size), 'wb'))

	print('Done')

if __name__ == "__main__":
	main()