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


def tokenize(text):
	tknzr = TweetTokenizer()
	return tknzr.tokenize(text)

def check(tweet_list_tokens):
	'''replace empty tweets with empty'''
	return [t if t != [] else ['empty'] for t in tweet_list_tokens]

def del_stopWords(tweet_list_tokens, stop_words):
	'''remove stopwords from tokens list'''
	return [[t for t in tokens if t not in stop_words] for tokens in tweet_list_tokens]

def tweet2vector(tweet_tokens, model, tf_idf_dict):
	default = np.zeros_like(model['happy'])
	return sum([tf_idf_dict[word]*model[word] if word in model.wv.vocab and word in tf_idf_dict.keys() else default
				for word in tweet_tokens ])/len(tweet_tokens)

def main():
	print('Loading data .... ')
	[pos, neg, test] = pickle.load(open('corrected_datasets_pos_neg_test.p','rb'))
	pos = list(set(pos))
	neg = list(set(neg))

	#Tokenize and lemmatize
	print('Tokenizing and lemmatizing ...')
	lemmatizer = WordNetLemmatizer()

	print('~~~ positive ...')
	pos_tokens = []
	for tweet in pos:
		pos_tokens.append([lemmatizer.lemmatize(w) for w in tokenize(tweet)])
	pos_tokens = list(np.unique(check(pos_tokens)))
	
	print('~~~ negative ...')
	neg_tokens = []
	for tweet in neg:
		neg_tokens.append([lemmatizer.lemmatize(w) for w in tokenize(tweet)])
	neg_tokens = list(np.unique(check(neg_tokens)))

	print('~~~ test ...')
	test_tokens = []
	for tweet in test:
		test_tokens.append([lemmatizer.lemmatize(w) for w in tokenize(tweet)])


	#Loading Stopwords
	stop_words = [line.rstrip('\n').lower() for line in open('stopwords.txt')]

	#Remove Stopwords
	print('Removing stopwords ...')
	pos_tokens = del_stopWords(pos_tokens, stop_words)
	neg_tokens = del_stopWords(neg_tokens, stop_words)
	test_tokens = del_stopWords(test_tokens, stop_words)

	#Replace empty tweets with empty
	pos_tokens = list(np.unique(check(pos_tokens)))
	neg_tokens = list(np.unique(check(neg_tokens)))
	test_tokens = list(check(test_tokens))

	#Generate word2vec model
	print('Generating word2vec model ...')
	model = gensim.models.Word2Vec(pos_tokens + neg_tokens + test_tokens, size=200, window=6, min_count=3, workers=5)

	#Store w2v model
	print('Storing word2vec model ...')
	pickle.dump(model, open('word2vec_model.p', 'wb'))

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

	#Tweet2vec
	print('Tweets to vectors ...')
	pos_vec = np.asarray([tweet2vector(tweet, model, tf_idf_dict) for tweet in pos_tokens])
	neg_vec = np.asarray([tweet2vector(tweet, model, tf_idf_dict) for tweet in neg_tokens])
	test_vec = np.asarray([tweet2vector(tweet, model, tf_idf_dict) for tweet in test_tokens])

	#Concatenate pos and neg vectors
	print('Preparing final data ...')
	X = np.vstack((pos_vec, neg_vec))
	y = [1 for i in range(len(pos_vec))] + [-1 for i in range(len(neg_vec))]

	#Separate data to train and test
	print('train test split ...')
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=0)

	#Store final data
	print('Storing final data ...')
	pickle.dump([X_train, X_test, y_train, y_test, test_vec], open('train_test_splited_word2Vec0.05.p', 'wb'))

	print('Done')

if __name__ == "__main__":
	main()