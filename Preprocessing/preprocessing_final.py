###############################################################
########################## IMPORTS ############################
###############################################################
import numpy as np
import nltk
import string
from nltk.tokenize import TweetTokenizer
from wordsegment import load, segment
from collections import Counter
from nltk.stem.snowball import SnowballStemmer
import pickle
from itertools import groupby
import re
import sys
import time
###############################################################
########################## FUNCTIONS ##########################
###############################################################
def tokenize(text):
	'''
	Tokenize string using nltk tweet tokenizer
	param text: string
	return: list of tokens
	'''
	tknzr = TweetTokenizer()
	return tknzr.tokenize(text)

def correct_char_repetition(word):
	'''
	correct word with consecutive repeat characters (more than 3) to 2 charcters
	params word: string
	return corrected string
	'''
	if word == '': 
		return word
	word = word.lower()
	occurance = [(k, sum(1 for i in g)) for k,g in groupby(word)]
	if len(occurance)==1: 
		return word
	if max([j for (_,j) in occurance]) > 2:
		corrected_word = ''
		for (i,j) in occurance:
			if j>2:
				corrected_word += 2*i
			else:
				corrected_word += i*j
		return corrected_word
	else:
		return word

def check_file(path):
	'''
	Check the existence of file
	param path: string
	return bool
	'''
	try:
	  open(path, "r")
	  return True
	except IOError:
	  print("Error: File does not appear to exist.")
	  return False

def load_file(path):
	'''
	load file line by line
	param file: string
	return: list of string
	'''
  	if check_file(path):
  		return [line.rstrip('\n').lower() for line in open(path, encoding = 'utf8')]

#....................................................................................
#.....................................Spell_check....................................
#....................................................................................
def spell_check(word, english_dictionary, final_dict):
	if word in english_dictionary:
		return word
	cand_word = candidates(word, english_dictionary)
	if len(cand_word) == 1:
		return list(cand_word)[0]
	else:
		pr = [english_dictionary[cand] for cand in cand_word]
		return list(cand_word)[np.argmax(pr)]

	
def candidates(word, english_dictionary): 
	'''
	Generate possible spelling corrections for word
	param word: string
	param english_dictionary: dictionary
	returns spell checking candidates
	'''
	return (known(edits1(word), english_dictionary) or known(edits2(word), english_dictionary) or [word])

def known(words, english_dictionary): 
	'''
	The subset of `words` that appear in the dictionary of WORDS
	param word: string
	param english_dictionary: dictionary
	returns the subset of `words` that appear in the dictionary of WORDS
	'''
	return set(w for w in words if w in english_dictionary)

def edits1(word):
	'''
	All edits that are one edit away from `word`
	param word: string
	return: set of words, all combinations of possible typos with one error (deletes, transposes,
	replaces, inserts)
	'''
	letters	= 'abcdefghijklmnopqrstuvwxyz'
	splits	 = [(word[:i], word[i:])	for i in range(len(word) + 1)]
	deletes	= [L + R[1:]			   for L, R in splits if R]
	transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
	replaces   = [L + c + R[1:]		   for L, R in splits if R for c in letters]
	inserts	= [L + c + R			   for L, R in splits for c in letters]
	return set(deletes + transposes + replaces + inserts)

def edits2(word): 
	'''
	All edits that are two edits away from `word`
	param word: string
	return: set of words, all combinations of possible typos with two error (deletes, transposes,
	replaces, inserts)
	'''
	return (e2 for e1 in edits1(word) for e2 in edits1(e1))
################################################################################################

def correct_tweet_list(tweet_list, english_dictionary, final_dict):
	'''
	correct tweet list
	param tweet_list: list of strings, tweets
	param english_dictionary: dictionary
	param final_dict: dictionary
	return: list of strings, corrected tweets
	'''
	new_tweet_list = []
	count = 0
	for tweet in tweet_list:
		new_tweet, final_dict = correct_tweet(tweet, english_dictionary, final_dict)
		new_tweet_list.append(new_tweet)
		if count % 10000 == 0: print(time.ctime(), '  count = ', count, len(final_dict))
		count += 1
	return new_tweet_list, final_dict

def correct_tweet(tweet, english_dictionary, final_dict):
	'''
	correct tweet
	param tweet: string, tweet
	param english_dictionary: dictionary
	param final_dict: dictionary
	return: string, corrected tweet
	'''
	tokens = tokenize(tweet)
	new_tokens = []
	for word in tokens:
		(new_word, found) = check_word(word, english_dictionary, final_dict)
		if not found:
			(new_word, found2) = filter1(word, english_dictionary, final_dict)
			if not found2:
				new_word2 = correct(new_word, english_dictionary, final_dict)
				if new_word2 == new_word and hasDigits(new_word):
					new_word = ''.join([s for s in new_word if not s.isdigit()])
					new_word = correct(new_word, english_dictionary, final_dict)
				else:
					new_word = new_word2
		final_dict[word] = new_word
		new_tokens.append(new_word)
	new_tweet = ' '.join(new_tokens)
	return new_tweet, final_dict

def hasDigits(word):
	'''
	check whether a word contains digits
	param word: string
	return: bool
	'''
	return any(char.isdigit() for char in word)

def check_word(word, english_dictionary, final_dict, treshold = 1):
	'''
	check if a word is in the dictionary or has length less than the threshold
	param word: string
	param final_dict: dictionary
	param threshold: int, default value 1
	return: bool
	'''
	
	if word in final_dict: 
		return (final_dict[word], True)
	new_word = correct_char_repetition(word)
	if len(new_word) < treshold:
		return ('', True)
	else:
		return (new_word, False)
	
def filter1(word, english_dictionary, final_dict):
	'''
	remove punctuations/ char correction
	param word: string
	param english_dictionary: dictionary
	param final_dict: dictionary
	return filtered word
	'''
	new_word_ = correct_char_repetition(word)
	new_word_ = ''.join([s for s in word if s not in set(string.punctuation) and not s.isdigit()])
	
	return check_word(new_word_, english_dictionary, final_dict)

def correct(word, english_dictionary, final_dict):
	'''
	correct word by applying dictionary check, segmentation and spell check 
	param word: string
	param english_dictionary: dictionary
	param final_dict: dictionary
	return corrected word
	'''
	(new_c, truth_c) = check_word(word, english_dictionary, final_dict)
	if truth_c:
		return new_c
	else:
		segments = segment(word)
		if len(segments)==1:
			return spell_check(word, english_dictionary, final_dict)
		else:
			return ' '.join([correct(ws, english_dictionary, final_dict) for ws in segments])


###############################################################
############################ MAIN #############################
###############################################################
def main():
	# Check if exactly three file path are given
	if len(sys.argv) != 4:
		print('ERROR:: 3 parameters are required \n>>>> 1: train_pos \n>>>> 2: train_neg \n>>>> 3: test')
	else:

		# Loading data
		data_path = '../data/'
		print(time.ctime(), data_path + sys.argv[1])
		pos = load_file(data_path + sys.argv[1])
		neg = load_file(data_path + sys.argv[2])
		test = load_file(data_path + sys.argv[3])
		data = [pos, neg, test]
		del pos, neg, test
		print(time.ctime(), '  Files loaded successfully ......')


		new_data = []
		load()

		# Loading initial dictionary
		dict_path = data_path + 'dictionaries.p'
		dictionaries = pickle.load( open(dict_path, 'rb'))
		final_dict = {k:v for k,v in dictionaries[0].items() if len(k)>1}
		english_dictionary = {k:v for k,v in dictionaries[1].items() if len(k)>1}

		for tweet_list in data:
			print(time.ctime(), '  Started for data xx')
			print(time.ctime(), '  Dictionaray length = ', len(final_dict))
			new_tweet_list, final_dict = correct_tweet_list(tweet_list, english_dictionary, final_dict)
			new_data.append(new_tweet_list)
			print(time.ctime(), '  Done for data xx ::::')
			print(time.ctime(), '  Dictionaray length = ', len(final_dict))

		# Store the final dict
		BASE = '../data/corrected_data/'
		pickle.dump(final_dict, open( BASE + "final_tokens_dictionary.p", "wb" ))
		pickle.dump(new_data, open( BASE + "corrected_datasets_pos_neg_test.p", "wb" ))
		print(time.ctime(), '  Finished .... data stored in {} ::: {}'.format(BASE))

if __name__ == "__main__":
	main()