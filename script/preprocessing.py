#import
import re
import sys
import time
import nltk
import string
import spell_check
import numpy as np

from nltk.tokenize import TweetTokenizer
from wordsegment import load, segment
from collections import Counter
from nltk.stem.snowball import SnowballStemmer
from itertools import groupby

def upload_dict(file, dup=False, order=False):
    '''if dup, duplicate smiley set by adding -
    if order, take first half as keys'''
    path = '../data/'+file
    word_list = [line.rstrip('\n') for line in open(path)]
    keys = np.asarray(word_list)[2*np.arange(int(len(word_list)/2))]
    keys = [k.strip() for k in keys]
    values = np.asarray(word_list)[2*np.arange(int(len(word_list)/2))+1]
    word_dict = dict(zip(keys, values))
    if dup:
        keys2 = [k.replace('-','') for k in keys]
        word_dict = dict(zip(keys+keys2, values+values))
    return word_dict
def replace_neg_verbs(tweet_list):
    #Verbs dataset
    verb_dict = upload_dict('neg_verbs.txt')
    
    new_pos = []
    for tweet in tweet_list:
        words = tokenize(tweet)
        for i in range(len(words)):
            if words[i] in verb_dict.keys():
                words[i] = verb_dict[words[i]]
        new_pos.append(' '.join(word for word in words))
    return new_pos
def remove_punctuation(tweet_list):
    '''Remove all punctuations from string'''
    new_pos = []
    for tweet in tweet_list:
        new_pos.append("".join(l for l in tweet if l not in string.punctuation))
    return new_pos
def tokenize(text):
    tknzr = TweetTokenizer()
    return tknzr.tokenize(text)
def remove_stopwords(tweet_list):
    '''Remove all stopwords'''
    stop_words = [line.rstrip('\n').lower() for line in open('../data/stopwords.txt')]
    
    new_pos = []
    for tweet in tweet_list:
        words = [word for word in tokenize(tweet) if not word in stop_words]
        new_pos.append(' '.join(word for word in words))
    return new_pos
def replace_smiley_slang(tweet_list):
    #Smiley dataset
    smiley_dict = upload_dict('smiley.txt')
    
    #Slangs datasets 1
    slang_dict = upload_dict('slang.txt')
    
    #Slang dataset 2
    slang_dict2 = upload_dict('slang2.txt')
    
    new_pos = []
    for tweet in tweet_list:
        words = tokenize(tweet)
        for i in range(len(words)):
            if words[i] in smiley_dict.keys():
                words[i] = smiley_dict[words[i]]
            elif words[i] in slang_dict.keys():
                words[i] = slang_dict[words[i]]
            elif words[i] in slang_dict2.keys():
                words[i] = slang_dict2[words[i]]
        new_pos.append(' '.join(word for word in words))
    return new_pos
def separate(words):
    return ' '.join(word for word in segment(words))
def segmentation(tweet_list):
    new = []
    dictionary = Counter(tokenize(open('../data/english_words.txt').read()))
    for tweet in tweet_list:
        tokens = tokenize(tweet)
        for i in range(len(tokens)):
            if tokens[i] not in dictionary:
                tokens[i] = separate(tokens[i])
        new.append(' '.join(word for word in tokens))
    return new
def correct_char_repetition(word):
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
def tweet_correct_char_repetition(tweet_list):
	new = []
	for tweet in tweet_list:
		tokens = tokenize(tweet)
		for i in range(len(tokens)):
			tokens[i] = correct_char_repetition(tokens[i])
		new.append(' '.join(word for word in tokens))
	return new

def write_to_file(tweet_list, file):
	path = '../data/filterd_data/filterd_' + file
	f = open(path, 'w')
	for tweet in tweet_list:
		f.write(tweet + '\n')
	f.close()
	print('the file has been successfully created in :: ', path)

def check_file(path):
    try:
      open(path, "r")
      return True
    except IOError:
      print("Error: File does not appear to exist.")
      return False

def filter(file):
	path = '../data/'+file
	print('Cleaning started for the file :: ', path)
	if check_file(path):
		tweet_list = [line.rstrip('\n') for line in open(path, encoding="utf8")]
		tweet_list = replace_neg_verbs(tweet_list)
		tweet_list = replace_smiley_slang(tweet_list)
		tweet_list = remove_punctuation(tweet_list)
		tweet_list = segmentation(tweet_list)
		tweet_list = tweet_correct_char_repetition(tweet_list)
		tweet_list = remove_stopwords(tweet_list)
		write_to_file(tweet_list, file)


def main():
	for arg in sys.argv[1:]:
		load()
		filter(arg)
        
if __name__ == "__main__":
	main()