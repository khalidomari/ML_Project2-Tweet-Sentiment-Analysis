#import
import numpy as np
import nltk
import string
from nltk.tokenize import TweetTokenizer
import spell_check
from wordsegment import load, segment
from collections import Counter
from nltk.stem.snowball import SnowballStemmer










def main():
	pos_train = [line.rstrip('\n') for line in open('../data/train_pos.txt')]
	neg_train = [line.rstrip('\n') for line in open('../data/train_neg.txt')]
	for arg in sys.argv[1:]:
        print arg



if __name__ == "__main__":
	main()
