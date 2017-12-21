###############################################################
########################## IMPORTS ############################
###############################################################

import fasttext
import numpy as np
import csv
from fasttext_grid_search import load_file, write_to_file
from fasttext import skipgram
from fasttext import supervised


###############################################################
############################ FUNCTIONS ########################
###############################################################

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    params: ids (event ids associated with each prediction)
            y_pred (predicted class labels)
            name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})

###############################################################
############################ MAIN #############################
###############################################################

def main():

	#Setting best paramters for classifier.
	dim = 200
	lr = 0.05
	ws = 3
	epoch = 3
	word_ngrams = 3

	#Loading data
	pos = load_file('../data/train_pos_full.txt')
	neg = load_file('../data/train_neg_full.txt')
	test = load_file('../data/test_data.txt')

	#Adding labels.
	pos_label=[p + ' __label__positive' for p in pos]
	neg_label=[n + ' __label__negative' for n in neg]

	#Creating training set.
	train = list(pos_label+neg_label)
	full_data=list(pos+neg)

	#writing files.
	write_to_file(train, 'train.txt')
	write_to_file(full_data,'data.txt')

	#Running fasttext on best parameters found by grid_search
	model = skipgram('data.txt' , 'model', dim=dim, lr=lr, ws=ws, epoch=epoch, word_ngrams=word_ngrams)

	#Creating classifier and predicting sentiments.                         
	classifier = supervised('train.txt', 'model', label_prefix='__label__')
	labels=classifier.predict(test)

	#
	pred = [1 if t==['positive'] else -1 for t in labels]

	#Create csv file.
	create_csv_submission(np.arange(1, len(pred)+1), pred, 'sub.csv')

if __name__ == "__main__":
	main()