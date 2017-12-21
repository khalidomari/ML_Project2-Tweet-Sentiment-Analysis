###############################################################
########################## IMPORTS ############################
##############################################################
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
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

###############################################################
########################## FUNCTIONS ##########################
##############################################################
def accuracy(y1, y2): 
   '''Calculate accuracy'''
   return 100 - np.sum(np.abs(np.asarray(y1) - np.asarray(y2))/2)*100/len(y1)


###############################################################
############################ MAIN ############################
##############################################################
# Train various linear classifiers using grid search and print the obtained accuracies
    
def main():

    [X_train, X_test, y_train, y_test, test_vec] = pickle.load( open('../data/embeddings/word2vec/train_test_splited_word2Vec0.05_d400.p', 'rb') )
    
    size = 2000000
    X = X_train[:size]
    y = y_train[:size]

    # Centering and normalizing the data
    print('Scaler fitting ...')
    scaler = StandardScaler()
    scaler.fit(X)
    print('Scaling the training set ...')
    X = scaler.transform(X)
    X_test = scaler.transform(X_test)
    
    # Define hyperparameters
	C = [10, 1, 1e-1, 1e-2]

	test_acc = []
	for C_ in C:
	    print('C: ', C_)
	    linearSVC = LinearSVC(loss="squared_hinge", dual=False, C=C_, verbose=2)

	    linearSVC.fit(X_train, y_train)

	    y_pred = linearSVC.predict(X_test)
	    test_acc.append(accuracy(y_test, y_pred))
            
        # Save obtained results 
        test_params.append("C=" + str(C_)
        test_acc.append(accuracy(y_test, y_pred))


    for key, value in zip(test_params,test_acc):
        print(key, value) 

if __name__ == "__main__":
	main()