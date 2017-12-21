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
from sklearn.linear_model import SGDClassifier

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
    lossfun = ["hinge", "modified_huber", "log"]
    alpha = [1e-5, 1e-4, 1e-3]
    l1_ratio = [0, 0.15, 0.3, 0.6, 0.9]

    test_params = []
    test_acc = []
    for lossfun_ in lossfun:
        print(lossfun_)
        for alpha_ in alpha:
            for l1_ratio_ in l1_ratio:
                print('alpha: ', alpha_)
                print('l1_ratio: ', l1_ratio_)
                
                # Define and train SGD classifier
                sgd_svc = SGDClassifier(loss=lossfun_, penalty="elasticnet", alpha=alpha_, n_jobs=-1, max_iter=20, l1_ratio=l1_ratio_, verbose=2)
                sgd_svc.fit(X_train, y_train)

                # Evaluate accuracy of classifier on testing set
                y_pred = sgd_svc.predict(X_test)
            
                # Save obtained results 
                test_params.append(str(lossfun_) + ", alpha=" + str(alpha_) + ", gamma=" + str(l1_ratio_))
                test_acc.append(accuracy(y_test, y_pred))


    for key, value in zip(test_params,test_acc):
        print(key, value) 

if __name__ == "__main__":
	main()