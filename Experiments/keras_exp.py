###############################################################
########################## IMPORTS ############################
###############################################################
import pickle
import time
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import GlobalAveragePooling1D
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.layers import LSTM

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
#from features import dumpFeatures

###############################################################
########################## FUNCTIONS ##########################
###############################################################
def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    name += '.csv'
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})

def accuracy(y1, y2):
	'''Calculate accuracy'''
	return 100 - np.sum(np.abs(np.asarray(y1) - np.asarray(y2))/2)*100/len(y1)

def update_description(lines):
	with open("descriptions.txt", "a") as f:
		for line in lines:
			f.write(line + "\n")

###############################################################
############################ MAIN #############################
###############################################################
def main():
	print(time.ctime(), '  Loading files .... ')
	DATA_PATH = '../data/embeddings/'
	GLOVE    = DATA_PATH + 'glove/train_test_splited_glove_data.200d.p'
	WORD2VEC = DATA_PATH + 'word2vec/train_test_splited_word2Vec0.05.p'

	EMBD = 'WORD2VEC'
	[X_train, X_test, y_train, y_test, test_vec] = pickle.load( open(WORD2VEC, 'rb'))
	max_features = 200

	size = 1000
	X_train = X_train[:size]
	y_train = y_train[:size]

	"""
	print('Scaler fitting ...')
	scaler = StandardScaler()
	scaler.fit(X_train)

	X = X_train
	test = X_test

	print('Scaling the training set ...')
	X = scaler.transform(X_train)
	print('Scaling the testing set ...')
	test = scaler.transform(X_test)
	"""


	################## Model ##########################
	###################################################
	print(time.ctime(), '  Generating model .... ')

	model = Sequential()
	model.add(Embedding(max_features+1, 200, input_length=X_train.shape[1]))
	model.add(Convolution1D(activation="relu", filters=32, kernel_size=3, padding="same"))
	model.add(MaxPooling1D(pool_size=2))
	model.add(Flatten())
	model.add(Dense(250, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	print(model.summary())

	"""
	Fitting with 0.1 validation split
	"""
	print('Fitting ...')
	model.fit(X_train, y_train, validation_split=0.1, epochs=2, batch_size=128, verbose=1)

	print(time.ctime(), '  Predictions .... ')
	train_1 = model.predict_proba(X_train, batch_size=128)
	test_1 = model.predict_proba(X_test)

	train_acc = accuracy(y, train_pred)
	test_acc = accuracy(y_test, test_pred)

	results = '  Results :::: train accuracy = {} and test accuracy = {}'.format(train_acc, test_acc)
	print(time.ctime(), results)
	print('Done')

	"""
	Dump the results of model 1
	"""

	#cPickle.dump(train_1, open('features/train/train_conv1_pretrained.dat', 'wb'))
	#cPickle.dump(test_1, open('features/test/test_conv1_pretrained.dat', 'wb'))

	###################################################
	###################################################
	"""
	print('predictions ...')
	train_pred = mlp.predict(X)
	test_pred = mlp.predict(test)

	train_acc = accuracy(y, train_pred)
	test_acc = accuracy(y_test, test_pred)

	results = 'Results :::: train accuracy = {} and test accuracy = {}'.format(train_acc, test_acc)
	print(results)

	#Generate predictions
	print('Generate predictions')
	test = scaler.transform(test_vec)
	predictions = mlp.predict(test)
	create_csv_submission(np.arange(len(predictions))+1, predictions, 'submissions/' + t)

	DESCRIPTION = [str(t), '	' + EMBD,'	MLPClassifier(hidden_layer_sizes=(100,100,100), verbose=True) with fitting ', '		' + results]
	update_description(DESCRIPTION)
	print('Done')
	"""


if __name__ == "__main__":
	main()