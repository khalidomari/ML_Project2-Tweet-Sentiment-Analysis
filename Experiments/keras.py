###############################################################
########################## IMPORTS ############################
###############################################################
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
import numpy as np 
import pickle
import time
import csv
import keras.backend as K
import multiprocessing
import tensorflow as tf

from gensim.models.word2vec import Word2Vec

from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv1D
from keras.optimizers import Adam

from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import RegexpTokenizer

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
	print('Loading files .... ')
	DATA_PATH = '../data/embeddings/'
	GLOVE    = DATA_PATH + 'glove/train_test_splited_glove_data.200d.p'
	WORD2VEC = DATA_PATH + 'word2vec/train_test_splited_word2Vec0.05.p'

	EMBD = 'WORD2VEC'
	[X_train, X_test, y_train, y_test, test_vec] = pickle.load( open(WORD2VEC, 'rb'))

	size = 1000
	X_train = X_train[:size]
	y = y_train[:size]

	print('Scaler fitting ...')
	scaler = StandardScaler()
	scaler.fit(X_train)

	X = X_train
	test = X_test

	print('Scaling the training set ...')
	X = scaler.transform(X_train)
	print('Scaling the testing set ...')
	test = scaler.transform(X_test)


	max_tweet_length = 30
	vector_size = 200
	        
	# Keras convolutional model
	batch_size = 32
	nb_epochs = 100

	model = Sequential()

	model.add(Conv1D(32, kernel_size=3, activation='elu', padding='same', input_shape=(max_tweet_length, vector_size)))
	model.add(Conv1D(32, kernel_size=3, activation='elu', padding='same'))
	model.add(Conv1D(32, kernel_size=3, activation='elu', padding='same'))
	model.add(Conv1D(32, kernel_size=3, activation='elu', padding='same'))
	model.add(Dropout(0.25))

	model.add(Conv1D(32, kernel_size=2, activation='elu', padding='same'))
	model.add(Conv1D(32, kernel_size=2, activation='elu', padding='same'))
	model.add(Conv1D(32, kernel_size=2, activation='elu', padding='same'))
	model.add(Conv1D(32, kernel_size=2, activation='elu', padding='same'))
	model.add(Dropout(0.25))

	model.add(Flatten())

	model.add(Dense(256, activation='tanh'))
	model.add(Dense(256, activation='tanh'))
	model.add(Dropout(0.5))

	model.add(Dense(2, activation='softmax'))

	# Compile the model
	model.compile(loss='categorical_crossentropy',
	              optimizer=Adam(lr=0.0001, decay=1e-6),
	              metrics=['accuracy'])

	# Fit the model
	model.fit(X_train, Y_train,
	          batch_size=batch_size,
	          shuffle=True,
	          epochs=nb_epochs,
	          validation_data=(X_test, Y_test),
	          callbacks=[EarlyStopping(min_delta=0.00025, patience=2)])


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


if __name__ == "__main__":
	main()