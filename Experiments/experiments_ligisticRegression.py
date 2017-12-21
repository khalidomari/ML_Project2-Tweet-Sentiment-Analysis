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

###############################################################
########################## FUNCTIONS ##########################
###############################################################
def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    param ids: list of int, event ids associated with each prediction
    param y_pred: list of int, predicted class labels
    param name: string, name of .csv output file to be created
    """
    name += '.csv'
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})

def accuracy(y1, y2):
	'''
	Calculate accuracy between y1 and y2
	param y1, y2: list of int
	return: the accuracy
	'''
	return 100 - np.sum(np.abs(np.asarray(y1) - np.asarray(y2))/2)*100/len(y1)

def update_description(lines):
	'''
	Store experiment information by updating the decription.txt file
	param lines: list of string
	'''
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
	WORD2VEC = DATA_PATH + 'word2vec/train_test_splited_word2Vec_ s100_w3_tests0.05.p'

	EMBD = 'WORD2VEC'
	[X_train, X_test, y_train, y_test, test_vec] = pickle.load( open(WORD2VEC, 'rb'))

	size = -1
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

	print(np.shape(X))

	model = LogisticRegression(verbose=1, random_state=6, solver='sag')

	#model = LogisticRegression(verbose=1)
	print('model fitting ...')
	model.fit(X,y)
	t = str(time.time())
	pickle.dump(model, open( "models/model." + t + EMBD + ".model", "wb" ))
	print('model pickled ....')

	print('predictions ...')
	train_pred = model.predict(X)
	test_pred = model.predict(test)

	train_acc = accuracy(y, train_pred)
	test_acc = accuracy(y_test, test_pred)

	results = 'Results :::: train accuracy = {} and test accuracy = {}'.format(train_acc, test_acc)
	print(results)

	#Generate predictions
	print('Generate predictions')
	test = scaler.transform(test_vec)
	predictions = model.predict(test)
	create_csv_submission(np.arange(len(predictions))+1, predictions, 'submissions/' + t)

	DESCRIPTION = [str(t), '	' + EMBD,'	modelClassifier(hidden_layer_sizes=(100,100,100), verbose=True) with fitting ', '		' + results]
	update_description(DESCRIPTION)
	print('Done')


if __name__ == "__main__":
	main()