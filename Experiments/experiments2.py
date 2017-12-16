###############################################################
########################## IMPORTS ############################
###############################################################
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
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

def mlp_model(X, y, test, y_test, test_f, nlayers = 1000, EMBD='WORD2VEC'):

	mlp = MLPClassifier(hidden_layer_sizes=(nlayers,nlayers), verbose=True, random_state=1, activation='relu', learning_rate='constant', solver='sgd', learning_rate_init =0.001, max_iter =500)
	#mlp = RandomForestClassifier(max_depth=20000, random_state=0)
	print('model fitting ...')
	mlp.fit(X,y)
	t = str(time.time())
	pickle.dump(mlp, open( "models/mlp." + t + EMBD + ".model", "wb" ))
	print('model pickled ....')

	print('predictions ...')
	train_pred = mlp.predict(X)
	test_pred = mlp.predict(test)

	train_acc = accuracy(y, train_pred)
	test_acc = accuracy(y_test, test_pred)

	results = 'Results :::: train accuracy = {} and test accuracy = {}'.format(train_acc, test_acc)
	print(results)

	#Generate predictions
	print('Generate predictions')
	
	predictions = mlp.predict(test_f)
	create_csv_submission(np.arange(len(predictions))+1, predictions, 'submissions/' + t)

	DESCRIPTION = ['{}		{}'.format(t,EMBD),'	hidden_layer_sizes={}'.format(nlayers), '		{}'.format(results)]
	update_description(DESCRIPTION)
	print('Done')

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
	test_f = scaler.transform(test_vec)

	#mlp = MLPClassifier(hidden_layer_sizes=(200,200), verbose=True)
	#mlp = LinearSVC(random_state=0, verbose=1)
	#mlp = MLPClassifier(hidden_layer_sizes=(100,100,100), verbose=True)
	hd_size_list = [1000, 500, 200, 100]
	mlp_model(X=X, y=y, test=test, test_f=test_f, y_test=y_test)


if __name__ == "__main__":
	main()