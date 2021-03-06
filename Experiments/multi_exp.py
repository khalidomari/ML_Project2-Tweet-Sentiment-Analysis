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
def update(i):
	print(i , ':::::::::::::::::::::::::::::::::::::::::::')
	return i+1

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
	
	#size = 3600
	#X_train = X_train[:size]
	#y_train = y_train[:size]


	idx_list = np.split(np.arange(len(X_train)), 9)

	test = X_test
	final_pred = []
	final_y_test = []
	i = 0
	for idx in idx_list:
		print(len(idx))
		i = update(i)
		X = np.asarray(X_train)[idx]
		y = np.asarray(y_train)[idx]

		print('Scaler fitting ...')
		scaler = StandardScaler()
		scaler.fit(X)

		print('Scaling the training set ...')
		X = scaler.transform(X)
		print('Scaling the testing set ...')
		test = scaler.transform(X_test)


		#mlp = MLPClassifier(hidden_layer_sizes=(200,200), verbose=True)
		#mlp = LinearSVC(random_state=0, verbose=1)
		#mlp = MLPClassifier(hidden_layer_sizes=(100,100,100), verbose=True)
		mlp = MLPClassifier(hidden_layer_sizes=(30,30,30), verbose=True)
		#mlp = LogisticRegression(verbose=1)
		print('model fitting ...')
		mlp.fit(X,y)
		#pickle.dump(mlp, open( "models/mlp." + t + EMBD + ".model", "wb" ))
		print('model pickled ....')

		print('predictions ...')
		train_pred = mlp.predict(X)
		test_pred = mlp.predict(test)

		train_acc = accuracy(y, train_pred)
		test_acc = accuracy(y_test, test_pred)
		final_y_test.append(test_pred)

		results = 'Results :::: train accuracy = {} and test accuracy = {}'.format(train_acc, test_acc)
		print(results)

		#Generate predictions
		print('Generate predictions')
		test = scaler.transform(test_vec)
		predictions = mlp.predict(test)
		final_pred.append(predictions)
		#DESCRIPTION = [str(t), '	' + EMBD,'	MLPClassifier(hidden_layer_sizes=(100,100,100), verbose=True) with fitting ', '		' + results]
		#update_description(DESCRIPTION)
		print('Done')

	t = str(time.time())

	final_y_test = [np.asarray(0*final_y_test[0]) + np.asarray(pred) for pred in final_y_test][0]
	final_y_test = [-1 if p<0 else 1 for p in final_y_test]
	test_acc = accuracy(y_test, final_y_test)
	results = 'Results :::: test accuracy = {}'.format( test_acc)
	print(results)

	predictions = [np.asarray(0*final_pred[0]) + np.asarray(pred) for pred in final_pred][0]
	predictions = [-1 if p<0 else 1 for p in predictions]

	create_csv_submission(np.arange(len(predictions))+1, predictions, 'submissions/' + t)


if __name__ == "__main__":
	main()