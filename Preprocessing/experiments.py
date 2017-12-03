import numpy as np 
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import time


def accuracy(y1, y2):
	return 100 - np.sum(np.abs(np.asarray(y1) - np.asarray(y2))/2)*100/len(y1)

def main():
	[X_train, X_test, y, y_test] = pickle.load( open('train_test_splited_word2Vec0.05.p', 'rb'))
	scaler = StandardScaler()
	scaler.fit(X_train)

	X = scaler.transform(X_train)
	test = scaler.transform(X_test)

	mlp = MLPClassifier(hidden_layer_sizes=(30,30,30))
	mlp.fit(X,y)

	train_pred = mlp.predict(X)
	test_pred = mlp.predict(test)

	train_acc = accuracy(y, train_pred)
	test_acc = accuracy(y_test, test_pred)

	results = 'Results :::: train accuracy = {} and test accuracy = {}'.format(train_acc, test_acc)
	file = "models/mlp." + str(time.time()) + ".model"
	pickle.dump(mlp, open( file, "wb" ))
	print(results)

if __name__ == "__main__":
	main()