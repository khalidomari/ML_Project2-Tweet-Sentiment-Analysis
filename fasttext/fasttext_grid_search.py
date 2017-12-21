###############################################################
########################## IMPORTS ############################
###############################################################

import fasttext
import numpy as np

###############################################################
########################## FUNCTIONS ############################
###############################################################

def split_data(X,X_unlabeled, ratio=0.1):
    """
    splits dataset to training and testing set with he corresponding ratio
    params : X : list of tweets with labels.
               X_unlabeled : list of tweets without labels.
               ratio : ratio of the split
    Returns : tweets of train with labels, tweets of test with labels and tweets of train without labels.
               
    """
    np.random.seed(50)
    X=np.array(X)
    X_unlabeled=np.array(X_unlabeled)
    N_test=len(X)*ratio
    N_test = int(N_test)
    idx_list = np.arange(len(X))
    np.random.shuffle(idx_list)
    test_idx = idx_list[:N_test]
    train_idx = idx_list[N_test:]
    
    return  X[train_idx],X[test_idx],X_unlabeled[test_idx],X_unlabeled[train_idx]

def load_file(path):
    
    return [line.rstrip('\n').lower() for line in open(path, encoding = 'utf8')]

def write_to_file(tweet_list, file):
    """
    Writes a text file in the path file passed as paramter
    params: tweet_list : list of tweets as text.
               file : path in which the file will be written
               
    """
    path = file
    f = open(path, 'w', encoding='utf-8') 
    
    #Iterating over tweets.
    for tweet in tweet_list:
        
        #write each tweet to file
        f.write(tweet + '\n')
    f.close()
    print('the file has been successfully created in :: ', path)

def grid_search(dims, lr, train, full_data, test, test_labels,epoch,ngrams,ws):
    """
    Performs a grid seach over the parameters dims and lr.
    params: dims : list of dimension parameters
               lr : list of learning rate parameters
    Returns : Best parameters and best accuracy. 
    """
    
    #Initialzing best accuracy and best parameters
    best_accracy=0.0
    best_params = (0.0,0.0)
    
    #preprocessing testing set.
    actual = [1 if '__label__positive' in t else -1 for t in test_labels]

    #Iterating over paramters
    for grams in ngrams : 
        print('ngrams = ',grams)
        for w in ws :
            print('ws = ',w)
            for k in epoch :
                print('epoch = ', str(k))
                for i in dims : 
                    print('dim = ', str(i))
                    for j in lrs : 
                        print('learning rate = ', str(j)) 

                        #writing files
                        write_to_file(train, 'train.txt')
                        write_to_file(full_data,'data.txt')

                        #building model.
                        model = fasttext.skipgram('data.txt' , 'model',dim=i,lr=j,epoch=k,word_ngrams=grams,ws=w)
                        classifier = fasttext.supervised('train.txt', 'model', label_prefix='__label__')
                        labels = classifier.predict(test)
                        pred = [1 if t==['positive'] else -1 for t in labels]

                        #Computing accuracy.
                        accuracy = 1 - np.mean(np.abs(np.array(pred )- np.array(actual))/2)
                        print('accuracy = '+str(accuracy))
                        if(accuracy > best_accracy) : 
                            best_accracy = accuracy
                            best_params=(i,j,k,w,grams)
    return best_accracy,best_params

def main() : 
#Choosing parameter to perform grid search on
    dims=[150,175,200]
    lrs=[0.01,0.05,0.075]
    epoch=[3,6]
    ngrams=[2,3]
    ws=[4]

    #Performing the grid search for best parameters of fasttext on our data.
    best_accuracy, best_params = grid_search(dims, lrs , train,full_data,test,test_labels,epoch,ngrams,ws)
    print(best_accuracy, best_params)

if __name__ == "__main__":
    main()