# Tweet Sentiment Analysis
Project 2 for the Machine Learning Course (Fall 2017)

## Project description

The goal of this project is to predict whether a tweet message used to contain a positive smiley ’:)’ or a positive smiley ’:(’ by considering only the remaining text of the tweet. The topic of twitter emotion classification is of great interest in market research, where e.g. a company would like to evaluate customer reaction to a new line of products. The task can be split into two parts.
First, a suitable feature representation of the input text must be found. Multiple approaches for this exist in the literature, and various toolboxes are available for this task. Second, based on this representation a text classifier must be trained.
The aim is to minimize the missclassification error score of the classifier.
### Required libraries
NLTK - http://www.nltk.org/ <br>
WordSegment - https://pypi.python.org/pypi/wordsegment <br>
gensim - https://radimrehurek.com/gensim/ <br>
Fasttext - https://pypi.python.org/pypi/fasttext

### `run.py` [Fasttext]

This script generates the best predictions submited on Kaggle, using Fasttext on the original datasets:
`train_neg.txt`, `train_pos.txt` and `test_data.txt` should be in the `data` folder
- Preprocess the data (Fasttext)
- Generates model
- Generates and saves prediction

### Other Approaches:

### Preprocessing
`scraping.py`: Scrape text data from various online sources <br>
`initial_dict.py`: Compound text data into single dictionary <br>
`preprocessing_final.py`: Contains all the functions to tokenize, check, filter and correct the tweets. <br>
`stop_words.py`: Remove common stop words from the final dictionary and tweets (optional)

How to run:
1) run `scraping.py`
2) run `initial_dict.py`
3) run `preprocessing_final.py train_pos.txt train_neg.txt test.txt`

### Word Embeddings
`word2vec.py`: Generates a Word2vec model using the preprocessed tweets. The result is a vector representation of each tweet. <br> 
`glove_emb.py`: Generate a vector representation of each tweet based on a pre-trained GloVe model.

The resulting files are pickled and saved in the folder: `/data/embeddings`

### Linear Classification
`Train_SGDClassifiers.py`: Train linear hinge, modified huber and logistic regression loss classifiers using stochastic gradient descent. Hyperparameters are evaluated using grid search.
`Train_linearsvc.py`: Train a linear SVC classifier using LinearSVC. Hyperparameters are evaluated using grid search.

The scripts can be run without additional parameters and print the obtained accuracies from cross-validation for each choice of hyperparameters in the terminal.

