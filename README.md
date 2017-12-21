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
3) run `preprocessing_final.py train_pos_full.txt train_neg_full.txt test.txt`

### `implementation.py`

This script contains the required machine learning algorithms for this project:
- Least Squares (normal equations, GD, SGD)
- Ridge Regression
- Logistic Regression
- Regularized Logistic Regression
As well as some helper functions for the machine learning algorithms compute_sigmoid and so on ...

### `cross_validation.py`

Contains the cross_validation function to perform cross validation on the training set in order to find the best hyperparameters and compare models.


### `proj1_helpers.py`

Contains helpers functions to load the dataset, predict labels and create submission as csv file.


## Generate predictions

Make sure the train.csv and test.csv (should be downloaded from https://www.kaggle.com/c/epfml-higgs/data) are in the data folder, and all py files are in scripts folder, then run 'run.py' by executing the command:
    `python run.py`
