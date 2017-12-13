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

# Set random seed (for reproducibility)
np.random.seed(1000)

# Select whether using Keras with or without GPU support
# See: https://stackoverflow.com/questions/40690598/can-keras-with-tensorflow-backend-be-forced-to-use-cpu-or-gpu-at-will
use_gpu = True

config = tf.ConfigProto(intra_op_parallelism_threads=multiprocessing.cpu_count(), 
                        inter_op_parallelism_threads=multiprocessing.cpu_count(), 
                        allow_soft_placement=True, 
                        device_count = {'CPU' : 1, 
                                        'GPU' : 1 if use_gpu else 0})

session = tf.Session(config=config)
K.set_session(session)

dataset_location = '/twitter/dataset.csv'
model_location = '/twitter/model/'

corpus = []
labels = []

# Parse tweets and sentiments
with open(dataset_location, 'r', encoding='utf-8') as df:
    for i, line in enumerate(df):
        if i == 0:
            # Skip the header
            continue

        parts = line.strip().split(',')
        
        # Sentiment (0 = Negative, 1 = Positive)
        labels.append(int(parts[1].strip()))
        
        # Tweet
        tweet = parts[3].strip()
        if tweet.startswith('"'):
            tweet = tweet[1:]
        if tweet.endswith('"'):
            tweet = tweet[::-1]
        
        corpus.append(tweet.strip().lower())
        
print('Corpus size: {}'.format(len(corpus)))

# Tokenize and stem
tkr = RegexpTokenizer('[a-zA-Z0-9@]+')
stemmer = LancasterStemmer()

tokenized_corpus = []

for i, tweet in enumerate(corpus):
    tokens = [stemmer.stem(t) for t in tkr.tokenize(tweet) if not t.startswith('@')]
    tokenized_corpus.append(tokens)
    
# Gensim Word2Vec model
vector_size = 512
window_size = 10

# Create Word2Vec
word2vec = Word2Vec(sentences=tokenized_corpus,
                    size=vector_size, 
                    window=window_size, 
                    negative=20,
                    iter=50,
                    seed=1000,
                    workers=multiprocessing.cpu_count())

# Copy word vectors and delete Word2Vec model  and original corpus to save memory
X_vecs = word2vec.wv
del word2vec
del corpus

# Train subset size (0 < size < len(tokenized_corpus))
train_size = 1000000

# Test subset size (0 < size < len(tokenized_corpus) - train_size)
test_size = 100000

# Compute average and max tweet length
avg_length = 0.0
max_length = 0

for tweet in tokenized_corpus:
    if len(tweet) > max_length:
        max_length = len(tweet)
    avg_length += float(len(tweet))
    
print('Average tweet length: {}'.format(avg_length / float(len(tokenized_corpus))))
print('Max tweet length: {}'.format(max_length))

# Tweet max length (number of tokens)
max_tweet_length = 15

# Create train and test sets
# Generate random indexes
indexes = set(np.random.choice(len(tokenized_corpus), train_size + test_size, replace=False))

X_train = np.zeros((train_size, max_tweet_length, vector_size), dtype=K.floatx())
Y_train = np.zeros((train_size, 2), dtype=np.int32)
X_test = np.zeros((test_size, max_tweet_length, vector_size), dtype=K.floatx())
Y_test = np.zeros((test_size, 2), dtype=np.int32)

for i, index in enumerate(indexes):
    for t, token in enumerate(tokenized_corpus[index]):
        if t >= max_tweet_length:
            break
        
        if token not in X_vecs:
            continue
    
        if i < train_size:
            X_train[i, t, :] = X_vecs[token]
        else:
            X_test[i - train_size, t, :] = X_vecs[token]
            
    if i < train_size:
        Y_train[i, :] = [1.0, 0.0] if labels[index] == 0 else [0.0, 1.0]
    else:
        Y_test[i - train_size, :] = [1.0, 0.0] if labels[index] == 0 else [0.0, 1.0]
        
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