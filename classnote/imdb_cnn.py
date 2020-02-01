# CNN for the IMDB problem
#  Sentiment analysis
"""
The dataset used in this project is the Large Movie Review Dataset often referred to as the
IMDB dataset1. The IMDB dataset contains 50,000 highly-polar movie reviews (good or bad)
for training and the same amount again for testing. The problem is to determine whether a
given movie review has a positive or negative sentiment.
"""
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
# The Internet Movie Database is an online database of information related to films, television programs and video games, including cast, production crew, fictional characters, biographies, plot summaries
# pad dataset to a maximum review length in words
max_words = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)
# create the model
model = Sequential()
model.add(Embedding(top_words, 32, input_length=max_words))
# # the model will take as input an integer matrix of size (batch, input_length).
# the largest integer (i.e. word index) in the input should be no larger than 4999 (vocabulary size).
# now model.output_shape == (None, 10, 32), where None is the batch dimension.
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')) 
"""
This layer creates a convolution kernel that is convolved
with the layer input over a single spatial (or temporal) dimension
to produce a tensor of outputs.
padding: One of `"valid"`, `"causal"` or `"same"` (case-insensitive).
`"valid"` means "no padding".
"same"` results in padding the input such that
the output has the same length as the original input.
`"causal"` results in causal (dilated) convolutions
 """
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2, batch_size=128, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
