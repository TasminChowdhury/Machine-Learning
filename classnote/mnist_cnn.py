# Simple CNN for the MNIST Dataset
import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
# fix dimension ordering issue
from keras import backend as K
K.set_image_dim_ordering('th')
"""
What is a "backend"?
Keras is a model-level library, providing high-level building blocks for developing deep learning models. It does not handle itself low-level operations such as tensor products, convolutions and so on. Instead, it relies on a specialized, well-optimized tensor manipulation library to do so, serving as the "backend engine" of Keras. Rather than picking one single tensor library and making the implementation of Keras tied to that library, Keras handles the problem in a modular way, and several different backend engines can be plugged seamlessly into Keras.
At this time, Keras has three backend implementations available: the TensorFlow backend, the Theano backend, and the CNTK backend.
TensorFlow is an open-source symbolic tensor manipulation framework developed by Google.
Theano is an open-source symbolic tensor manipulation framework developed by LISA Lab at Université de Montréal.
CNTK is an open-source toolkit for deep learning developed by Microsoft.
"""

#th" format means that the convolutional kernels will have the shape (depth, input_depth, rows, cols)
#"tf" format means that the convolutional kernels will have the shape (rows, cols, input_depth, depth)


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# http://yann.lecun.com/exdb/mnist/

"""
When one learns how to program, there's a tradition that the first thing you do is print "Hello World." Just like programming has Hello World, machine learning has MNIST.
MNIST is a simple computer vision dataset. It consists of images of handwritten digits like these:
 
It also includes labels for each image, telling us which digit it is. For example, the labels for the above images are 5, 0, 4, and 1.
In this tutorial, we're going to train a model to look at images and predict what digits they are. Our goal isn't to train a really elaborate model that achieves state-of-the-art performance -- although we'll give you code to do that later! -- but rather to dip a toe into using TensorFlow. As such, we're going to start with a very simple model, called a Softmax Regression.
The actual code for this tutorial is very short, and all the interesting stuff happens in just three lines. However, it is very important to understand the ideas behind it: both how TensorFlow works and the core machine learning concepts. Because of this, we are going to very carefully work through the code.
"""

# reshape to be [samples][channels][width][height]
# flatten 28*28 images to a 784 vector for each image
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')
"""
mnist.load_data() supplies the MNIST digits with structure (nb_samples, 28, 28) i.e. with 2 dimensions per example representing a greyscale image 28x28.
The Convolution2D layers in Keras however, are designed to work with 3 dimensions per example. They have 4-dimensional inputs and outputs. This covers colour images (nb_samples, nb_channels, width, height), but more importantly, it covers deeper layers of the network, where each example has become a set of feature maps i.e. (nb_samples, nb_features, width, height).
The greyscale image for MNIST digits input would either need a different CNN layer design (or a param to the layer constructor to accept a different shape), or the design could simply use a standard CNN and you must explicitly express the examples as 1-channel images. The Keras team chose the latter approach, which needs the re-shape.
"""

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
# define a simple CNN model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
"""
we de
fine our neural network model. Convolutional neural networks are more complex
than standard Multilayer Perceptrons, so we will start by using a simple structure to begin
with that uses all of the elements for state-of-the-art results. Below summarizes the network
architecture.
1. The 
rst hidden layer is a convolutional layer called a Conv2D. The layer has 32 feature
maps, with the size of 5  5 and a recti
er activation function. This is the input layer,
expecting images with the structure outline above.
2. Next we de
ne a pooling layer that takes the maximum value called MaxPooling2D. It is
con
gured with a pool size of 2  2.
3. The next layer is a regularization layer using dropout called Dropout. It is con
gured to
randomly exclude 20% of neurons in the layer in order to reduce over
tting.
4. Next is a layer that converts the 2D matrix data to a vector called Flatten. It allows the
output to be processed by standard fully connected layers.
5. Next a fully connected layer with 128 neurons and recti
er activation function is used.
6. Finally, the output layer has 10 neurons for the 10 classes and a softmax activation function
to output probability-like predictions for each class.
As before, the model is trained using logarithmic loss and the ADAM gradient descent
algorithm.
"""


# build the model
model = baseline_model()
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, batch_size=200)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
