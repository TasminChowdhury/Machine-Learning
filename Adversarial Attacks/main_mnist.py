"""
This file trains and saves a 3 layer Dense model on phase component of
permuted MNIST dataset. The pixel2phase is part of the pipeline implemented
in Tensorflow such that attacks can later on use derivative of pixel2phase
transformation. By default models are saved in "Adversarial Attacks\\saved_models\\mnistdense".

Example of usage: python mnist.py --nb_epochs=100 --permutation_seed_list=100,101,102
                trains 3 models with permutation seeds 100,101,102.
Typical accuracy of models should be between 95% to 96%.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags

import logging
import os
from utils_tf import LossCrossEntropy
from cleverhans.utils import set_log_level
from cleverhans.utils_tf import train, model_eval
from model_structure import ModelDense
from utils import permute

FLAGS = flags.FLAGS


def pptrain_mnist(permutation_seed,
                  nb_epochs=6,
                  batch_size=128,
                  learning_rate=0.001,
                  augmentation=False,
                  model_path=os.path.join("Adversarial Attacks", "saved_models", "mnistdense")):
    """
    Train and save a permuted phase model for MNIST
    :param permutation_seed: permutation_seed used to permute images
    :param nb_epochs: number of epochs to train model
    :param batch_size: size of training batches
    :param learning_rate: learning rate for training
    :param augmentation: boolean denoting whether or not to use data augmentation in training
    :param model_path: path to the model file. Note that permutation_seed will be added to the path.
    :return:
    """
    directory = os.path.join(model_path, str(permutation_seed))
    if not os.path.exists(directory):
        os.makedirs(directory)
    model_path = os.path.join(directory, 'mnist')

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    # Create TF session
    sess = tf.Session()
    print("Created TensorFlow session.")

    set_log_level(logging.DEBUG)

    # Get MNIST data and permute with the permutation_seed
    print('augmentation =', augmentation)
    if not augmentation:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    else:
        print('data ')
        # augment data, save it on disc for later reference and load it.
        path_to_training_data = os.path.join(os.getcwd(), 'mnist_png', 'training')
        from utils import augment, load_images_from_folder
        augment(path_to_training_data, nb_samples=60000)
        folder = os.path.join(os.getcwd(), 'mnist_png')
        x_train, y_train = load_images_from_folder(os.path.join(folder, 'training'))
        x_test, y_test = load_images_from_folder(os.path.join(folder, 'testing'))
    x_train = np.expand_dims(x_train, axis=3).astype('float32') / 255
    x_test = np.expand_dims(x_test, axis=3).astype('float32') / 255
    x_train = permute(x_train, permutation_seed)
    x_test = permute(x_test, permutation_seed)

    nb_classes = 10
    y_train = tf.keras.utils.to_categorical(y_train, nb_classes)
    y_test = tf.keras.utils.to_categorical(y_test, nb_classes)
    print("x_train shape =", x_train.shape)
    print('y_train shape =', y_train.shape)
    print("x_test shape =", x_test.shape)

    # Obtain Image Parameters
    img_rows, img_cols, nb_channels = x_train.shape[1:4]

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols, nb_channels))
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))

    # Define TF model graph
    reg = 5e-3
    model = ModelDense('model1', nb_classes, reg=reg)
    preds = model.get_logits(x)
    loss = LossCrossEntropy(model, smoothing=0.)
    print("Defined TensorFlow model graph.")

    ###########################################################################
    # Training the model using TensorFlow
    ###########################################################################

    # Train an MNIST model
    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'train_dir': os.path.join(*os.path.split(model_path)[:-1]),
        'filename': os.path.split(model_path)[-1]
    }

    eval_params = {'batch_size': batch_size}
    def evaluate():
        train_acc = model_eval(sess, x, y, preds, x_train, y_train, args=eval_params)
        test_acc = model_eval(sess, x, y, preds, x_test, y_test, args=eval_params)
        print('Train accuracy: {0} ------ Test accuracy: {1}'.format(train_acc, test_acc))

    rng = np.random.RandomState([2019, 1, 1])
    optimizer = tf.train.AdagradOptimizer(learning_rate=train_params['learning_rate'])
    train(sess, loss, x, y, x_train, y_train, args=train_params, save=True, rng=rng, evaluate=evaluate, optimizer=optimizer)

    # Evaluate the accuracy of the MNIST model on legitimate test examples
    accuracy = model_eval(sess, x, y, preds, x_test, y_test, args=eval_params)
    print('Test accuracy on legitimate test examples: {0}'.format(accuracy))

    # Close TF session
    sess.close()


def main(argv=None):
    for permutation_str in FLAGS.permutation_seed_list:
        tf.reset_default_graph()
        print('-'*50)
        print('-'*50)
        print('Training model for seed=', permutation_str)
        permutation_seed = None if permutation_str.upper() == 'NONE' else int(permutation_str)
        pptrain_mnist(permutation_seed=permutation_seed,
                      nb_epochs=FLAGS.nb_epochs,
                      batch_size=FLAGS.batch_size,
                      learning_rate=FLAGS.learning_rate,
                      augmentation=FLAGS.augmentation,
                      model_path=FLAGS.model_path)


if __name__ == '__main__':
    flags.DEFINE_list('permutation_seed_list', ['None'], 'Permutation seeds for images')
    flags.DEFINE_integer('nb_epochs', 6, 'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
    flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')
    flags.DEFINE_boolean('augmentation', False, 'whether or not using data augmentation for training')
    flags.DEFINE_string('model_path', os.path.join("Adversarial Attacks", "saved_models", "mnistdense"),
                        'Path to save or load the model file')

    tf.app.run()
