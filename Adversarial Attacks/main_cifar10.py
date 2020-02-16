"""
This file trains and saves a 3 layer Dense model on phase component of
permuted CIFAR10 dataset. The pixel2phase is part of the pipeline implemented
in Tensorflow such that attacks can later on use derivative of pixel2phase
transformation. By default models are saved in "Adversarial Attacks\\saved_models\\cifar10dense".

Example of usage: python main_cifar10.py --nb_epochs=100 --permutation_seed_list=100,101,102
                trains 3 models with permutation seeds 100,101,102.
Typical accuracy of models should be between 44% to 46%.
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


def pptrain_cifar10(permutation_seed,
                    nb_epochs=6,
                    batch_size=128,
                    learning_rate=0.001,
                    version=1,
                    nb_res_blocks=3,
                    model_path=os.path.join("Adversarial Attacks", "saved_models", "cifar10dense")):
    """
    Train and save a permuted phase model for MNIST
    :param permutation_seed: permutation_seed used to permute images
    :param nb_epochs: number of epochs to train model
    :param batch_size: size of training batches
    :param learning_rate: learning rate for training
    :param version: version of resnet (available versions are 1 and 2)
    :param nb_res_blocks: number of residual blocks
    :param model_path: path to the model file. Note that permutation_seed will be added to the path.
    :return:
    """
    directory = os.path.join(model_path, str(permutation_seed))
    if not os.path.exists(directory):
        os.makedirs(directory)
    model_path = os.path.join(directory, 'cifar10')

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    # Create TF session
    sess = tf.Session()
    # writer = tf.summary.FileWriter(os.path.join('graphs'), sess.graph)
    print("Created TensorFlow session.")

    set_log_level(logging.DEBUG)

    # Get MNIST data and permute with the permutation_seed
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
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
    # model = ModelResNet('model1', nb_classes=nb_classes, n=nb_res_blocks, version=version)
    reg = 5e-2
    model = ModelDense('model1', nb_classes=nb_classes, reg=reg)
    preds = model.get_logits(x)
    loss = LossCrossEntropy(model, smoothing=0.)

    # tf.summary.scalar('accuracy', tf.metrics.accuracy(tf.argmax(y), tf.argmax(preds)))
    # tf.summary.scalar('loss', loss)
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

    rng = np.random.RandomState([2017, 8, 30])
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
        pptrain_cifar10(permutation_seed=permutation_seed,
                        nb_epochs=FLAGS.nb_epochs,
                        batch_size=FLAGS.batch_size,
                        learning_rate=FLAGS.learning_rate,
                        version=FLAGS.version,
                        nb_res_blocks=FLAGS.nb_res_blocks,
                        model_path=FLAGS.model_path)


if __name__ == '__main__':
    flags.DEFINE_list('permutation_seed_list', ['None'], 'Permutation seeds for images')
    flags.DEFINE_integer('nb_epochs', 6, 'Number of epochs to train model')
    flags.DEFINE_integer('version', 1, 'Version of ResNet')
    flags.DEFINE_integer('nb_res_blocks', 3, 'Number of residual blocks in the ResNet')
    flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
    flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')
    flags.DEFINE_string('model_path', os.path.join("Adversarial Attacks", "saved_models", "cifar10dense"),
                        'Path to save or load the model file')

    tf.app.run()
