#!/usr/bin/env python3
import os
import numpy
from numpy import random
import random as random2
import scipy
import matplotlib
import mnist
import pickle
matplotlib.use('agg')
from matplotlib import pyplot

import tensorflow as tf
mnist = tf.keras.datasets.mnist
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
import time
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Activation

### hyperparameter settings and other constants
batch_size = 128
num_classes = 10
epochs = 10
mnist_input_shape = (28, 28, 1)
d1 = 1024
d2 = 256
alpha = 0.1
beta = 0.9
alpha_adam = 0.001
rho1 = 0.99
rho2 = 0.999
### end hyperparameter settings


# load the MNIST dataset using TensorFlow/Keras
def load_MNIST_dataset():
    mnist = tf.keras.datasets.mnist
    (Xs_tr, Ys_tr), (Xs_te, Ys_te) = mnist.load_data()
    Xs_tr = Xs_tr / 255.0
    Xs_te = Xs_te / 255.0
    Xs_tr = Xs_tr.reshape(Xs_tr.shape[0], 28, 28, 1) # 28 rows, 28 columns, 1 channel
    Xs_te = Xs_te.reshape(Xs_te.shape[0], 28, 28, 1)
    return (Xs_tr, Ys_tr, Xs_te, Ys_te)


# evaluate a trained model on MNIST data, and print the usual output from TF
#
# Xs        examples to evaluate on
# Ys        labels to evaluate on
# model     trained model
#
# returns   tuple of (loss, accuracy)
def evaluate_model(Xs, Ys, model):
    (loss, accuracy) = model.evaluate(Xs, Ys)
    return (loss, accuracy)


# train a fully connected two-hidden-layer neural network on MNIST data using SGD, and print the usual output from TF
#
# Xs        training examples
# Ys        training labels
# d1        the size of the first layer
# d2        the size of the second layer
# alpha     step size parameter
# beta      momentum parameter (0.0 if no momentum)
# B         minibatch size
# epochs    number of epochs to run
#
# returns   a tuple of
#   model       the trained model (should be of type tensorflow.python.keras.engine.sequential.Sequential)
#   history     the history of training returned by model.fit (should be of type tensorflow.python.keras.callbacks.History)
def train_fully_connected_sgd(Xs, Ys, d1, d2, alpha, beta, B, epochs):
    model = Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=mnist_input_shape))
    model.add(Dense(d1, activation='relu'))
    model.add(Dense(d2, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=SGD(learning_rate=alpha, momentum=beta),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy']
    )
    history = model.fit(Xs, Ys, epochs=epochs, batch_size=B, verbose=0, validation_split=0.1)
    return model, history

# train a fully connected two-hidden-layer neural network on MNIST data using Adam, and print the usual output from TF
#
# Xs        training examples
# Ys        training labels
# d1        the size of the first layer
# d2        the size of the second layer
# alpha     step size parameter
# rho1      first moment decay parameter
# rho2      second moment decay parameter
# B         minibatch size
# epochs    number of epochs to run
#
# returns   a tuple of
#   model       the trained model (should be of type tensorflow.python.keras.engine.sequential.Sequential)
#   history     the history of training returned by model.fit (should be of type tensorflow.python.keras.callbacks.History)
def train_fully_connected_adam(Xs, Ys, d1, d2, alpha, rho1, rho2, B, epochs):
    model = Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=mnist_input_shape))
    model.add(Dense(d1, activation='relu'))
    model.add(Dense(d2, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=alpha, beta_1=rho1, beta_2=rho2),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    history = model.fit(Xs, Ys, epochs=epochs, batch_size=B, verbose=0, validation_split=0.1)
    return model, history

# train a fully connected two-hidden-layer neural network with Batch Normalization on MNIST data using SGD, and print the usual output from TF
#
# Xs        training examples
# Ys        training labels
# d1        the size of the first layer
# d2        the size of the second layer
# alpha     step size parameter
# beta      momentum parameter (0.0 if no momentum)
# B         minibatch size
# epochs    number of epochs to run
#
# returns   a tuple of
#   model       the trained model (should be of type tensorflow.python.keras.engine.sequential.Sequential)
#   history     the history of training returned by model.fit (should be of type tensorflow.python.keras.callbacks.History)
def train_fully_connected_bn_sgd(Xs, Ys, d1, d2, alpha, beta, B, epochs):
    model = Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=mnist_input_shape))
    model.add(Dense(d1))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(d2))
    model.add(BatchNormalization())
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=SGD(lr=alpha, momentum=beta),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy']
    )
    history = model.fit(Xs, Ys, epochs=epochs, batch_size=B, verbose=0, validation_split=0.1)
    return model, history

# train a convolutional neural network on MNIST data using SGD, and print the usual output from TF
#
# Xs        training examples
# Ys        training labels
# alpha     step size parameter
# rho1      first moment decay parameter
# rho2      second moment decay parameter
# B         minibatch size
# epochs    number of epochs to run
#
# returns   a tuple of
#   model       the trained model (should be of type tensorflow.python.keras.engine.sequential.Sequential)
#   history     the history of training returned by model.fit (should be of type tensorflow.python.keras.callbacks.History)
def train_CNN_sgd(Xs, Ys, alpha, rho1, rho2, B, epochs):
    model = Sequential()
    model.add(Conv2D(32, (5, 5), activation='relu', input_shape=mnist_input_shape))
    model.add(MaxPool2D((2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu', input_shape=mnist_input_shape))
    model.add(MaxPool2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=alpha, beta_1=rho1, beta_2=rho2),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    history = model.fit(Xs, Ys, epochs=epochs, batch_size=B, verbose=0, validation_split=0.1)

    return model, history


def plot_history(histories, key):
    """ histories is a list of tuples  of the form [(history, label)...]"""
    x_axis = numpy.arange(len(histories[0][0].history['loss']))
    fig, (ax1) = pyplot.subplots(4, 2, figsize=(10, 8))
    colors = ['orange', 'red', 'green', 'blue', 'yellow']
    for ind, (history, label) in enumerate(histories):
        ax1.plot(x_axis, history.history[key], color=colors[ind], label=label)
    ax1.set_title(key)
    ax1.legend()
    pyplot.savefig(key + ".png")

def plot_loss(axis, i, j, history, test, label):
    x_axis = numpy.arange(epochs) + 1
    axis[i][j].plot(x_axis, history.history['loss'], color='orange', label='loss')
    axis[i][j].plot(x_axis, history.history['val_loss'], color='green', label='validation loss')
    axis[i][j].plot(x_axis, [test] * 10, color = 'blue', label='test loss')
    axis[i][j].set_title(label)
    axis[i][j].legend()

def plot_accuracy(axis, i, j, history, test, label):
    x_axis = numpy.arange(epochs) + 1
    axis[i][j].plot(x_axis, history.history['accuracy'], color='orange', label='accuracy')
    axis[i][j].plot(x_axis, history.history['val_accuracy'], color='green', label='validation accuracy')
    axis[i][j].plot(x_axis, [test] * 10, color='blue', label='test accuracy')
    axis[i][j].set_title(label)
    axis[i][j].legend()


def part1_fully_connected_SGD(Xs_tr, Ys_tr, Xs_te, Ys_te, axis):
    before_time = time.time()
    model, history = train_fully_connected_sgd(Xs_tr, Ys_tr, d1, d2, alpha, 0.0, batch_size, epochs)
    after_time = time.time() - before_time
    test_loss, test_acc = model.evaluate(Xs_te, Ys_te)
    print('===========part1_fully_connected_SGD===============')
    print('test loss: ', test_loss)
    print('test accuracy:, ', test_acc)
    print('time: ', after_time)
    plot_accuracy(axis, 0, 0, history, test_acc, 'SGD')
    plot_loss(axis, 0, 1, history, test_loss, 'SGD')
    return history


def part1_fully_connected_momentum(Xs_tr, Ys_tr, Xs_te, Ys_te, axis):
    before_time = time.time()
    model, history = train_fully_connected_sgd(Xs_tr, Ys_tr, d1, d2, alpha, 0.9, batch_size, epochs)
    after_time = time.time() - before_time
    test_loss, test_acc = model.evaluate(Xs_te, Ys_te)
    print('===========part1_fully_connected_momentum===============')
    print('test loss: ', test_loss)
    print('test accuracy:, ', test_acc)
    print('time: ', after_time)
    plot_accuracy(axis, 1, 0, history, test_acc, 'Momentum ')
    plot_loss(axis, 1, 1, history, test_loss, 'Momentum')
    return history

def part1_fully_connected_adam(Xs_tr, Ys_tr, Xs_te, Ys_te, axis):
    before_time = time.time()
    model, history = train_fully_connected_adam(Xs_tr, Ys_tr, d1, d2, 0.001, rho1, rho2, batch_size, epochs)
    after_time = time.time() - before_time
    test_loss, test_acc = model.evaluate(Xs_te, Ys_te)
    print('===========part1_fully_connected_adam===============')
    print('test loss: ', test_loss)
    print('test accuracy:, ', test_acc)
    print('time: ', after_time)
    plot_accuracy(axis, 2, 0, history, test_acc, 'Adam ')
    plot_loss(axis, 2, 1, history, test_loss, 'Adam')

    return history

def part1_fully_connected_BN(Xs_tr, Ys_tr, Xs_te, Ys_te, axis):
    before_time = time.time()
    model, history = train_fully_connected_bn_sgd(Xs_tr, Ys_tr, d1, d2, alpha, 0.9, batch_size, epochs)
    after_time = time.time() - before_time
    test_loss, test_acc = model.evaluate(Xs_te, Ys_te)
    print('===========part1_fully_connected_BN===============')
    print('test loss: ', test_loss)
    print('test accuracy:, ', test_acc)
    print('time: ', after_time)
    plot_accuracy(axis, 3, 0, history, test_acc, 'Batch Norm')
    plot_loss(axis, 3, 1, history, test_loss, 'Batch Norm')

    return history

def part2_1_step_size_grid_search(Xs_tr, Ys_tr, Xs_te, Ys_te):
    step_sizes = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001]
    validation_accuracy = {}
    validation_loss = {}
    for step_size in step_sizes:
        model, history = train_fully_connected_sgd(Xs_tr, Ys_tr, d1, d2, step_size, 0.9, batch_size, epochs)
        validation_accuracy[step_size] = history.history['val_accuracy'][-1]
        validation_loss[step_size] = history.history['val_loss'][-1]
    print('================part2_1_step_size_grid_search===============')
    print('validation accuracy: ')
    print(validation_accuracy)
    print('-------------------------------------------')
    print('validation loss: ')
    print(validation_loss)

def part2_2_grid_search(Xs_tr, Ys_tr, Xs_te, Ys_te):
    alphas = [0.1, 0.03, 0.01]
    betas = [0.99, 0.9, 0.7]
    batch_sizes = [128, 64, 32]
    validation_accuracy = {}
    validation_loss = {}
    for alpha1 in alphas:
        for beta1 in betas:
            for batch_size1 in batch_sizes:
                key_str = 'a: ' + str(alpha1) + ', beta: ' + str(beta1) + ', B: ' + str(batch_size1)
                model, history = train_fully_connected_sgd(Xs_tr, Ys_tr, d1, d2, alpha1, beta1, batch_size1, epochs)
                validation_loss[key_str] = history.history['val_loss'][-1]
                validation_accuracy[key_str] = history.history['val_accuracy'][-1]
    print('================part2_2_grid_search===============')
    print('validation accuracy: ')
    print(validation_accuracy)
    print('-------------------------------------------')
    print('validation loss: ')
    print(validation_loss)

def part2_3_random_search(Xs_tr, Ys_tr, Xs_te, Ys_te):
    alphas = [0.1, 0.03, 0.01]
    betas = [0.99, 0.9, 0.7]
    batch_sizes = [128, 64, 32]
    distribution = []
    for alpha1 in alphas:
        for beta1 in betas:
            for batch_size1 in batch_sizes:
                distribution.append((alpha1, beta1, batch_size1))

    random_sample = random2.sample(distribution, 10)
    validation_accuracy = {}
    validation_loss = {}
    for alpha2, beta2, batch_size2 in random_sample:
        key_str = 'a: ' + str(alpha2) + ', beta: ' + str(beta2) + ', B: ' + str(batch_size2)
        model, history = train_fully_connected_sgd(Xs_tr, Ys_tr, d1, d2, alpha2, beta2, batch_size2, epochs)
        validation_loss[key_str] = history.history['val_loss'][-1]
        validation_accuracy[key_str] = history.history['val_accuracy'][-1]
    print('================part2_3_random_search===============')
    print('validation accuracy: ')
    print(validation_accuracy)
    print('-------------------------------------------')
    print('validation loss: ')
    print(validation_loss)

def part3_CNN(Xs_tr, Ys_tr, Xs_te, Ys_te):
    before_time = time.time()
    model, history = train_CNN_sgd(Xs_tr, Ys_tr, .001, rho1, rho2, batch_size, epochs)
    after_time = time.time() - before_time
    test_loss, test_acc = model.evaluate(Xs_te, Ys_te)
    print('===========part3_CNN===============')
    print('test loss: ', test_loss)
    print('test accuracy:, ', test_acc)
    print('time: ', after_time)
    fig, (axis) = pyplot.subplots(1, 2, figsize=(10, 8))

    x_axis = numpy.arange(epochs) + 1
    axis[0].plot(x_axis, history.history['accuracy'], color='orange', label='accuracy')
    axis[0].plot(x_axis, history.history['val_accuracy'], color='green', label='validation accuracy')
    axis[0].plot(x_axis, [test_acc] * 10, color='blue', label='test accuracy')
    axis[0].set_title('CNN')
    axis[0].legend()

    axis[1].plot(x_axis, history.history['loss'], color='orange', label='loss')
    axis[1].plot(x_axis, history.history['val_loss'], color='green', label='validation loss')
    axis[1].plot(x_axis, [test_loss] * 10, color = 'blue', label='test loss')
    axis[1].set_title('CNN')
    axis[1].legend()

    pyplot.savefig("CNN.png")

    return history

if __name__ == "__main__":
    (Xs_tr, Ys_tr, Xs_te, Ys_te) = load_MNIST_dataset()
    fig, (axis) = pyplot.subplots(4, 2, figsize=(14, 18))

    sgd_history = part1_fully_connected_SGD(Xs_tr, Ys_tr, Xs_te, Ys_te, axis)
    momentum_history = part1_fully_connected_momentum(Xs_tr, Ys_tr, Xs_te, Ys_te, axis)
    adam_history = part1_fully_connected_adam(Xs_tr, Ys_tr, Xs_te, Ys_te, axis)
    bn_history = part1_fully_connected_BN(Xs_tr, Ys_tr, Xs_te, Ys_te, axis)
    pyplot.savefig("8Plots" + ".png")

    part2_1_step_size_grid_search(Xs_tr, Ys_tr, Xs_te, Ys_te)
    grid_time = time.time()
    part2_2_grid_search(Xs_tr, Ys_tr, Xs_te, Ys_te)
    print('###### Grid Search time: ', time.time()-grid_time)
    random_time = time.time()
    part2_3_random_search(Xs_tr, Ys_tr, Xs_te, Ys_te)
    print('###### Random Search time: ', time.time()-random_time)

    model, history = train_CNN_sgd(Xs_tr, Ys_tr, alpha, rho1, rho2, batch_size, epochs)
    histories = [(sgd_history, 'SGD')]
    plot_history(histories, 'loss')
    plot_history(histories, 'accuracy')
    plot_history(histories, 'val_loss')
    plot_history(histories, 'val_accuracy')
    part3_CNN(Xs_tr, Ys_tr, Xs_te, Ys_te)


