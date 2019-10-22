#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Created on Tue Oct 22 22:04:41 2019

@author: Filip
"""

import numpy as np
import tensorflow as tf
from math import ceil
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
import pandas as pd


def load_dataset():
    train_set_x_orig = np.load('train.pickle',
                               allow_pickle=True)['features']
    train_set_y_orig = np.load('train.pickle',
                               allow_pickle=True)['labels']

    valid_set_x_orig = np.load('valid.pickle',
                               allow_pickle=True)['features']
    valid_set_y_orig = np.load('valid.pickle',
                               allow_pickle=True)['labels']

    classes = valid_set_y_orig.max() + 1  # No. of classes

    # correcting dimensions

    train_set_y_orig = train_set_y_orig.reshape((1,
            train_set_y_orig.shape[0]))
    valid_set_y_orig = valid_set_y_orig.reshape((1,
            valid_set_y_orig.shape[0]))

    # Flattening the training and test images

    X_train_flatten = \
        train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    X_valid_flatten = \
        valid_set_x_orig.reshape(valid_set_x_orig.shape[0], -1).T

    # Normalizing images' vectors

    X_train = X_train_flatten / 255.
    X_valid = X_valid_flatten / 255.

    # Convert training and test labels to one hot matrices

    y_train = np.eye(classes)[train_set_y_orig.reshape(-1)].T
    y_valid = np.eye(classes)[valid_set_y_orig.reshape(-1)].T

    return (
        X_train,
        y_train,
        X_valid,
        y_valid,
        valid_set_x_orig,
        X_valid_flatten,
        classes,
        )


def random_mini_batches(X, Y, minibatch_size=64):
    m = X.shape[1]
    X_minibatches = []
    Y_minibatches = []

    # Shuffling X and Y

    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0], m))

    # Dividing into batches

    minibatches_no = ceil(m / minibatch_size)
    for k in range(0, minibatches_no):
        mini_batch_X = shuffled_X[:, k * minibatch_size:k
                                  * minibatch_size + minibatch_size]
        mini_batch_Y = shuffled_Y[:, k * minibatch_size:k
                                  * minibatch_size + minibatch_size]
        X_minibatches.append(mini_batch_X)
        Y_minibatches.append(mini_batch_Y)

    return (X_minibatches, Y_minibatches)


def initialize_parameters(hidden_layers, inputs_no, outputs_no):

    weights = []
    biases = []

    if len(hidden_layers):
        weights.append(tf.get_variable(name='W1',
                       shape=[hidden_layers[0], inputs_no],
                       initializer=tf.contrib.layers.xavier_initializer()))
        biases.append(tf.get_variable(name='b1',
                      shape=[hidden_layers[0], 1],
                      initializer=tf.zeros_initializer()))

    for i in range(len(hidden_layers) - 1):
        weights.append(tf.get_variable(name='W' + str(i + 2),
                       shape=[hidden_layers[i + 1], hidden_layers[i]],
                       initializer=tf.contrib.layers.xavier_initializer()))
        biases.append(tf.get_variable(name='b' + str(i + 2),
                      shape=[hidden_layers[i + 1], 1],
                      initializer=tf.zeros_initializer()))

    weights.append(tf.get_variable(name='W' + str(len(hidden_layers)
                   + 1), shape=[outputs_no, hidden_layers[-1]],
                   initializer=tf.contrib.layers.xavier_initializer()))
    biases.append(tf.get_variable(name='b' + str(len(hidden_layers)
                  + 1), shape=[outputs_no, 1],
                  initializer=tf.zeros_initializer()))

    return (weights, biases)


def forward_propagation(
    X,
    weights,
    biases,
    k_prob,
    is_train,
    Z_shapes,
    ):

    activations = []
    nn_depth = len(weights)  # Number of layers, excluding input
    A = X  # X is actvation of input layer
    for i in range(nn_depth):
        Z = tf.matmul(weights[i], A) + biases[i]
        Z = tf.reshape(Z, [Z_shapes[i], -1])
        Z_norm = tf.layers.batch_normalization(Z, training=is_train,
                axis=0)
        A = tf.nn.relu(Z_norm)
        A = tf.nn.dropout(A, keep_prob=k_prob)
        activations.append(A)  # dropout implementation
    return Z_norm


def compute_cost(
    Z3,
    Y,
    weights,
    biases,
    lambda_reg,
    ):

    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    cost = \
        tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                       labels=labels))
    reg = 0
    for W in weights:
        reg += tf.nn.l2_loss(W)
    cost = tf.reduce_mean(cost + lambda_reg * reg)

    return cost


def model(
    X_train,
    Y_train,
    X_valid,
    X_valid_orig,
    Y_valid,
    im_to_val,
    hidden_layers=[100, 50],
    learning_rate_init=0.0001,
    decay_rate=0.1,
    num_epochs=6,
    minibatch_size=32,
    print_cost=True,
    keep_prob_val=0.7,
    lambda_reg=0.01,
    ):

    # to be able to rerun the model without overwriting tf variables

    ops.reset_default_graph()

    # (n_x: input size, m : number of examples in the train set)

    (n_x, m) = X_train.shape

    # output size

    n_y = Y_train.shape[0]

    # buffers for evaluation

    cost_train_axis = []
    cost_valid_axis = []
    iter_axis = []
    learning_rate_axis = []

    # creating placeholders for inputs and labels

    X = tf.placeholder(dtype=tf.float32, shape=(n_x, None))
    Y = tf.placeholder(dtype=tf.float32, shape=(n_y, None))

    # training flag for batch normalization

    is_train = tf.placeholder(tf.bool, name='is_train')

    # dropout parameter

    keep_prob = tf.placeholder(dtype=tf.float32)

    # a placeholder for epoch no.

    epoch_no = tf.placeholder(tf.float32, shape=())

    # learning rate decay

    learning_rate = learning_rate_init / (1 + decay_rate * epoch_no)

    # parameters = initialize_parameters()

    (weights, biases) = initialize_parameters(hidden_layers, n_x, n_y)

    # last layer output calculation

    Z = forward_propagation(
        X,
        weights,
        biases,
        keep_prob,
        is_train,
        hidden_layers + [n_y],
        )

    # cost calculation with L2 regularizatoin

    cost = compute_cost(Z, Y, weights, biases, lambda_reg=lambda_reg)

    # explicit control of update of normalization parameters' moving average in batch normalization

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = \
            tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    predictions = tf.argmax(Z)
    correct_prediction = tf.equal(tf.argmax(Z), tf.argmax(Y))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):

            # initializing a cost related to an epoch

            epoch_cost = 0.

            # random division into minibatches

            num_minibatches = int(m / minibatch_size)
            (minibatches_X, minibatches_Y) = \
                random_mini_batches(X_train, Y_train, minibatch_size)
            for i in range(len(minibatches_X)):

                # selecting the minibatch for training

                minibatch_X = minibatches_X[i]
                minibatch_Y = minibatches_Y[i]

                # training

                (_, minibatch_cost) = sess.run([optimizer, cost],
                        feed_dict={
                    X: minibatch_X,
                    Y: minibatch_Y,
                    keep_prob: keep_prob_val,
                    epoch_no: epoch,
                    is_train: True,
                    })
                epoch_cost += minibatch_cost / num_minibatches

            # printing the cost for every 5th epoch

            if print_cost == True and (epoch + 1) % 5 == 0:
                learning_rate_axis.append(learning_rate.eval({epoch_no: epoch
                        + 1}))

                # Evaluation on cross validation set performed without dropout

                cost_train = cost.eval({
                    X: X_train,
                    Y: Y_train,
                    keep_prob: 1,
                    is_train: False,
                    })
                cost_valid = cost.eval({
                    X: X_valid,
                    Y: Y_valid,
                    keep_prob: 1,
                    is_train: False,
                    })

                print (f"Results after epoch {epoch+1}:")
                print(f"Train cost: {cost_train:.2f}\tValidation cost: {cost_valid:.2f}")
                print (f"Train accuracy: {100*accuracy.eval({X: X_train, Y: Y_train, keep_prob : 1, is_train : False}):.2f}%\tTest accuracy: {100*accuracy.eval({X: X_valid, Y: Y_valid, keep_prob : 1, is_train : False}):.2f}%")

                cost_train_axis.append(cost_train)
                cost_valid_axis.append(cost_valid)
                iter_axis.append(epoch + 1)

                # plotting the cost

                plt.plot(iter_axis, cost_train_axis)
                plt.plot(iter_axis, cost_valid_axis)
                plt.ylabel('cost')
                plt.xlabel('Epochs')
                plt.legend(['Training set', 'Cross validation set'])
                plt.show()

            # plotting the learning rate decay

            if print_cost == True and epoch + 1 % 25 == 0:
                plt.plot(iter_axis, learning_rate_axis)
                plt.show()

        labels = pd.read_csv('Sign_labels.txt', delimiter=',')
        predictions = sess.run(predictions, feed_dict={
            X: X_valid,
            Y: Y_valid,
            keep_prob: 1,
            is_train: False,
            })
        rand_idx = np.random.randint(1, 1000, im_to_val)
        for no in range(im_to_val):
            number = rand_idx[no]
            my_image = X_valid_orig[number]
            my_image_prediction = predictions[number]
            plt.imshow(my_image)
            plt.show()
            print ('Algorithm predicts: y = ' \
                + str(np.squeeze(my_image_prediction)))
            print ('Algorithm predicts: y = ' + labels[labels['ClassId']
                    == int(my_image_prediction)]['SignName'].values)

        # saving learnt parameters

        (weights, biases) = sess.run([weights, biases])
        print ('Parameters have been trained!')

        return (weights, biases)


def main(validation_images_no):

    # Loading the dataset

    (
        X_train,
        y_train,
        X_valid,
        y_valid,
        X_valid_orig,
        X_valid_flatten,
        classes,
        ) = load_dataset()

    # learning the model and saving the learnt parameters

    (weights, biases) = model(
        X_train,
        y_train,
        X_valid,
        X_valid_orig,
        y_valid,
        im_to_val=validation_images_no,
        hidden_layers=[100, 50],
        learning_rate_init=0.0001,
        decay_rate=0.1,
        num_epochs=100,
        minibatch_size=32,
        print_cost=True,
        keep_prob_val=0.7,
        lambda_reg=0.01,
        )

    return (weights, biases)


if __name__ == '__main__':
    main(5)
