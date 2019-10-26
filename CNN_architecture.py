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

  # Normalizing images' vectors

    X_train = train_set_x_orig / 255.
    X_valid = valid_set_x_orig / 255.

  # Convert training and test labels to one hot matrices

    y_train = np.eye(classes)[train_set_y_orig.reshape(-1)]
    y_valid = np.eye(classes)[valid_set_y_orig.reshape(-1)]

    return (X_train, y_train, X_valid, y_valid, classes)


def random_mini_batches(X, Y, minibatch_size=64):
    m = X.shape[0]
    X_minibatches = []
    Y_minibatches = []

  # Shuffling X and Y

    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :, :, :]
    shuffled_Y = Y[permutation, :]

  # Dividing into batches

    minibatches_no = ceil(m / minibatch_size)
    for k in range(0, minibatches_no):
        mini_batch_X = shuffled_X[k * minibatch_size:k
                * minibatch_size + minibatch_size, :, :, :]
        mini_batch_Y = shuffled_Y[k * minibatch_size:k
                * minibatch_size + minibatch_size, :]
        X_minibatches.append(mini_batch_X)
        Y_minibatches.append(mini_batch_Y)

    return (X_minibatches, Y_minibatches)


def initialize_parameters(channels_input, Layers):

    weights = []

  # Weights for a first hidden layer

    if Layers[0].layer_type == 'Convolutional':
        weights.append(tf.get_variable('W1',
                       [Layers[0].filter_size,
                       Layers[0].filter_size, channels_input,
                       Layers[0].filters_no],
                       initializer=tf.contrib.layers.xavier_initializer()))

    for i in range(len(Layers) - 1):
        if Layers[i + 1].layer_type == 'Max Pooling':
            Layers[i + 1].filters_no = Layers[i].filters_no
            weights.append([])
        if Layers[i + 1].layer_type == 'Convolutional':
            weights.append(tf.get_variable('W' + str(i + 2),
                           [Layers[i + 1].filter_size, Layers[i
                           + 1].filter_size, Layers[i].filters_no,
                           Layers[i + 1].filters_no],
                           initializer=tf.contrib.layers.xavier_initializer()))

    return weights


def forward_propagation(
    X,
    weights,
    keep_prob,
    is_train,
    Layers,
    ):

    nn_depth = len(Layers)  # Number of layers, excluding input
    A = X  # X is actvation of input layer
    for i in range(nn_depth - 1):

        if Layers[i].layer_type == 'Convolutional':
            Z = tf.nn.conv2d(A, weights[i], strides=[1,
                             Layers[i].stride, Layers[i].stride,
                             1], padding=Layers[i].padding)
            A = tf.nn.relu(Z)
            A = tf.nn.dropout(A, keep_prob=keep_prob)
        if Layers[i].layer_type == 'Max Pooling':
            A = tf.nn.max_pool(A, ksize=[1, Layers[i].filter_size,
                               Layers[i].filter_size, 1],
                               strides=[1, Layers[i].stride,
                               Layers[i].stride, 1],
                               padding=Layers[i].padding)
            A = tf.nn.dropout(A, keep_prob=keep_prob)
        if Layers[i].layer_type == 'Fully Connected':
            F = tf.contrib.layers.flatten(A)
            A = tf.contrib.layers.fully_connected(F,
                    Layers[i].output_nodes,
                    activation_fn=Layers[i].activation_fn)
    F = tf.contrib.layers.flatten(A)
    A = tf.contrib.layers.fully_connected(F, Layers[nn_depth
            - 1].output_nodes, activation_fn=Layers[nn_depth
            - 1].activation_fn)
    return A


def compute_cost(Z, Y, weights):

    cost = \
        tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z,
                       labels=Y))
    cost = tf.reduce_mean(cost)

    return cost


class Layer:

    def __init__(
        self,
        layer_type,
        filter_size=1,
        stride=1,
        filters_no=1,
        padding='SAME',
        activation_fn=tf.nn.relu,
        output_nodes=0,
        ):
        self.layer_type = layer_type
        self.weights = []
        if self.layer_type == 'Max Pooling':
            self.padding = 'VALID'
            self.filter_size = filter_size
            self.stride = stride
        if self.layer_type == 'Convolutional':
            self.padding = 'SAME'
            self.filter_size = filter_size
            self.stride = stride
            self.filters_no = filters_no
        if self.layer_type == 'Fully Connected':
            self.output_nodes = output_nodes
            self.activation_fn = activation_fn
            if activation_fn == None:
                self.activation_fn = None


def model(
    X_train,
    Y_train,
    X_valid,
    Y_valid,
    im_to_val,
    learning_rate_init=0.0001,
    decay_rate=0.1,
    num_epochs=6,
    minibatch_size=32,
    keep_prob_val=0.7,
    ):

  # to be able to rerun the model without overwriting tf variables

    ops.reset_default_graph()

    (m, height_input, width_input, channels_input) = X_train.shape

  # output size

    labels_no = Y_train.shape[1]

  # buffers for evaluation

    cost_train_axis = []
    cost_valid_axis = []
    iter_axis = []
    learning_rate_axis = []

  # creating placeholders for inputs and labels

    X = tf.placeholder(dtype=tf.float32, shape=(None, height_input,
                       width_input, channels_input))
    Y = tf.placeholder(dtype=tf.float32, shape=(None, labels_no))

  # training flag for batch normalization

    is_train = tf.placeholder(tf.bool, name='is_train')

  # dropout parameter

    keep_prob = tf.placeholder(dtype=tf.float32)

  # a placeholder for epoch no.

    epoch_no = tf.placeholder(tf.float32, shape=())

  # learning rate decay

    learning_rate = learning_rate_init / (1 + decay_rate * epoch_no)

  # Defining layers in the model. This model is based on VGG-16 architecture

    Layers = []
    Layers.append(Layer(layer_type='Convolutional', filter_size=3,
                  stride=1, filters_no=64))
    Layers.append(Layer(layer_type='Convolutional', filter_size=3,
                  stride=1, filters_no=64))
    Layers.append(Layer(layer_type='Max Pooling', filter_size=2,
                  stride=2))
    Layers.append(Layer(layer_type='Convolutional', filter_size=3,
                  stride=1, filters_no=128))
    Layers.append(Layer(layer_type='Convolutional', filter_size=3,
                  stride=1, filters_no=128))
    Layers.append(Layer(layer_type='Max Pooling', filter_size=2,
                  stride=2))
    Layers.append(Layer(layer_type='Fully Connected',
                  output_nodes=1024))
    Layers.append(Layer(layer_type='Fully Connected',
                  output_nodes=128))
    Layers.append(Layer(layer_type='Fully Connected',
                  output_nodes=43, activation_fn=None))

  # Initialization of parameters

    weights = initialize_parameters(channels_input, Layers)

  # last layer output calculation

    Z = forward_propagation(X, weights, keep_prob, is_train, Layers)

  # cost calculation with L2 regularizatoin

    cost = compute_cost(Z, Y, weights)

  # explicit control of update of normalization parameters' moving average in batch normalization

    optimizer = \
        tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

  # for saving model's parameters

    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    init = tf.global_variables_initializer()

    predictions = tf.argmax(Z, 1)
    correct_prediction = tf.equal(tf.argmax(Z, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):

          # initializing a cost related to an epoch

            epoch_cost = 0.
            epoch_acc = 0

          # random division into minibatches

            (minibatches_X, minibatches_Y) = \
                random_mini_batches(X_train, Y_train,
                    minibatch_size)
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
                epoch_cost += minibatch_cost / len(minibatches_X)
                epoch_acc += 100 * accuracy.eval({
                    X: minibatch_X,
                    Y: minibatch_Y,
                    keep_prob: 1,
                    is_train: False,
                    }) / len(minibatches_X)

          # printing the cost for every 5th epoch

            if (epoch + 1) % 1 == 0:
                learning_rate_axis.append(learning_rate.eval({epoch_no: epoch
                        + 1}))

              # Evaluation on cross validation set performed without dropout

                cost_valid = cost.eval({
                    X: X_valid,
                    Y: Y_valid,
                    keep_prob: 1,
                    is_train: False,
                    })

                print (f"Results after epoch {epoch+1}:")
                print(f"Train cost: {epoch_cost:.2f}\tValidation cost: {cost_valid:.2f}")
                print (f"Train accuracy: {epoch_acc:.2f}%\tTest accuracy: {100*accuracy.eval({X: X_valid, Y: Y_valid, keep_prob : 1, is_train : False}):.2f}%")

                cost_train_axis.append(epoch_cost)
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

            if (epoch + 1) == num_epochs:
                plt.plot(iter_axis, learning_rate_axis)
                plt.title("Learning rate")
                plt.show()

        labels = pd.read_csv('label_names.csv', delimiter=',')
        predictions = sess.run(predictions, feed_dict={
            X: X_valid,
            Y: Y_valid,
            keep_prob: 1,
            is_train: False,
            })
        rand_idx = np.random.randint(1, 1000, im_to_val)
        for no in range(im_to_val):
            number = rand_idx[no]
            my_image = X_valid[number]
            my_image_prediction = predictions[number]
            plt.imshow(my_image)
            plt.show()
            print ('Algorithm predicts: y = ' \
                + str(np.squeeze(my_image_prediction)))
            print ('Algorithm predicts: y = ' \
                + labels[labels['ClassId']
                         == int(my_image_prediction)]['SignName'].values)

      # saving learnt parameters

        weights = sess.run(weights)
        print ('Parameters have been trained!')
        variables = sess.run(variables)
        return (weights, Layers, variables)


def forward_propagation_forpredict(
    X,
    weights_conv,
    param_fc,
    keep_prob,
    is_train,
    Layers,
    ):

    nn_depth = len(Layers)  # Number of layers, excluding input
    A = X  # X is actvation of input layer
    param_counter = 0  # used to extract fully connected layers' weights and biases
    for i in range(nn_depth - 1):

        if Layers[i].layer_type == 'Convolutional':
            Z = tf.nn.conv2d(A, weights_conv[i], strides=[1,
                             Layers[i].stride, Layers[i].stride,
                             1], padding=Layers[i].padding)
            A = tf.nn.relu(Z)
            A = tf.nn.dropout(A, keep_prob=keep_prob)
            param_counter += 1
        if Layers[i].layer_type == 'Max Pooling':
            A = tf.nn.max_pool(A, ksize=[1, Layers[i].filter_size,
                               Layers[i].filter_size, 1],
                               strides=[1, Layers[i].stride,
                               Layers[i].stride, 1],
                               padding=Layers[i].padding)
            A = tf.nn.dropout(A, keep_prob=keep_prob)
        if Layers[i].layer_type == 'Fully Connected':
            F = tf.contrib.layers.flatten(A)
            A = tf.contrib.layers.fully_connected(F,
                    Layers[i].output_nodes,
                    activation_fn=Layers[i].activation_fn,
                    weights_initializer=tf.constant_initializer(param_fc[param_counter],
                    dtype=tf.float32),
                    biases_initializer=tf.constant_initializer(param_fc[param_counter
                    + 1], dtype=tf.float32))
            param_counter += 2
    F = tf.contrib.layers.flatten(A)
    A = tf.contrib.layers.fully_connected(F, Layers[nn_depth
            - 1].output_nodes, activation_fn=Layers[nn_depth
            - 1].activation_fn,
            weights_initializer=tf.constant_initializer(param_fc[param_counter],
            dtype=tf.float32),
            biases_initializer=tf.constant_initializer(param_fc[param_counter
            + 1], dtype=tf.float32))
    return A


def predict(
    X_eval,
    Y_eval,
    weights,
    param_fc,
    Layers,
    im_to_val,
    ):

    labels = pd.read_csv('label_names.csv', delimiter=',')
    X = tf.placeholder(dtype=tf.float32, shape=(None,
                       X_eval.shape[1], X_eval.shape[2],
                       X_eval.shape[3]))

  # Calculation of last layer's output. Because it is testing phase, the probability of keeping a node in dropout layer is set to 100%.

    Z = forward_propagation_forpredict(
        X,
        weights,
        param_fc,
        1,
        False,
        Layers,
        )
    with tf.Session() as predict:
        predict.run(tf.global_variables_initializer())
        predict.run(tf.local_variables_initializer())
        predictions = predict.run(tf.argmax(Z, 1),
                feed_dict={X: X_eval})

  # Visualizing the predictions

    rand_idx = np.random.randint(1, 1000, im_to_val)
    for no in range(im_to_val):
        number = rand_idx[no]
        my_image = X_valid[number]
        my_image_prediction = predictions[number]
        plt.imshow(my_image)
        plt.show()
        print ('Algorithm predicts: y = ' \
            + str(np.squeeze(my_image_prediction)))
        print ('Algorithm predicts: y = ' + labels[labels['ClassId']
                == int(my_image_prediction)]['SignName'].values)


def main(validation_images_no):

  # import os

  # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  # Loading the dataset

    (X_train, y_train, X_valid, y_valid, classes) = load_dataset()

  # learning the model and saving the learnt parameters

    (weights, Layers, variables) = model(
        X_train,
        y_train,
        X_valid,
        y_valid,
        im_to_val=validation_images_no,
        learning_rate_init=0.0001,
        decay_rate=0.02,
        num_epochs=50,
        minibatch_size=32,
        keep_prob_val=0.5,
        )

    return (weights, Layers, variables)


if __name__ == '__main__':

    (X_train, y_train, X_valid, y_valid, classes) = load_dataset()
    (weights, Layers, variables) = main(5)
    predict(
        X_valid,
        y_valid,
        weights,
        variables,
        Layers,
        5,
        )
