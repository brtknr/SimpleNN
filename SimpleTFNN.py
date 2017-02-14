# Implementation of a simple MLP network with one hidden layer. Tested on the iris data set.
# Requires: numpy, sklearn, tensorflow

# NOTE: In order to make the code simple, we rewrite x * W_1 + b_1 = x' * W_1'
# where x' = [x | 1] and W_1' is the matrix W_1 appended with a new row with elements b_1's.
# Similarly, for h * W_2 + b_2
import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)


def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights)

def forwardprop(X, w_1, w_2, w_3, w_4, keep_prob):
    """
    Forward-propagation.
    IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
    """
    h_1  = tf.nn.tanh(tf.matmul(X, w_1))  # The \sigma function
    h_1  = tf.nn.dropout(h_1,keep_prob)
    h_2  = tf.nn.tanh(tf.matmul(h_1, w_2))  # The \sigma function
    h_2  = tf.nn.dropout(h_2,keep_prob)
    h_3  = tf.nn.tanh(tf.matmul(h_2, w_3))  # The \sigma function
    h_3  = tf.nn.dropout(h_3,keep_prob)
    yhat = tf.matmul(h_3, w_4)  # The \varphi function
    return yhat

def get_iris_data():
    """ Read the iris data set and split them into training and test sets """
    iris   = datasets.load_iris()
    data   = iris["data"]
    target = iris["target"]

    # Prepend the column of 1s for bias
    # N, M  = data.shape
    # all_X = np.ones((N, M + 1))
    # all_X[:, 1:] = data

    all_X = data

    # Convert into one-hot vectors
    num_labels = len(np.unique(target))
    all_Y = np.eye(num_labels)[target]  # One liner trick!
    return train_test_split(all_X, all_Y, test_size=0.33, random_state=RANDOM_SEED)

def get_modulus_data():
    """ Generate a dataset where the output is the modulus of the sum of a random array """
    all_X = np.rint(np.random.random((750,3)))
    target = np.int_(all_X.sum(axis=1)%2)
    # target = np.int_(np.round(np.random.uniform(0,2,150)))

    # Convert into one-hot vectors
    num_labels = len(np.unique(target))
    all_Y = np.eye(num_labels)[target]  # One liner trick!

    return train_test_split(all_X, all_Y, test_size=0.33, random_state=RANDOM_SEED)

def main():
    # train_X, test_X, train_y, test_y = get_iris_data()
    train_X, test_X, train_y, test_y = get_modulus_data()

    # Layer's sizes
    x_size = train_X.shape[1]   # Number of input nodes: 4 features and 1 bias
    h_size = 10               # Number of hidden nodes
    y_size = train_y.shape[1]   # Number of outcomes (3 iris flowers)

    # Symbols
    X = tf.placeholder("float", shape=[None, x_size])
    y = tf.placeholder("float", shape=[None, y_size])
    keep_prob = tf.placeholder(tf.float32)

    # Weight initializations
    w_1 = init_weights((x_size, h_size))
    w_2 = init_weights((h_size, h_size))
    w_3 = init_weights((h_size, h_size))
    w_4 = init_weights((h_size, y_size))

    # Forward propagation
    yhat    = forwardprop(X, w_1, w_2, w_3, w_4, keep_prob)
    predict = tf.argmax(yhat, dimension=1)

    # Backward propagation
    loss    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(yhat, y))

    updates = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
    # updates = tf.train.AdamOptimizer.minimize(loss)

    # Run SGD
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    batch_size = 10

    CORRECT_COLOUR = 42
    INCORRECT_COLOUR = 41

    for epoch in range(1000):
        # Train with each example

        for i in range(len(train_X)//batch_size):
            slide = i*batch_size
            sess.run(updates, feed_dict={X: train_X[slide: slide + batch_size], y: train_y[slide: slide + batch_size], keep_prob: 1.0})

        train_accuracy = np.mean(np.argmax(train_y, axis=1) == sess.run(predict, feed_dict={X: train_X, y: train_y, keep_prob: 1.0}))
        test_accuracy  = np.mean(np.argmax(test_y, axis=1) == sess.run(predict, feed_dict={X: test_X, y: test_y, keep_prob: 1.0}))


        if epoch%10 == 0:
            print("Epoch = {}, train accuracy = {:0.2f}%, test accuracy = {:0.2f}%".format(epoch + 1, 100. * train_accuracy, 100. * test_accuracy))

            prediction = sess.run(predict, feed_dict={X: test_X, y: test_y, keep_prob: 1.0})
            success = np.argmax(test_y, axis=1) == prediction

            output = ''
            for p,s in zip(prediction,success):
                # Guide on how to colour text:
                # http://stackoverflow.com/questions/287871/print-in-terminal-with-colors-using-python
                output += '\x1b[5;37;{}m {} \x1b[0m'.format(CORRECT_COLOUR if s else INCORRECT_COLOUR, p)
            print (output)


    sess.close()

if __name__ == '__main__':
    main()