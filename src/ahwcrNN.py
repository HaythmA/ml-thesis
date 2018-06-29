import numpy as np
import pandas as pd
import tensorflow as tf

# Load data from hard-disk (provide path in path variable).
# TrainImages.csv and TestImages.csv contain the input images.
# The input features are stored in matrix (DataFrame)
# of order number of examples by number of features.
# TrainLabels.csv and TestLabels.csv containt the corresponding labels.
# The output labels are stored in a vector (DataFrame)
# of order number of examples by 1.
path = ""
XTr = pd.read_csv(path + "TrainImages.csv", header=None)
yTr = pd.read_csv(path + "TrainLabels.csv", header=None)
XTe = pd.read_csv(path + "TestImages.csv", header=None)
yTe = pd.read_csv(path + "TestLabels.csv", header=None)

# Change the label vector to one-hot format.
# The labels are now stored as a matrix (DataFrame) of
# order number of features by number of classes.
# Change the data types of the training and test data
# from DataFrame to Numpy Array.
YTr = pd.get_dummies(yTr[0])
YTe = pd.get_dummies(yTe[0])
XTr = XTr.values.astype('float32')
XTe = XTe.values.astype('float32')
YTr = YTr.values.astype('float32')
YTe = YTe.values.astype('float32')

# Initialize some variables for example number of training
# and test examples pixels, classes.
# Initiallize parameters for the Convolution Neural Network.
# Number of layers and neurons in each layer and batch size.
NTr = XTr.shape[0]
NTe = XTe.shape[0]
D = XTr.shape[1]
K = len(np.unique(yTr))
nodes = [D, 500, 500, 500, 500, K]
L = len(nodes) - 1
batchSize = 120

# Define necessary placeholders for running tensorflow 
# sessions. X and Y are for training on training set and
# evaluation on training and test data. train and rate are
# used for dropout which is done only during training.
X = tf.placeholder('float', [None, D])
Y = tf.placeholder('float', [None, K])
train = tf.placeholder('bool')
rate = tf.placeholder('float')


def neuralNet(Xdata, train, dropRate):
    """
     Define neural network model.
     This method peforms the forward propagation of the neural network
     with parameters defined above.

    :param Xdata: Input Features.
    :param train: the boolean indication of training or evaluation.
    :param dropRate: dropout rate.
    :return: the predicted output of the network.
    """

    network = Xdata
    for i in range(L):
        network = tf.contrib.layers.fully_connected(network, nodes[i + 1],
                                                    activation_fn=None)
        if (i < L - 1):
            network = tf.nn.relu(network)
        if (i == 2 or i == 3):
            network = tf.layers.dropout(network, rate=dropRate, training=train)
    return network


def trainNeuralNet(XTr, YTr, XTe, YTe, dropRate, showMess=True, epochs=10):
    """
    This method trains the neural network defined above.
    It uses the neuralNet method to predict the output of
    given input image for training and evaluation.

    :param XTr: Images in the training set.
    :param YTr: Lables in training set (one-hot).
    :param XTe: Images in the test set.
    :param YTe: Lables in test set (one-hot).
    :param dropRate: dropout rate.
    :param showMess: show the accuracies and loss for every epoch during training.
    :param epochs: total number of epochs.
    :return: The method returns the training and test accuracy and evaluation of the
trained network on test data.
    """
    H = neuralNet(X, train, rate)
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=H, labels=Y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    init = tf.global_variables_initializer()

    h = tf.argmax(H, 1)
    correctPredictions = tf.equal(h, tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correctPredictions, 'float'))
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(epochs):
            epochLoss = 0
            NTr = XTr.shape[0]
            M = int(NTr / batchSize)
            for i in range(M):
                ind = np.arange(0, batchSize) + i * batchSize
                XBatch = XTr[ind, :]
                YBatch = YTr[ind, :]
                _, loss = sess.run([optimizer, cost], feed_dict={
                    X: XBatch, Y: YBatch, train: True, rate: dropRate
                })
                epochLoss += loss / float(M)
            accTr = accuracy.eval(
                {X: XTr, Y: YTr, train: False, rate: dropRate})
            accTe = accuracy.eval(
                {X: XTe, Y: YTe, train: False, rate: dropRate})
            hTe = h.eval({X: XTe, Y: YTe, train: False, rate: dropRate})
            if (showMess):
                print(
                    "Epoch ", epoch + 1, " completed out of ", epochs,
                    ", Loss: ",
                    loss)
                print("Training Accuracy: ", accTr)
                print("Test Accuracy: ", accTe)
                print()

    return accTr, accTe, hTe


def validateNeuralNet(Xdata, Ydata, split):
    """
    Find optimal dropout rate through cross-validation.
    This method calls The trainNeuralNet method.
    :param Xdata: Input Features of given set.
    :param Ydata: Labels of given set (one-hot).
    :param split: Fraction of the given set to be used for validation.
    :return: optimal value of dropout rate.
    """

    numRates = 12
    start = 0.1
    step = 0.05
    dropRates = step * np.arange(numRates) + start
    accTr = np.zeros(numRates)
    accCv = np.zeros(numRates)
    N = Xdata.shape[0]
    ind = np.random.rand(N) <= split
    XTr = Xdata[ind, :]
    XCv = Xdata[~ind, :]
    YTr = Ydata[ind, :]
    YCv = Ydata[~ind, :]
    for i in range(0, numRates):
        accTr[i], accCv[i], _ = trainNeuralNet(XTr, YTr, XCv, YCv, dropRates[i],
                                               False, 50)
        print("Drop Rate: ", dropRates[i])
        print("Training Accuracy: ", accTr[i])
        print("Validation Accuracy: ", accCv[i])
        print()
    i = np.argmax(accCv)

    dropRates = np.reshape(dropRates, [numRates, 1])
    accTr = np.reshape(accTr, [numRates, 1]) * 100
    accCv = np.reshape(accCv, [numRates, 1]) * 100
    arr = np.concatenate((dropRates, accTr, accCv), axis=1)
    np.savetxt(path + "HogNN.csv", arr, delimiter=",")
    return dropRates[i, 0]


# Call the method validateNeuralNet to perform
# dropRate = validateNeuralNet(XTr, YTr, 0.3)
# print(dropRate)
_, _, hTe = trainNeuralNet(XTr, YTr, XTe, YTe, 0.45, epochs=1)
# np.savetxt(path + "PredictionsHogNN.csv", hTe.astype(int), delimiter=",")
