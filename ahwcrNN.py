import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn


# load data
path = "C:\\SYM\\Freelancing\\Haythm - Arabic HW Recognition\\"
# path = "C:\\Users\\Sheheryar Mehmood\\Desktop\\MNIST\\"
XTr = pd.read_csv(path + "TrainImages.csv", header = None)
yTr = pd.read_csv(path + "TrainLabels.csv", header = None)
XTe = pd.read_csv(path + "TestImages.csv", header = None)
yTe = pd.read_csv(path + "TestLabels.csv", header = None)

YTr = pd.get_dummies(yTr[0])
YTe = pd.get_dummies(yTe[0])
XTr = XTr.values.astype('float32')
XTe = XTe.values.astype('float32')
YTr = YTr.values.astype('float32')
YTe = YTe.values.astype('float32')


NTr = XTr.shape[0]
NTe = XTe.shape[0]
D = XTr.shape[1]
K = len(np.unique(yTr))
nodes = [D, 500, 500, 500, 500, K]
L = len(nodes) - 1

batchSize = 120

X = tf.placeholder('float', [None, D])
Y = tf.placeholder('float', [None, K])
train = tf.placeholder('bool')
rate = tf.placeholder('float')


def neuralNet(Xdata, train, dropRate):
	
	network = Xdata
	for i in range(L):
		network = tf.contrib.layers.fully_connected(network, nodes[i+1], activation_fn = None)
		if (i<L-1):
			network = tf.nn.relu(network)
		if (i==2 or i==3):
			network = tf.layers.dropout(network, rate = dropRate, training = train)
	return network

def trainNeuralNet(XTr, YTr, XTe, YTe, dropRate, showMess = True, epochs = 10):
	H = neuralNet(X, train, rate)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = H, labels = Y))
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
			M = int(NTr/batchSize)
			for i in range(M):
				ind = np.arange(0, batchSize) + i*batchSize
				XBatch = XTr[ind, :]
				YBatch = YTr[ind, :]
				_, loss = sess.run([optimizer, cost], feed_dict = {
					X: XBatch, Y: YBatch, train: True, rate: dropRate
					})
				epochLoss += loss/float(M)
			accTr = accuracy.eval({X: XTr, Y: YTr, train: False, rate: dropRate})
			accTe = accuracy.eval({X: XTe, Y: YTe, train: False, rate: dropRate})
			hTe = h.eval({X: XTe, Y: YTe, train: False, rate: dropRate})
			if (showMess):
				print("Epoch ", epoch+1, " completed out of ", epochs, ", Loss: ", loss)
				print("Training Accuracy: ", accTr)
				print("Test Accuracy: ", accTe)
				print()

	return accTr, accTe, hTe

def validateNeuralNet(Xdata, Ydata, split):
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
		accTr[i], accCv[i], _ = trainNeuralNet(XTr, YTr, XCv, YCv, dropRates[i], False, 50)
		print("Drop Rate: ", dropRates[i])
		print("Training Accuracy: ", accTr[i])
		print("Validation Accuracy: ", accCv[i])
		print()
	i = np.argmax(accCv)

	dropRates = np.reshape(dropRates, [numRates, 1])
	accTr = np.reshape(accTr, [numRates, 1]) * 100
	accCv = np.reshape(accCv, [numRates, 1]) * 100
	arr = np.concatenate((dropRates, accTr, accCv), axis = 1)
	np.savetxt(path + "HogNN.csv", arr, delimiter = ",")

	return dropRates[i, 0]



'''dropRate = validateNeuralNet(XTr, YTr, 0.3)
print(dropRate)'''
_, _, hTe = trainNeuralNet(XTr, YTr, XTe, YTe, 0.45, epochs = 1)
hTe = np.reshape(hTe, [NTe, 1])
# np.savetxt(path + "PredictionsHogNN.csv", hTe.astype(int), delimiter=",")