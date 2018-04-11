import numpy as np
import pandas as pd
import tensorflow as tf


path = "C:\\SYM\\Freelancing\\Haythm - Arabic HW Recognition\\"
XTr = pd.read_csv(path + "TrainImages.csv", header = None)
yTr = pd.read_csv(path + "TrainLabels.csv", header = None)
XTe = pd.read_csv(path + "TestImages.csv", header = None)
yTe = pd.read_csv(path + "TestLabels.csv", header = None)

NTr = XTr.shape[0]
NTe = XTe.shape[0]
D = XTr.shape[1]
d = int(np.sqrt(D))
K = len(np.unique(yTr))

XTr = XTr.values.astype('float32')
XTe = XTe.values.astype('float32')
XTr = XTr.reshape(NTr, d, d, 1)
XTe = XTe.reshape(NTe, d, d, 1)
YTr = pd.get_dummies(yTr[0])
YTe = pd.get_dummies(yTe[0])
YTr = YTr.values.astype('float32')
YTe = YTe.values.astype('float32')

featureMaps = [80, 64, 40]
kernelSizes = [3, 3, 3]
strides = [1, 1, 1]
pooling = [True, False, False]
neurons = [1024, 512, K]
convLayers = len(featureMaps)
fullLayers = len(neurons)

batchSize = 120


X = tf.placeholder('float', [None, d, d, 1])
Y = tf.placeholder('float', [None, K])
train = tf.placeholder('bool')
rate = tf.placeholder('float')

def convNet(Xdata, train, dropRate):
	network = Xdata
	for layer in range(convLayers):
		network = tf.layers.conv2d(network, featureMaps[layer], kernelSizes[layer],
			strides = strides[layer], data_format = "channels_last")
		print("C", layer+1, ": ", network)
		network = tf.nn.relu(network)
		if (pooling[layer]):
			network = tf.layers.max_pooling2d(network, 2, 2)
			print("P", layer+1, ": ", network)
	
	network = tf.contrib.layers.flatten(network)

	for layer in range(fullLayers):
		network = tf.contrib.layers.fully_connected(network, neurons[layer], activation_fn = None)
		if (layer < fullLayers-1):
			network = tf.nn.relu(network)
		network = tf.layers.dropout(network, rate = dropRate, training = train)

	return network

def trainConvNet(XTr, YTr, XTe, YTe, dropRate, showMess = True, epochs = 10):
	H = convNet(X, train, rate)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = H, labels = Y))
	optimizer = tf.train.AdamOptimizer().minimize(cost)
	init = tf.global_variables_initializer()

	h = tf.argmax(H, 1)
	correctPredictions = tf.equal(h, tf.argmax(Y, 1))
	accuracy = tf.reduce_mean(tf.cast(correctPredictions, 'float'))
	with tf.Session() as sess:
		sess.run(init)
		NTr = XTr.shape[0]
		NTe = XTe.shape[0]
		M = int(NTr/batchSize)
		for epoch in range(epochs):
			epochLoss = 0
			for i in range(M):
				ind = np.arange(0, batchSize) + i*batchSize
				XBatch = XTr[ind, :, :, :]
				YBatch = YTr[ind, :]
				_, loss = sess.run([optimizer, cost], 
					feed_dict = {X: XBatch, Y: YBatch, train: True, rate: dropRate})
				epochLoss += loss/float(M)
			accTr = 0.
			accTe = 0.
			hTe = np.zeros([0, 1])
			B = 840
			MTr = int(NTr/B)
			MTe = int(NTe/B)
			for i in range(MTr):
				ind = np.arange(0, B) + i*B
				XBatch = XTr[ind, :, :, :]
				YBatch = YTr[ind, :]
				accTr += accuracy.eval({X: XBatch, Y: YBatch, train: False, rate: dropRate})/MTr
			for i in range(MTe):
				ind = np.arange(0, B) + i*B
				XBatch = XTe[ind, :, :, :]
				YBatch = YTe[ind, :]
				accTe += accuracy.eval({X: XBatch, Y: YBatch, train: False, rate: dropRate})/MTe
				hBatch = h.eval({X: XBatch, Y: YBatch, train: False, rate: dropRate})
				hBatch = np.reshape(hBatch, [B, 1])
				hTe = np.concatenate((hTe, hBatch))
				
			if (showMess):
				print("Epoch ", epoch+1, " completed out of ", epochs, ", Loss: ", loss)
				print("Training Accuracy: ", accTr)
				print("Test Accuracy: ", accTe)
				print()
	
	return accTr, accTe, hTe

def validateConvNet(Xdata, Ydata, split):
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
		accTr[i], accCv[i], _ = trainConvNet(XTr, YTr, XCv, YCv, dropRates[i], False, 10)
		print("Drop Rate: ", dropRates[i])
		print("Training Accuracy: ", accTr[i])
		print("Validation Accuracy: ", accCv[i])
		print()
	i = np.argmax(accCv)

	dropRates = np.reshape(dropRates, [numRates, 1])
	accTr = np.reshape(accTr, [numRates, 1]) * 100
	accCv = np.reshape(accCv, [numRates, 1]) * 100
	arr = np.concatenate((dropRates, accTr, accCv), axis=1)
	np.savetxt(path + "CNN2.csv", arr, delimiter=",")

	return dropRates[i, 0]

# dropRate = validateConvNet(XTr, YTr, 0.3)
# print(dropRate)
_, _, hTe = trainConvNet(XTr, YTr, XTe, YTe, 0.2, epochs = 30)
# np.savetxt(path + "PredictionsCNN2.csv", hTe.astype(int), delimiter=",")
