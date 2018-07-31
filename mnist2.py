import numpy as np
import matplotlib.pyplot as plt
import csv
import time

# Load and prepare the data

def load_data(filename):
	labels = []
	images = []
	with open(filename, 'r') as csv_file:
		for data in csv.reader(csv_file):
			# The first column is the label
			labels.append(data[0])

			# The rest of columns are pixels
			pixels = data[1:]
			pixels = np.array(pixels, dtype='float')
			images.append(pixels)
	
	X = np.array(images) / 255

	Y = []
	for label in labels:
		y = [0]*10
		y[int(label)] = 1
		Y.append(y)

	return X, np.array(Y)

train_X, train_Y = load_data('mnist_train.csv')
test_X, test_Y = load_data('mnist_test.csv')
print("Loaded MNIST data")

# Math

def sigmoid(x, deriv=False):
	if deriv:
		return sigmoid(x) * (1 - sigmoid(x))
	else:
		return 1 / (1 + np.exp(-x))

def mse(x, y):
	return np.dot(x - y, x - y) / len(x)

def cross_entropy(x, y):
	return -np.mean(y * np.log(x) + (1 - y) * np.log(1 - x))

# Fully-connected feed-forward neural network

class Network:

	def __init__(self, layer_sizes):
		self.layer_sizes = layer_sizes
		self.num_layers = len(layer_sizes)

		sigma = 0.1 # TODO: do some smart weight initialization
		self.weights = [sigma * np.random.randn(layer_sizes[i + 1], layer_sizes[i]) for i in range(len(layer_sizes) - 1)]
		self.biases = [np.zeros(layer_sizes[i]) for i in range(1, len(layer_sizes))]

	def feed_forward(self, x):
		for W, b in zip(self.weights, self.biases):
			x = sigmoid(np.dot(W, x) + b)
		return x

	def backprop(self, x, y):
		# Forward pass
		Z = []
		A = [x]
		for W, b in zip(self.weights, self.biases):
			Z.append(np.dot(W, A[-1]))
			A.append(sigmoid(Z[-1]))

		# Backward pass
		weight_grads = [0] * (self.num_layers - 1)
		bias_grads = [0] * (self.num_layers - 1)

		delta = (A[-1] - y) * sigmoid(Z[-1], deriv=True)
		weight_grads[-1] = np.outer(delta, A[-2].transpose())
		bias_grads[-1] = delta

		for i in range(2, self.num_layers - 1):
			delta = np.dot(self.weights[-i+1].transpose(), delta) * sigmoid(Z[-i], deriv=True)
			weight_grads[-i] = np.outer(delta, A[-i-1].transpose())
			bias_grads[-i] = delta

		loss = cross_entropy(A[-1], y)

		return loss, weight_grads, bias_grads

	def update_from_batch(self, X, Y, learning_rate, momentum):
		errors = []
		weight_updates = [0] * self.num_layers
		bias_updates = [0] * self.num_layers
		for x, y in zip(X, Y):
			loss, weight_grads, bias_grads = self.backprop(x, y)
			errors.append(loss)

			for j in range(self.num_layers - 1):
				weight_updates[j] = learning_rate * weight_grads[j] + momentum * weight_updates[j]
				bias_updates[j] = learning_rate * bias_grads[j]  + momentum * bias_updates[j]

		for j in range(self.num_layers - 1):
			self.weights[j] = self.weights[j] - weight_updates[j]
			self.biases[j] = self.biases[j] - bias_updates[j]

		return errors

	def batch_train(self, X, Y, batch_size, num_epochs, learning_rate, momentum):
		errors = []
		for epoch in range(num_epochs):
			t0 = time.clock()

			for i in range(int(len(X) / batch_size)):
				batch_X = X[i*batch_size:(i+1)*batch_size]
				batch_Y = Y[i*batch_size:(i+1)*batch_size]

				errors.extend(self.update_from_batch(batch_X, batch_Y, learning_rate, momentum))

			print("Epoch: %.2d; Avg. loss: %.5f; Time: %.2f" % (epoch + 1, np.mean(errors), time.clock() - t0))

net = Network((28*28, 128, 10))
net.batch_train(train_X, train_Y, 10, 30, 0.005, 0.9)

# Evaluation

right_answers = 0
for (x, y) in zip(test_X, test_Y):
	if np.argmax(net.feed_forward(x)) == np.argmax(y):
		right_answers += 1
print("Classifier accuracy: %.4f" % (right_answers / test_X.shape[0]))