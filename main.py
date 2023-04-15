from math import exp
import numpy as np
import matplotlib.pyplot as plt


def simulated_boolean_function(x1, x2, x3, x4):
    return int(((not x1) or (not x2) or (not x3)) and ((not x2) or (not x3) or x4))


X1 = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
X2 = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]
X3 = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
X4 = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

Y = np.array([simulated_boolean_function(X1[i], X2[i], X3[i], X4[i]) for i in range(16)])
print('Y = ', Y)

X_test = [[X1[i], X2[i], X3[i], X4[i]] for i in range(16)]
C = np.array([[0, 1, 1, 0],
              [0, 1, 1, 1],
              [1, 1, 1, 1]])

lr = 0.3


def loss(y_true, y_predicted):
    return y_true - y_predicted


def get_predict(net, X_test):
    y_predicted = [net.forward(x) for x in X_test]
    return np.array(y_predicted)


class RadialBasisFunctionNeuron:
    def __init__(self, center):
        self.center = center

    def get_phi(self, X):
        summary = np.sum((X - self.center) ** 2)
        return exp(-summary)


class BinaryStepNet:

    def __init__(self, neurons_count, centers):
        self.neurons_count = neurons_count
        self.neuron = []
        for i in range(self.neurons_count):
            self.neuron.append(RadialBasisFunctionNeuron(centers[i]))
        self.W = np.zeros(shape=neurons_count)
        self.b = 0

    def forward(self, x):
        self.phi = []
        for i in range(self.neurons_count):
            self.phi.append(self.neuron[i].get_phi(x))
        self.phi = np.array(self.phi)
        self.net = np.dot(self.W, self.phi) + self.b
        self.binary_step = int(self.net >= 0)
        return self.binary_step

    def backward(self, delta, lr=0.3):
        self.dW = np.dot(lr * delta, self.phi)
        self.db = lr * delta
        self.W = self.W + self.dW
        self.b = self.b + self.db


# Пробуем обучиться на всех данных
net = BinaryStepNet(3, C)
L_iter = []
size_of_train = 16

for epoch in range(100):
    error = 0.
    # print("\n\tEpoch ", epoch)
    Y_predicted = []
    for i in range(size_of_train):
        x = [X1[i], X2[i], X3[i], X4[i]]
        y_true = Y[i]
        y_predicted = net.forward(x)
        delta = loss(y_true, y_predicted)
        error += np.abs(delta)
        net.backward(delta, lr)
        Y_predicted.append(y_predicted)
    L_iter.append(error)
    # print("W = ", net.W, 'b = ', net.b)
    # print("Y_pred = ", np.array(Y_predicted))
    # print("Y_true = ", Y)
    # print("E = ", error)
    if (error == 0):
        break

fig, ax = plt.subplots()
ax.plot(L_iter)
ax.set_xlabel('epoch number')
ax.set_ylabel('error')
ax.set_title('Error(epoch)')
plt.grid()
plt.show()

# Обучаемся на полученной наименьшей комбинации
net = BinaryStepNet(3, C)
L_iter = []
idxs = [1, 2, 7, 14, 15]

for epoch in range(100):
    error_on_train = 0.
    # print("\n\tEpoch ", epoch)
    for i in idxs:
        x = [X1[i], X2[i], X3[i], X4[i]]
        y_true = Y[i]
        y_predicted = net.forward(x)
        delta = loss(y_true, y_predicted)
        error_on_train += abs(delta)
        net.backward(delta, lr)
    Y_predicted = get_predict(net, X_test)
    error_on_test = np.sum(np.abs(loss(Y_predicted, Y)))
    L_iter.append(error_on_test)
    # print("W = ", net.W)
    # print("Y = ", Y_predicted)
    # print("E = ", error_on_test)
    if (error_on_test == 0):
        break

# print("\nfinal W = ", net.W)
fig, ax = plt.subplots()
ax.plot(L_iter)
ax.set_xlabel('epoch number')
ax.set_ylabel('error')
ax.set_title('Error(epoch)')
plt.grid()
plt.show()


class SoftsignNet:

    def __init__(self, neurons_count, centers):
        self.neurons_count = neurons_count
        self.neuron = []
        for i in range(self.neurons_count):
            self.neuron.append(RadialBasisFunctionNeuron(centers[i]))
        self.W = np.zeros(shape=neurons_count)
        self.b = 0

    def forward(self, x, threshold=0.5):
        self.phi = []
        for i in range(self.neurons_count):
            self.phi.append(self.neuron[i].get_phi(x))
        self.phi = np.array(self.phi)
        self.net = np.dot(self.W, self.phi) + self.b
        self.softsign = 0.5 * (self.net / (1 + np.abs(self.net)) + 1)
        return int(self.softsign > threshold)

    def backward(self, delta, lr=0.3):
        self.dz = 0.5 / (1 + np.abs(self.softsign) ** 2)
        self.dW = np.dot(lr * delta * self.dz, self.phi)
        self.db = lr * delta * self.dz
        self.W = self.W + self.dW
        self.b = self.b + self.db


net = SoftsignNet(3, C)
L_iter = []
size_of_train = 16

for epoch in range(100):
    error = 0.
    # print("\n\tEpoch ", epoch)
    Y_predicted = []
    for i in range(size_of_train):
        x = [X1[i], X2[i], X3[i], X4[i]]
        y_true = Y[i]
        y_predicted = net.forward(x)
        delta = loss(y_true, y_predicted)
        error += abs(delta)
        net.backward(delta, lr)
        Y_predicted.append(y_predicted)
    L_iter.append(error)
    # print("W = ", net.W, 'b = ', net.b)
    # print("Y_pred = ", np.array(Y_predicted))
    # print("Y_true = ", Y)
    # print("E = ", error)
    if (error == 0):
        break

fig, ax = plt.subplots()
ax.plot(L_iter)
ax.set_xlabel('epoch number')
ax.set_ylabel('error')
ax.set_title('Error(epoch)')
plt.grid()
plt.show()

# Обучаемся на полученной наименьшей комбинации
net = SoftsignNet(3, C)
L_iter = []
idxs = [6, 7, 10, 15]

for epoch in range(100):
    error_on_train = 0.
    # print("\n\tEpoch ", epoch)
    for i in idxs:
        x = [X1[i], X2[i], X3[i], X4[i]]
        y_true = Y[i]
        y_predicted = net.forward(x)
        delta = loss(y_true, y_predicted)
        error_on_train += abs(delta)
        net.backward(delta, lr)
    Y_predicted = get_predict(net, X_test)
    error_on_test = np.sum(np.abs(loss(Y_predicted, Y)))
    L_iter.append(error_on_test)
    # print("W = ", net.W)
    # print("Y = ", Y_predicted)
    # print("E = ", error_on_test)
    if (error_on_test == 0):
        break

# print("\nfinal W = ", net.W)
fig, ax = plt.subplots()
ax.plot(L_iter)
ax.set_xlabel('epoch number')
ax.set_ylabel('error')
ax.set_title('Error(epoch)')
plt.grid()
plt.show()
