import numpy as np
from activation import sigmoid

def initNetwork():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5],[0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])    
    network['W2'] = np.array([0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4])
    network['b3'] = np.array([0.1, 0.2])
    return network

# output layer's activation function
def identity(x):
    return x

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y  = identity(a3)
    return y

def batch(network, x, t):
    batchSize = 100
    accuracyCount = 0

    for i in range(len(x), batchSize):
        x_batch = x[i:i+batchSize]
        y_batch = forward(network, x_batch)
        p = np.argmax(y_batch, axis=1)
        accuracyCount += np.sum(p == t[i:i+batchSize])

    return accuracyCount / len(x)

if __name__=="__main__":
    network = initNetwork()
    x = np.array([1.0, 0.5])
    t = np.array([1.2, 0.7])
    y = forward(network, x)

    print("Batch Accur: ", batch(network, x, t))
