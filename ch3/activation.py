import numpy as np
import matplotlib.pylab as plt

def step(x):
    return np.array(x > 0, dtype = np.int)

def sigmoid(x):
    return 1 / ( 1 + np.exp(-x))

def ReLU(x):
    return np.maximum(0, x)

def softmax(x):
    y = np.exp(x) / np.sum(np.exp(x))
    # overflow solve
    C = np.max(x)
    y = np.exp(x - C) / np.sum(np.exp(x - C))
    return y

def pltShow(label, x, y):
    plt.label = label
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()
    return
    
if __name__ == "__main__":
    x = np.arange(-5.0, 5.0, 0.1)
    pltShow("Step Function", x, step(x))
    pltShow("Sigmoid Function", x, sigmoid(x))
    pltShow("ReLU Function", x, ReLU(x))
