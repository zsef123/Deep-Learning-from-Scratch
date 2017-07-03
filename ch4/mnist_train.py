import numpy as np
import sys, os
sys.path.insert(0, os.pardir)
from dataset.mnist import load_mnist
from ch3.activation import sigmoid
from ch3.activation import softmax
from cost import cse_label
from diff import numerical_gradient_arr as getGrad
def miniBatch(x, t, batch_size=10):
    train_size = x.shape[0]
    batch_mask = np.random.choice(train_size, batch_size)

    x_batch = x[batch_mask]
    t_batch = t[batch_mask]
    return (x_batch, t_batch)

class LayerNetwork:
    def __init__(self, input_size, hidden_size, output_size, weight_init=0.01):
        self.params = {}
        self.params['W1'] = weight_init * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        y0 = x * self.params['W1'] + self.params['b1']
        z0 = sigmoid(y0)
        y1 = z0 * self.params['W2'] + self.params['b2']
        z1 = softmax(y1)
        return z1

    def loss(self, x, t):
        y = self.predict(x)
        return cse_label(y, t)

    def accuracy(selt, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        return np.sum(y==t) / float(x.shape[0])

    def grad(self, x, t):
        lossW = lambda W : self.loss(x, t)
        grads = {}
        for (param, v) in self.params:
            grads[param] = getGrad(lossW, v)
        return grads

def pltShow(y1, y2):
    markers = {'train': 'o', 'test': 's'}
    x = np.arange(len(train_acc_list))
    plt.plot(x, y1, label='train acc')
    plt.plot(x, y2, label='test acc', linestyle='--')
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    plt.show()

if __name__ == "__main__":
    net = LayerNetwork(784, 100, 10)
    (x_train, t_train), (x_test, t_test) = load_mnist(one_hot_label=True)
    train_loss = []
    train_acc = []
    test_acc = []
    
    lr = 0.01
    batch_size = 100
    epoch = max(x_train.shape[0] / batch_size, 1)
    for i in range(10000):
        x_batch, t_batch = miniBatch(x_train, t_train, batch_size)
        grad = net.grad(x_batch, t_batch)
        for key in net.params.keys():
            net.params[k] -= lr * grad[k]
        loss = net.loss(x_batch, t_batch)
        train_loss.append(loss)

        if i % epoch == 0:
            train_acc.append(net.accuracy(x_train, t_train))
            test_acc.append(net.accuracy(x_test, t_test))
        
            
    pltShow(train_acc, test_acc)