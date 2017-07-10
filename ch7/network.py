import conv
import layers as Layers # book's file ## common/layers.py
from collections import OrderedDict
class CNN:
    def __init__(self, input_dim=(1,28,28),
                 conv_param={'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1},
                 hidden_size=100, output_size=10, weight_init_std=0.01):
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']

        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2 * filter_pad) / filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))

        self.params={}
        self.params['W1'] = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        self.params['W2'] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['Conv1'] = conv.Convolution(self.params['W1'], self.params['b1'], filter_stride, filter_pad)
        self.layers['ReLU1'] = Layers.ReLU()
        self.layers['Pool1'] = conv.Pool(2,2,2)
        self.layers['Affine1'] =Layers.Affine(self.params['W2'], self.params['b2'])
        self.layers['ReLU2'] = Layers.ReLU()
        self.layers['Affine2'] = Layers.Affine(self.params['W3'], self.params['b3'])
        self.last_layer = Layers.SoftmaxWithLoss()
    
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.foward(x)
        return x

    def loss(self, x, t):
        y = predict(x)
        return self.last_layer.forward(y, t)

    def gradient(self, x, t):
        selt.loss(x, t)

        dout = self.last_layer.backward(1)
        
        for layer in list(self.layers.values()).reverse():
            dout = layer.backward(dout)

        grads = {
            'W1':self.layers['Conv1'].dW
            'b1':self.layers['Conv1'].db
            'W2':self.layers['Affine1'].dW
            'b2':self.layers['Affine1'].db
            'W3':self.layers['Affine2'].dW
            'b3':self.layers['Affine2'].db
        }
        return grads
        