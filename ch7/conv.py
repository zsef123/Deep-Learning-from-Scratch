import numpy as np

# Follow book's prototype
# input_data - 4d numpy array, N,C,H,W
def im2col(input_data, filter_h, filter_w, stride = 1, pad = 0):
    N, C, H, W  = input_data.shape
    out_h = (H + 2* pad - filter_h) // stride + 1
    out_w = (W + 2* pad - filter_w) // stride + 1

    img = np.pad(input_data, [(0,0), (0,0) (pad, pad), (pad, pad)], 'constant')
    col = np.zeros(N, C, filter_h, filter_w, out_h, out_w)

    for y in range(filter_h):
        y_filter_range = y + stride * out_h
        for x in range(filter_w):
            x_filter_range = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_filter_range:stride, x:x_filter_range:stride]
    
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col

def col2im(input_data, filter_h, filter_w, stride = 1, pad = 0):
    N, C, H, W = input_data.shape
    out_L = lambda a,b: int((a + 2 * self.pad - b) // self.stride) + 1
    out_H = out_L(H, filter_h)
    out_W = out_L(W, filter_w)

    col = input_data.reshape(N, out_H, out_W, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1 ,2)
    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))

    for y in range(filter_h):
        y_filter_range = y + stride * out_h
        for x in range(filter_w):
            x_filter_range = x + stride * out_w
            img[:, :, y:y_filter_range:stride, x:x_filter_range:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H+pad, pad:W+pad]

class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
    
    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N,  C,  H,  W = x.shape
        
        out_L = lambda a,b: int((a + 2 * self.pad - b) // self.stride) + 1
        out_H = out_L(H, FH)
        out_W = out_L(W, FW)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T
        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_H, out_W, -1).transpose(0,3,1,2)
        return out

    def backward(self, y):
        return out