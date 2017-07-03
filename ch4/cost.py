import numpy as np

def mse(y, t):
    return 0.5 * np.sum((y-t) ** 2)

def cse(y, t):
    batch_size = 1
    # for apply on batch data's
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        batch_size = y.shape[0]
    return -np.sum( t * np.log(y) ) / batch_size

def cse_label(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    batch_size = y.shape[0]
    return -np.sum( np.log(y[np.arange(batch_size), t])) / batch_size