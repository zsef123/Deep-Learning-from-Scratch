import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D

#bad implementation
def numerical_diff(f ,x):
    h = 10*np.e - 50
    return (f(x+h) - f(x)) / h

def centered_diff(f ,x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / 2*h

def numerical_gradient(f, x):
    h=1e-4
    grad = np.zeros_like(x)
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val 
    return grad

def numerical_gradient_arr(f, X):    
    if X.ndim == 1:
        return numerical_gradient(f, X)
    else:
        grad = np.zeros_like(X)
        
        for idx, x in enumerate(X):
            grad[idx] = numerical_gradient(f, x)
        
        return grad
#y = 0.01 x^2 + 0.1x
def f1(x):
    return 0.01*x**2 + 0.1*x

def f2(x):
    if x.ndim == 1:
        return np.sum(x**2)
    else:
        return np.sum(x**2, axis=1)

def tangent_line(f, x, diff):
    d = diff(f, x)
    print(d)
    y = f(x) - d*x
    return lambda t: d*t + y

def pltShow(x, y, y2):
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(x, y, label = "f(x)")
    plt.plot(x, y2, label = "f'(x)")
    plt.legend()
    plt.show()

def pltShow2d(X, Y, grad):
    plt.figure()
    plt.quiver(X, Y, -grad[0], -grad[1],  angles="xy",color="#666666")#,headwidth=10,scale=40,color="#444444")
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.grid()
    plt.legend()
    plt.draw()
    plt.show()

if __name__=="__main__":
    x = np.arange(0.0, 20.0, 0.1)
    y = f1(x)
    y2 = tangent_line(f1, 5, centered_diff)(x)
    #pltShow(x, y, y2)

    x0 = np.arange(-2, 2.5, 0.25)
    x1 = np.arange(-2, 2.5, 0.25)
    X, Y = np.meshgrid(x0, x1)
    
    X = X.flatten()
    Y = Y.flatten()
    
    grad = numerical_gradient_arr(f2, np.array([X, Y]) )
    pltShow2d(X, Y, grad)