import numpy as np
import diff

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        grad = diff.numerical_gradient_arr(f, x)
        x -= lr * grad
    return x

def f(x):
    return x[0]**2 + x[1]**2
if __name__=="__main__":
    x = np.array([-3.0, 4.0])
    # lr is too big
    gradient_descent(f, init_x, 10)
    # lr is too small
    gradient_descent(f,init_x, 1e-10)