import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x*np.cos(0.25*np.pi*x)

def get_grad(x):
    return np.cos(0.25*np.pi*x) - 0.25*np.pi*x*np.sin(0.25*np.pi*x)

def get_RMS(alist):
    square = [x*x for x in alist]
    return np.sqrt(sum(square) / len(alist))

def gradient_descent(start, iter=10, lr=0.4):
    x = start
    xs = [x]
    ys = [f(x)]
    for _ in range(iter):
        grad = get_grad(x)
        x -= grad*lr
        xs.append(x)
        ys.append(f(x))
    return xs, ys

def SGD(start, iter=10, lr=0.4):
    x = start
    xs = [x]
    ys = [f(x)]
    for _ in range(iter):
        grad = get_grad(x) + np.random.randn()
        x -= grad*lr
        xs.append(x)
        ys.append(f(x))
    return xs, ys

def Adagrad(start, iter=10, lr=0.4, e=1e-6):
    x = start
    xs = [x]
    ys = [f(x)]
    grads = []
    for t in range(iter):
        grad = get_grad(x)
        grads.append(grad)
        sigma = get_RMS(grads) + e
        x = x - lr * grad / np.sqrt(t + 1) / sigma
        xs.append(x)
        ys.append(f(x))
    return xs, ys

def RMSProp(start, iter=10, lr=0.4, alpha=0.9, e=1e-6):
    x = start
    xs = [x]
    ys = [f(x)]
    sigma = 0
    for _ in range(iter):
        grad = get_grad(x)
        sigma = np.sqrt(alpha*sigma*sigma + (1-alpha)*grad*grad) + e
        x = x - lr * grad / sigma
        xs.append(x)
        ys.append(f(x))
    return xs, ys

def momentum(start, iter=10, lr=0.4, gamma=0.9):
    x = start
    xs = [x]
    ys = [f(x)]
    m = 0
    for _ in range(iter):
        grad = get_grad(x)
        m = gamma * m + lr * grad
        x -= m
        xs.append(x)
        ys.append(f(x))
    return xs, ys

def adam(start, iter=10, lr=.4, beta1=.9, beta2=.999, e=1e-6):
    x = start
    xs = [x]
    ys = [f(x)]
    sigma = 0
    m = 0
    for t in range(iter):
        grad = get_grad(x)
        m = beta1 * m + (1-beta1)*grad
        sigma = beta2 * sigma + (1 - beta2) * grad * grad
        # Adjust the bias
        m = m / (1 - beta1**(t+1))
        sigma = sigma / (1 - beta2**(t+1))
        # Update arg
        x = x - lr * m / (np.sqrt(sigma) + e)
        xs.append(x)
        ys.append(f(x))
    return xs, ys

def viz(start, iter=10, lr=.4, alpha=.9, gamma=.9, beta1=.9, beta2=.999):
    fig, axs = plt.subplots(2,3, figsize=(12,8))
    
    xs, ys = gradient_descent(start, iter, lr)
    axs[0, 0].plot(xs, ys, marker='o', markersize=6)
    axs[0, 0].set_title('GD')
    axs[0, 0].set_xlabel('X Axis')
    axs[0, 0].set_ylabel('Y Axis')
            
    xs, ys = SGD(start, iter, lr)
    axs[0, 1].plot(xs, ys, marker='o',markersize=6)
    axs[0, 1].set_title('SGD')
    axs[0, 1].set_xlabel('X Axis')
    axs[0, 1].set_ylabel('Y Axis')

    xs, ys = Adagrad(start, iter, lr)
    axs[0, 2].plot(xs, ys, marker='o',markersize=6)
    axs[0, 2].set_title('Adagrad')
    axs[0, 2].set_xlabel('X Axis')
    axs[0, 2].set_ylabel('Y Axis')

    xs, ys = RMSProp(start, iter, lr, alpha=alpha)
    axs[1, 0].plot(xs, ys, marker='o',markersize=6)
    axs[1, 0].set_title('RMSProp')
    axs[1, 0].set_xlabel('X Axis')
    axs[1, 0].set_ylabel('Y Axis')

    xs, ys = momentum(start, iter, lr, gamma=gamma)
    axs[1, 1].plot(xs, ys, marker='o',markersize=6)
    axs[1, 1].set_title('Momentum')
    axs[1, 1].set_xlabel('X Axis')
    axs[1, 1].set_ylabel('Y Axis')

    xs, ys = adam(start, iter, lr, beta1=beta1, beta2=beta2)
    axs[1, 2].plot(xs, ys, marker='o',markersize=6)
    axs[1, 2].set_title('Adam')
    axs[1, 2].set_xlabel('X Axis')
    axs[1, 2].set_ylabel('Y Axis')

    plt.tight_layout()
    plt.show()

viz(start=-4, iter=50, lr=0.4, alpha=.5,gamma=.3, beta1=.3, beta2=.5)




