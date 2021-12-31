from matplotlib import pyplot
import os
import numpy as np
data = np.loadtxt(os.path.join('Data', 'ex1data1.txt'), delimiter=',')
X, y = data[:, 0], data[:, 1]
m = y.size
pyplot.plot(X, y, 'ro', ms=10, mec='k')
X = np.stack([np.ones(m), X], axis=1)
def gradientDescent(X, y, theta, alpha, num_iters):
    m = y.shape[0]
    theta = theta.copy()
    for i in range(num_iters):
        theta = theta - (alpha/m)*(np.dot(X.T, np.dot(X, theta) - y))
    return theta
theta = np.zeros(2)
iterations = 1500
alpha = 0.01
theta = gradientDescent(X ,y, theta, alpha, iterations)
print('Theta found by gradient descent: {:.4f}, {:.4f}'.format(*theta))
print('Expected theta values (approximately): [-3.6303, 1.1664]')
pyplot.plot(X[:, 1], np.dot(X, theta), '-')
pyplot.title("Figure 2: Training data with linear regression fit\n", fontsize = 14)
pyplot.legend(['Training data', 'Linear regression'])
pyplot.show()