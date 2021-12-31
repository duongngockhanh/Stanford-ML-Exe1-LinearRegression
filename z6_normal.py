import numpy as np
import os
from matplotlib import pyplot


data = np.loadtxt(os.path.join('Data', 'ex1data2.txt'), delimiter=',')
X = data[:, :2]
y = data[:, 2]
m = y.size
X = np.concatenate([np.ones((m, 1)), X], axis=1)

def normalEqn(X, y):
    theta = np.zeros(X.shape[1])
    theta = np.dot((np.linalg.inv(np.dot(X.T,X))),np.dot(X.T,y))
    return theta

theta = normalEqn(X, y)
# print('Theta computed from the normal equations: {:s}'.format(str(theta)))
price = 0
price = np.dot([1, 1650, 3], theta)
# print('Predicted price of a 1650 sq-ft, 3 br house (using normal equations): ${:.0f}'.format(price))
print(theta)