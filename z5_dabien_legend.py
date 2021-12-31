import numpy as np
from matplotlib import pyplot #dùng pyplot là chính
import os #để đọc file ở dòng 5

data = np.loadtxt(os.path.join('Data', 'ex1data2.txt'), delimiter=',')
X = data[:, :2]
y = data[:, 2]
m = y.size

def  featureNormalize(X):
    X_norm = X.copy()
    mu = np.zeros(X.shape[1])
    sigma = np.zeros(X.shape[1])
    
    n = X.shape[1]
    for i in range(n):
        mu[i] = np.mean(X[:,i])
    for i in range(n):
        sigma[i] = np.std(X[:,i])
    for i in range(n):
        X_norm[:,i] = (X[:,i]-mu[i])/sigma[i]   

    return X_norm, mu, sigma

X_norm, mu, sigma = featureNormalize(X)
X = np.concatenate([np.ones((m, 1)), X_norm], axis=1)

def computeCostMulti(X, y, theta):
    m = y.shape[0]
    J = 0
    J = np.sum(np.power(np.dot(X, theta) - y, 2)) / (2*m)
    return J

def gradientDescentMulti(X, y, theta, alpha, num_iters):
    m = y.shape[0]
    theta = theta.copy()
    J_history = []
    for i in range(num_iters):
        theta = theta - (alpha/m)*(np.dot(X.T, np.dot(X, theta) - y))
        J_history.append(computeCostMulti(X, y, theta))
    return theta, J_history

alpha = [0.1, 0.03, 0.01, 0.003]
num_iters = 400

fig, ax = pyplot.subplots()
type_line = ['k--', 'k:', 'k-.', 'k-']
labels = ['a = 0.1', 'a = 0.03', 'a = 0.01', 'a = 0.003']

for i in range(len(alpha)):
    theta = np.zeros(3)
    theta, J_history = gradientDescentMulti(X, y, theta, alpha[i], num_iters)
    pyplot.plot(np.arange(len(J_history)), J_history, type_line[i], lw=2, label=labels[i])
    pyplot.xlabel('Number of iterations')
    pyplot.ylabel('Cost J')

legend = ax.legend(loc='upper right', shadow=True, fontsize='x-large')
pyplot.show()

# print('theta computed from gradient descent: {:s}'.format(str(theta)))

# price = 0

# X_inputs = [1, 1650, 3]
# X_inputs[1:3]= (X_inputs[1:3] - mu[0:2]) / sigma[0:2]

# price = np.dot(X_inputs, theta)

# print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent): ${:.0f}'.format(price))