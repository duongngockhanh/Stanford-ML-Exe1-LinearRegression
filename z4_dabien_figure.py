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

fig = pyplot.figure(figsize=(12, 6))
fig.suptitle("Figure 3: Cost function J(θ)\n", fontsize=14)

position = 221

for i in alpha:
    theta = np.zeros(3)
    theta, J_history = gradientDescentMulti(X, y, theta, i, num_iters)
    ax = fig.add_subplot(position)
    pyplot.plot(np.arange(len(J_history)), J_history, lw=2)
    pyplot.xlabel('Number of iterations')
    pyplot.ylabel('Cost J')
    position += 1

pyplot.show()

# print('theta computed from gradient descent: {:s}'.format(str(theta)))

# price = 0

# X_inputs = [1, 1650, 3]
# X_inputs[1:3]= (X_inputs[1:3] - mu[0:2]) / sigma[0:2]

# price = np.dot(X_inputs, theta)

# print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent): ${:.0f}'.format(price))