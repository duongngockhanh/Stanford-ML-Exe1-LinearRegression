from matplotlib import pyplot
import os
import numpy as np

data = np.loadtxt(os.path.join('Data', 'ex1data1.txt'), delimiter=',')
X, y = data[:, 0], data[:, 1]
m = y.size

pyplot.plot(X, y, 'ro', ms=10, mec='k')
# pyplot.xlabel('Population of City in 10,000s')
# pyplot.ylabel('Profit in $10,000s')
# pyplot.title("Figure 1: Scatter plot of training data\n", fontsize = 14)

X = np.stack([np.ones(m), X], axis=1)

# print(X)

def computeCost(X, y, theta):
    m = y.size
    J = 0
    J = np.sum(np.power(np.dot(X, theta) - y, 2)) / (2*m)
    return J

# J = computeCost(X, y, theta=np.array([0.0, 0.0]))
# print('With theta = [0, 0] \nCost computed = %.2f' % J)
# print('Expected cost value (approximately) 32.07\n')

# # further testing of the cost function
# J = computeCost(X, y, theta=np.array([-1, 2]))
# print('With theta = [-1, 2]\nCost computed = %.2f' % J)
# print('Expected cost value (approximately) 54.24')

def gradientDescent(X, y, theta, alpha, num_iters):
    m = y.shape[0]
    theta = theta.copy()
    J_history = []
    for i in range(num_iters):
        theta = theta - (alpha/m)*(np.dot(X.T, np.dot(X, theta) - y))
        J_history.append(computeCost(X, y, theta))
    return theta, J_history

# initialize fitting parameters
theta = np.zeros(2)

# some gradient descent settings
iterations = 1500
alpha = 0.01

theta, J_history = gradientDescent(X ,y, theta, alpha, iterations)
print('Theta found by gradient descent: {:.4f}, {:.4f}'.format(*theta))
print('Expected theta values (approximately): [-3.6303, 1.1664]')

# plotData(X[:, 1], y)
pyplot.plot(X[:, 1], np.dot(X, theta), '-')
pyplot.title("Figure 2: Training data with linear regression fit\n", fontsize = 14)
pyplot.legend(['Training data', 'Linear regression'])
# pyplot.show()

# grid over which we will calculate J
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

# initialize J_vals to a matrix of 0's
J_vals = np.zeros((theta0_vals.shape[0], theta1_vals.shape[0]))

# Fill out J_vals
for i, theta0 in enumerate(theta0_vals):
    for j, theta1 in enumerate(theta1_vals):
        J_vals[i, j] = computeCost(X, y, [theta0, theta1])
        
# Because of the way meshgrids work in the surf command, we need to
# transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals.T

# surface plot
fig = pyplot.figure(figsize=(12, 5))
fig.suptitle("Figure 3: Cost function J(θ)\n", fontsize=14)
ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(theta0_vals, theta1_vals, J_vals, cmap='viridis')
pyplot.xlabel('theta0')
pyplot.ylabel('theta1')
pyplot.title('Surface')

# contour plot
# Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
ax = pyplot.subplot(122)
pyplot.contour(theta0_vals, theta1_vals, J_vals, linewidths=2, cmap='viridis', levels=np.logspace(-2, 3, 20))
pyplot.xlabel('theta0')
pyplot.ylabel('theta1')
pyplot.plot(theta[0], theta[1], 'ro', ms=10, lw=2)
pyplot.title('Contour, showing minimum')
pyplot.show()