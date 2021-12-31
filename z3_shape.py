from matplotlib import pyplot
import os
import numpy as np

a = np.array([0, 0]) #Đây là ndarray 1 chiều
print(a.shape) #Trả về (2,)
print(a) # [0 0]

a1 = np.array([[0, 0]]) #Đây là ndarray 2 chiều
print(a1.shape) #Trả về (1, 2)
print(a1) # [[0 0]]

b = np.ones(2) #Đây là ndarray 1 chiều
print(b.shape) #Trả về (2,)
print(b) # [1. 1.]

b1 = np.zeros((1, 2)) #Đây là ndarray 2 chiều
print(b1.shape) #Trả về (1, 2)
print(b1) # [[0. 0.]]

data = np.loadtxt(os.path.join('Data', 'ex1data1.txt'), delimiter=',')
X, y = data[:, 0], data[:, 1] #Đây là ndarray 1 chiều
m = y.shape #Trả về (97,)