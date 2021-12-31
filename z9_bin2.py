import numpy as np
import matplotlib.pyplot as plt

x = np.array([1,2,3,4,5,6,7,8,9,0])
y = x.reshape(-1, 1)
print(y.shape)