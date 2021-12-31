# Dùng thuật toán GD để tìm min của hàm fx = x^2 + 2x + 5
import numpy as np
import matplotlib.pyplot as plt

x = 0
X = np.array([x])

for i in range(600):
    x = x - 0.1*(2*x+2)
    X = np.concatenate((X,[x]))

y = X**2
plt.plot(y)
plt.xlabel('Số lần')
plt.ylabel('f(x)')
plt.title('Giá trị f(x) sau số lần thực hiện bước 2')
plt.show()