import numpy as np


dense = np.zeros(shape=(6, 5))

block1 = np.array([[1, 2, 1], [0.5, 0.25, -0.25]])
block2 = np.array([[-0.5, 1], [0.75, 1]])

dense[1:3, 0:3] = block1
dense[2:4, 3:5] = block2

print(dense)

vec = np.array([0.5, 0.25, -1, 1.5, -2]).reshape(-1, 1)
print(vec)

print(dense.dot(vec) * 1024)

