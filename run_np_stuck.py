import numpy as np

a = np.array([1, 2, 3, 4, 5])
b = np.array([-1, -2, -3, -4, -5])

c = np.hstack((a, b))
print(c)

d = np.vstack((a, b))
print(d)

e = np.stack((a,b))
print(e)
