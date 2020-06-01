import numpy as np

x = np.arange(0, 9)
y = np.arange(10, 15)

xCoodinate, yCoordinate = np.meshgrid(x, y)

print(xCoodinate)
print(yCoordinate)