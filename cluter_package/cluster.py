import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster

input = np.zeros([100, 2])

for index in range(0, 100):
    input[index][0] = index
    input[index][1] = np.random.randint(1, 100)

plt.scatter(input[:, 0], input[:, 1])
plt.show()

km = cluster.KMeans(n_clusters=3)

res = km.fit(input)

center_x = res.cluster_centers_[:, 0]
center_y = res.cluster_centers_[:, 1]

plt.scatter(center_x, center_y)
plt.show()

labels = res.labels_

for index, value in enumerate(input):
    if labels[index] == 0:
        plt.scatter(value[0], value[1], c='red')
    elif labels[index] == 1:
        plt.scatter(value[0], value[1], c='blue')
    else:
        plt.scatter(value[0], value[1], c='yellow')

plt.show()