import numpy as np
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt


input = np.random.randint(1, 100, [100, 2])

plt.scatter(input[:, 0], input[:, 1])
plt.show()

km = MiniBatchKMeans(n_clusters=3).fit(input)

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