import numpy as np

from common_functions.common_function import softmax

preds = np.array([2.05823, 5.293, -2.9529, 2.5923, -1.923])
labels = np.array([0, 1, 0, 0, 0])

def cross_entropy_error(preds, labels):
    loss = np.sum(-1 * labels * np.log(preds + 1e-7), axis=0)
    return loss

preds = softmax(preds)
loss = cross_entropy_error(preds,labels)
print(loss)

