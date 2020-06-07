import numpy as np


def softmax(preds):
    maxPreds = np.max(preds, axis=0)
    preds = preds - maxPreds
    x = np.exp(preds)
    y = np.sum(x, axis=0)

    return x / y
