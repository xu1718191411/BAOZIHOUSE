import sys, os

from extract_cnn_vgg16_pytorch import VGGNet

sys.path.append(os.pardir)
import h5py
import numpy as np

def get_imlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]


def main():
    list = get_imlist("./dataset")
    features = []
    names = []
    model = VGGNet()
    for l in list:
        feature = model.extractFeat(l)
        features.append(feature.detach().numpy())
        name = os.path.split(l)[1]
        names.append(name)
        print(name)

    output = "INDEX"

    h5f = h5py.File(output, 'w')
    h5f.create_dataset('dataset_1', data=features)
    h5f.create_dataset('dataset_2', data=np.string_(names))
    h5f.close()
    print("finish")

main()