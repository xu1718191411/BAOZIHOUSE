from extract_cnn_vgg16_pytorch import VGGNet
import numpy as np
import h5py

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

h5f = h5py.File("./INDEX","r")
features = h5f["dataset_1"][:]
names = h5f["dataset_2"][:]
h5f.close()

query = "query2.jpg"

model = VGGNet()


queryResult = model.extractFeat(query)
queryResult = queryResult.detach().numpy()

scores = np.dot(queryResult, features.T)
rank_ID = np.argsort(scores)[::-1]
rank_score = scores[rank_ID]

# number of top retrieved images to show
maxres = 3
imlist = [names[index] for i,index in enumerate(rank_ID[0:maxres])]
print("top %d images in order are: " %maxres, imlist)

# show top #maxres retrieved result one by one
for i,im in enumerate(imlist):
    print(str(im,'utf-8'))
