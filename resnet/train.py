from resnet.collate_fn import collater
from resnet.dataset import WeaponDataset
from resnet.image_transform import ImageTransform
from resnet.make_data_path import make_data_path
from resnet.restnet import RestNet50
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

trainPath = "/home/xuzhongwei/Source/Machine_Learning_Project/resnet/data/train"
valPath = "/home/xuzhongwei/Source/Machine_Learning_Project/resnet/data/val"
categories = ["tank", "jet"]

trainData, valData = make_data_path(trainPath, valPath, categories)

transform = ImageTransform(224, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
trainDataset = WeaponDataset(trainData, transform, "train")

valDataset = WeaponDataset(valData, transform, "val")

dataloader = DataLoader(dataset=trainDataset, shuffle=True, batch_size=1, collate_fn=collater)

# dataiter = iter(dataloader)
# image, category = dataiter.next()

model = RestNet50().cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

epochNum = 15
for i in range(epochNum):
    epochLoss = 0
    for index, data in enumerate(dataloader):

        images = data['images']
        categories = data['categories']

        images = images.cuda()
        outputs = model(images)
        categories = categories.cuda()
        optimizer.zero_grad()
        loss = criterion(outputs, categories)
        epochLoss += loss.item()
        loss.backward()
        optimizer.step()

    print("epoch {} total loss: {}".format(i, epochLoss / dataloader.dataset.__len__()))

valDataloader = DataLoader(dataset=valDataset, shuffle=True, batch_size=1)
total = 0
with torch.no_grad():
    for image, category in valDataloader:
        image = image.cuda()
        output = model(image)
        category = category.cuda()

        total += ((torch.argmax(output, dim=1) == category).sum().item())

print("total {}".format(total))
print("predict accuracy : {}".format(total / len(valData)))
