import torch
from torch.utils.data import DataLoader

from resnet.collate_fn import collater
from resnet.dataset import WeaponDataset
from resnet.image_transform import ImageTransform
from torchvision.models.resnet import resnet50
import torch.optim as optim
import torch.utils

from resnet.make_data_path import make_data_path

trainDataPath = "/home/xuzhongwei/Source/Machine_Learning_Project/resnet/dataset/train"
valDataPath = "/home/xuzhongwei/Source/Machine_Learning_Project/resnet/dataset/val"
categories = ["f15", "f16", "f18"]

trainDataPath, valDataPath = make_data_path(trainDataPath, valDataPath, categories)

trainTransform = ImageTransform(224, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

trainDataSet = WeaponDataset(trainDataPath, trainTransform, "train")
valDataSet = WeaponDataset(valDataPath, trainTransform, "val")

trainDataLoader = DataLoader(trainDataSet, shuffle=True, batch_size=1, collate_fn=collater)
valDataLoader = DataLoader(valDataSet, shuffle=True, batch_size=1, collate_fn=collater)

model = resnet50(pretrained=True)
model = model.train()
model.fc = torch.nn.Linear(in_features=2048, out_features=3, bias=True)
model = model.cuda()

print("traindataset {}".format(trainDataLoader.dataset.__len__()))

learning_parameters = ["fc.weight", "fc.bias"]
update_parameters = []
for name, param in model.named_parameters():
    if name in learning_parameters:
        update_parameters.append(param)

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

criterion = torch.nn.CrossEntropyLoss()

PATH = "./trainning_result/train_by_offical_model_39.pth"
model.load_state_dict(torch.load(PATH))
epoch = 50
startEpoch = 39
for epochIndex in range(startEpoch, epoch):
    epochLoss = 0

    for batchIndex, data in enumerate(trainDataLoader):
        optimizer.zero_grad()

        input = data["images"].cuda()
        categories = data["categories"].cuda()
        output = model(input)
        loss = criterion(output, categories)
        loss.backward()
        optimizer.step()
        epochLoss += loss.item()

    PATH = './trainning_result/train_by_offical_model_{}.pth'.format(epochIndex + 1)
    torch.save(model.state_dict(), PATH)

    print("epoch {} loss : {}".format(epochIndex, epochLoss / len(trainDataLoader.dataset)))

print("training finished")
PATH = './trainning_result/train_by_offical_model_final.pth'
torch.save(model.state_dict(), PATH)

print("start evaluating")
model.load_state_dict(torch.load(PATH))
model = model.eval()

total = 0
for batchIndex, data in enumerate(valDataLoader):
    input = data["images"].cuda()
    categories = data["categories"].cuda()
    output = model(input)
    pred = torch.argmax(output, dim=1)
    s = torch.sum(pred == categories)
    total += s

print("accuracy: {}".format(int(total) / len(valDataLoader.dataset)))
