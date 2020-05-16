import torchvision
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import accuracy_score

model = torchvision.models.resnet50(
    pretrained=True)  # if pretrained is False, the loss at first will be larger than 4.5

if __name__ == '__main__':
    print("main")
    path = os.path.dirname(__file__)
    root = os.path.join(path, "..", "data", "cifar100")
    transform = transforms.Compose([transforms.ToTensor()])

    train_data_set = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=transform)

    test_data_set = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=transform)

    train_data_loader = DataLoader(dataset=train_data_set, shuffle=True, batch_size=32)

    test_data_loader = DataLoader(dataset=test_data_set, shuffle=True, batch_size=32)

    epoch = 50

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model.fc = nn.Linear(in_features=2048, out_features=100, bias=True)
    model.cuda(device)
    learning_param_names = ["fc.weight", "fc.bias"]
    learning_params = []
    for name, params in model.named_parameters():
        print(name)
        if name in learning_param_names:
            params.requires_grad = True
            learning_params.append(params)
        else:
            params.requires_grad = False

    optimizer = torch.optim.SGD(learning_params, lr=0.01)

    criterion = nn.CrossEntropyLoss()
    for i in range(epoch):
        print("training epoch {}".format(i + 1))
        epoch_loss = 0

        for x, t in train_data_loader:
            x = x.to(device)
            t = t.to(device)

            optimizer.zero_grad()
            model.train()
            output = model(x)

            loss = criterion(output, t)
            epoch_loss = epoch_loss + loss.item()
            loss.backward()
            optimizer.step()

        print("epoch {} loss: {}".format((i + 1), epoch_loss / train_data_loader.__len__()))

        test_loss = 0
        test_acc = 0
        for x, t in test_data_loader:
            t = t.to(device)
            x = x.to(device)

            model.eval()
            output = model(x)

            loss = criterion(output, t)
            test_loss = test_loss + loss.item()

            output = torch.argmax(output, dim=-1)

            output = output.cpu().tolist()
            t = t.tolist()
            score = accuracy_score(t, output)
            test_acc = test_acc + score

        print("epoch {} loss: {} acc: {}".format((i+1),test_loss / test_data_loader.__len__(), test_acc / test_data_loader.__len__()))
