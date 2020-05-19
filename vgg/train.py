import torch
import torchvision
import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import accuracy_score

model = torchvision.models.vgg16(pretrained=True)
for name, param in model.named_parameters():
    print(name)

if __name__ == '__main__':
    print("main")
    path = os.path.dirname(__file__)
    root = os.path.join(path, "..", "data", "cifar100")
    transform = transforms.Compose([transforms.Resize(64), transforms.ToTensor()])

    train_data_set = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=transform)

    test_data_set = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=transform)

    train_data_loader = DataLoader(dataset=train_data_set, shuffle=True, batch_size=32)

    test_data_loader = DataLoader(dataset=test_data_set, shuffle=True, batch_size=32)

    epoch = 50

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model.classifier[6] = nn.Linear(in_features=4096, out_features=100, bias=True)
    model.cuda(device)
    weights = torch.load("./weights/epoch_6.pth")
    model.load_state_dict(weights)

    learning_param_names0 = ["features"]
    learning_param_names1 = ["classifier.0.weight", "classifier.0.bias", "classifier.3.weight", "classifier.3.bias"]
    learning_param_names2 = ["classifier.6.weight", "classifier.6.bias"]

    learning_params0 = []
    learning_params1 = []
    learning_params2 = []
    for name, params in model.named_parameters():
        print(name)
        if learning_param_names0[0] in name:
            params.requires_grad = True
            learning_params0.append(params)
        elif name in learning_param_names1:
            params.requires_grad = True
            learning_params1.append(params)
        elif name in learning_param_names2:
            params.requires_grad = True
            learning_params2.append(params)
        else:
            params.requires_grad = False

    optimizer = torch.optim.SGD([
        {"params": learning_params0, "lr": 1e-4, "momentum": 0.9},
        {"params": learning_params1, "lr": 5e-4, "momentum": 0.9},
        {"params": learning_params2, "lr": 1e-3, "momentum": 0.9},
    ])

    criterion = nn.CrossEntropyLoss()
    for i in range(7, epoch):
        print("training epoch {}".format(i + 1))
        epoch_loss = 0

        for x, t in train_data_loader:
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                x = x.to(device)
                t = t.to(device)
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
            with torch.set_grad_enabled(False):
                model.eval()
                output = model(x)
                loss = criterion(output, t)
                test_loss = test_loss + loss.item()

                output = torch.argmax(output, dim=-1)

                output = output.cpu().tolist()
                t = t.tolist()
                score = accuracy_score(t, output)
                test_acc = test_acc + score

        print("epoch {} loss: {} acc: {}".format((i + 1), test_loss / test_data_loader.__len__(),
                                                 test_acc / test_data_loader.__len__()))

        torch.save(model.state_dict(), "./weights/epoch_{}.pth".format(i + 1))
