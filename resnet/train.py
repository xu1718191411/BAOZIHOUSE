import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optimizers
from torch.utils.data import dataloader
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score

from resnet.net import ResNet50

if __name__ == '__main__':
    np.random.seed(1234)
    torch.manual_seed(1234)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    root = os.path.join(os.path.dirname(__file__), '..', 'data', "fashion_minist")
    transform = transforms.Compose([transforms.ToTensor()])

    mnist_train = torchvision.datasets.FashionMNIST(root=root, transform=transform, download=True, train=True)

    mnist_test = torchvision.datasets.FashionMNIST(root=root, transform=transform, download=True, train=False)

    train_data_loader = dataloader.DataLoader(mnist_train, shuffle=True, batch_size=100)
    test_data_loader = dataloader.DataLoader(mnist_test, shuffle=True, batch_size=100)

    model = ResNet50(10).to(device)

    criterion = nn.NLLLoss()
    optimizer = optimizers.Adam(model.parameters(), weight_decay=.01)

    epoch = 50

    for i in range(epoch):
        print("epoch {}".format(i + 1))
        train_loss = 0
        test_loss = 0
        test_acc = 0
        for x, t in train_data_loader:
            x, t = x.to(device), t.to(device)
            model.train()
            preds = model(x)
            loss = criterion(preds, t)
            train_loss = train_loss + loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(train_loss/train_data_loader.__len__())

        for x, t in test_data_loader:
            x, t = x.to(device), t.to(device)
            model.eval()
            preds = model(x)
            loss = criterion(preds, t)
            test_loss = test_loss + loss.item()
            test_acc = test_acc + accuracy_score(t.tolist(), preds.argmax(dim=-1).tolist())

        test_loss = test_loss / len(test_data_loader)
        test_acc = test_acc / len(test_data_loader)
        print("Epoch {}, valid cost {:.3f}, valid acc {:.3f}".format(i, test_loss, test_acc))


        torch.save(model.state_dict(),"./trainning_result/epoch_{}.pth".format(i+1))
