from torch.autograd import Variable
from torchvision import models, transforms
from PIL import Image

class VGGNet:
    def __init__(self):
        vgg16 = models.vgg16(pretrained=True)
        self.vgg16 = vgg16

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])

        self.preprocess = preprocess

    def extractFeat(self,imagePath):
        img = Image.open(imagePath)
        img_tensor = self.preprocess(img)
        img_tensor.unsqueeze_(0)
        out = self.vgg16(Variable(img_tensor))
        out = out[0]
        return out