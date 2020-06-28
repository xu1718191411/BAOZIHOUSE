from torchvision import transforms


class ImageTransform:
    def __init__(self, resize, mean, std):
        self.train_data_transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        self.eval_data_transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def __call__(self, image, phase):
        if phase == 'train':
            return self.train_data_transform(image)
        else:
            return self.eval_data_transform(image)
