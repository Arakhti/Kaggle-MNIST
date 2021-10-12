from torch.utils.data import Dataset


class MnistTestImageDataset(Dataset):
    def __init__(self, features, transform=None):
        self.features = features
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        image = self.features[idx]
        if self.transform:
            image = self.transform(image)
        return image