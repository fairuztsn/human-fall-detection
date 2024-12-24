import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class CustomImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths 
        self.labels = labels
        self.transform = transform

        for i in range(len(self.image_paths)):
            if self.labels is None:
                self.image_paths[i] = os.path.join("..", "data", "test", self.image_paths[i])
            else:
                self.image_paths[i] = "..\\" + self.image_paths[i]

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.labels is None:
            return image
        else:
            label = self.labels[idx]

        return image, label
    
    def to_numpy(self):
        features = []
        labels = []

        for i in range(len(self.image_paths)):
            image_path = self.image_paths[i]
            image = Image.open(image_path).convert("RGB")

            if self.transform:
                image = self.transform(image)
            
            features.append(np.array(image).flatten())

            if self.labels is not None:
                labels.append(self.labels[i])

        features = np.array(features)
        labels = np.array(labels) if self.labels is not None else None

        return features, labels