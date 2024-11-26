# utils/dataset.py
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

class TinyImageNetDataset:
    def __init__(self, root_dir, split='train', transform=None):
        self.transform = transform or transforms.Compose([
            transforms.Resize(224),  # ViT standard size
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.dataset = datasets.ImageFolder(
            root=f"{root_dir}/{split}",
            transform=self.transform
        )
        
    def get_loader(self, batch_size=32, shuffle=True, num_workers=4):
        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )