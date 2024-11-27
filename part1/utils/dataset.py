import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os
import shutil

class TinyImageNetDataset:
    def __init__(self, root_dir, split='train', transform=None):
        if split not in ['train', 'val', 'test']:
            raise ValueError(f"Split {split} not recognized. Use 'train', 'val', or 'test'")
            
        self.transform = transform or transforms.Compose([
            transforms.Resize(224),  # ViT standard size
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Load class mapping
        wnids_path = os.path.join(root_dir, 'wnids.txt')
        with open(wnids_path, 'r') as f:
            self.class_to_idx = {line.strip(): idx for idx, line in enumerate(f.readlines())}
        
        # For validation set, restructure if needed
        if split == 'val':
            val_dir = os.path.join(root_dir, 'val')
            images_dir = os.path.join(val_dir, 'images')
            if os.path.exists(images_dir):  # Need to restructure
                self._restructure_validation_set(val_dir)
        
        # Create dataset
        split_dir = os.path.join(root_dir, split)
        if not os.path.exists(split_dir):
            raise ValueError(f"Directory not found: {split_dir}")
            
        self.dataset = datasets.ImageFolder(
            root=split_dir,
            transform=self.transform
        )
        
        # Update class indices to match wnids.txt
        class_to_idx_reverse = {v: k for k, v in self.dataset.class_to_idx.items()}
        new_targets = []
        for target in self.dataset.targets:
            class_name = class_to_idx_reverse[target]
            new_target = self.class_to_idx[class_name]
            new_targets.append(new_target)
        self.dataset.targets = new_targets
        self.dataset.class_to_idx = self.class_to_idx
        
        # Print dataset information
        print(f"\nDataset Info ({split} split):")
        print(f"Number of images: {len(self.dataset)}")
        print(f"Number of classes: {len(self.dataset.classes)}")
        print(f"First few class indices: {dict(list(self.class_to_idx.items())[:5])}")
        print(f"Sample targets: {self.dataset.targets[:5]}")
        
    def _restructure_validation_set(self, val_dir):
        """Restructure validation set to match ImageFolder format"""
        print("Restructuring validation set...")
        
        # Read validation annotations
        val_annotations_file = os.path.join(val_dir, 'val_annotations.txt')
        images_dir = os.path.join(val_dir, 'images')
        
        if not os.path.exists(val_annotations_file):
            raise ValueError(f"Validation annotations file not found: {val_annotations_file}")
            
        # Create class directories
        for class_id in self.class_to_idx.keys():
            os.makedirs(os.path.join(val_dir, class_id), exist_ok=True)
        
        # Move images to class directories
        with open(val_annotations_file, 'r') as f:
            for line in f:
                img_name, class_id, *_ = line.strip().split('\t')
                src = os.path.join(images_dir, img_name)
                dst = os.path.join(val_dir, class_id, img_name)
                if os.path.exists(src):
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy2(src, dst)  # Use copy2 instead of move
        
        # Clean up only after successful copy
        if os.path.exists(images_dir) and all(os.path.exists(os.path.join(val_dir, class_id)) 
                                            for class_id in self.class_to_idx.keys()):
            shutil.rmtree(images_dir)
            
        print("Validation set restructured.")
    
    def get_loader(self, batch_size=32, shuffle=True, num_workers=4):
        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )