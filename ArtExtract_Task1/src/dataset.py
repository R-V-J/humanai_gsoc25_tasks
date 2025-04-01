import os
import zipfile
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ArtworkDataset(Dataset):
    def __init__(self, csv_file, zip_file_path, class_file, transform=None, extract_path=None):
        """
        Args:
            csv_file: Path to CSV with image paths and labels
            zip_file_path: Path to zip file containing images
            class_file: Path to file with class mappings
            transform: Optional transform to be applied on images
            extract_path: Path to extract images (if None, read directly from zip)
        """
        self.data_frame = pd.read_csv(csv_file)
        self.zip_file_path = zip_file_path
        self.transform = transform
        self.extract_path = extract_path

        self.class_mapping = {}
        with open(class_file, 'r') as f:
            for line in f:
                idx, name = line.strip().split()
                self.class_mapping[int(idx)] = name

        self.num_classes = len(self.class_mapping)
        self.zip_file = None

        if extract_path is None:
            self.zip_file = zipfile.ZipFile(zip_file_path, 'r')

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_path = self.data_frame.iloc[idx, 0]
        label = self.data_frame.iloc[idx, 1]

        if self.extract_path is not None:
            image_full_path = os.path.join(self.extract_path, 'wikiart', img_path)
            try:
                if os.path.exists(image_full_path):
                    img = Image.open(image_full_path).convert('RGB')
                else:
                    print(f"Warning: Image file not found: {image_full_path}")
                    img = Image.new('RGB', (224, 224), (0, 0, 0))
                    label = -1
            except OSError:
                print(f"Warning: Image file corrupted: {image_full_path}")
                img = Image.new('RGB', (224, 224), (0, 0, 0))
                label = -1
        else:
            with self.zip_file.open(img_path) as f:
                img = Image.open(f).convert('RGB')

        if self.transform and img is not None:
            img = self.transform(img)

        return img, label

    def get_class_name(self, class_idx):
        return self.class_mapping.get(class_idx, "Unknown")

    def close(self):
        if self.zip_file:
            self.zip_file.close()

def get_transforms():
    return {
        'train': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }