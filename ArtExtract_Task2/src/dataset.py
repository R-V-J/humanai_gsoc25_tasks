import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class ImageDataset(Dataset):

    def __init__(self, dataFrame, image_dir="./data/images", transform=None):
        self.dataFrame = dataFrame
        self.image_dir = image_dir
        self.transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Resize((200,200)),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, str(self.dataFrame.iloc[idx]['objectid'])+'.jpg')
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image

    def __len__(self):
        return len(self.dataFrame)