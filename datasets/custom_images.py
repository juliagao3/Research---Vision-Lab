import os
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
import json
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, data_root, image_size):
        super().__init__()
        self.data_root = data_root
        self.image_files = [f for f in os.listdir(data_root) if os.path.isfile(os.path.join(data_root, f))]
        # sort by name
        self.image_files.sort()
        self.transform = transforms.Compose([
            # transforms.Lambda(lambda img: img.crop((280, 0, img.width - 280, img.height))),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_root, self.image_files[idx])
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        kpts, masks = self.load_kpts_and_mask(img_path)
        kpts = torch.tensor(kpts)
        masks = torch.tensor(masks)
        sample = {'img': img, 'kpts': kpts, 'visibility': masks}
        return sample

    def __len__(self):
        return len(self.image_files)

    def load_kpts_and_mask(self, path):
        file = os.path.join(os.getcwd(), "pose_estimation/annotation/ak_P1/train.json")
        with open(file, 'r') as f:
            data = json.load(f)
        img = path.split('/')[-1]

        for entry in data:
            img_path = entry.get('image')
            full_image = img_path.split('/')[-1]
            if img == full_image:
                kpt = entry.get('joints')
                mask = entry.get('joints_vis')
                return kpt, mask