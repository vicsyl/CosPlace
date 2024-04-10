
import os
from pathlib import Path

import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

import datasets.dataset_utils as dataset_utils


class InferDataset(data.Dataset):
    def __init__(self, dataset_folder, database_folder="database",
                 queries_folder="queries", positive_dist_threshold=25,
                 image_size=512, resize_test_imgs=False):
        self.database_folder = dataset_folder + "/" + database_folder
        self.queries_folder = dataset_folder + "/" + queries_folder
        self.database_paths = dataset_utils.read_images_paths(self.database_folder, get_abs_path=True)
        self.queries_paths = dataset_utils.read_images_paths(self.queries_folder, get_abs_path=True)

        self.dataset_name = os.path.basename(dataset_folder)

        self.images_paths = self.database_paths + self.queries_paths

        self.database_num = len(self.database_paths)
        self.queries_num = len(self.queries_paths)

        transforms_list = []
        if resize_test_imgs:
            # Resize to image_size along the shorter side while maintaining aspect ratio
            transforms_list += [transforms.Resize(image_size, antialias=True)]
        transforms_list += [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        self.base_transform = transforms.Compose(transforms_list)

    @staticmethod
    def open_image(path):
        return Image.open(path).convert("RGB")

    def __getitem__(self, index):
        image_path = self.images_paths[index]
        pil_img = InferDataset.open_image(image_path)
        normalized_img = self.base_transform(pil_img)
        file_name = Path(image_path).name
        return normalized_img, file_name, index

    def __len__(self):
        return len(self.images_paths)

    def __repr__(self):
        return f"< {self.dataset_name} - #q: {self.queries_num}; #db: {self.database_num} >"
