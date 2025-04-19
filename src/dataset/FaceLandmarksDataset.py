import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class FaceLandmarksDataset(Dataset):
    def __init__(
        self,
        data_path: str = r'/kaggle/input/ibug-300w/ibug_300W_large_face_landmark_dataset/labels_ibug_300W_train.xml',
        target_size=(224, 224),
        bbox_scale_factor=1.2,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    ):
        # parse XML
        tree = ET.parse(data_path)
        root = tree.getroot()

        self.image_filenames = []
        self.landmarks = []
        self.crops = []
        self.transform = transform
        self.root_dir = os.path.dirname(data_path)  # thư mục chứa ảnh
        self.target_size = target_size
        self.bbox_scale_factor = bbox_scale_factor

        # đọc thông tin file và landmarks
        for filename in root[2]:
            img_path = os.path.join(self.root_dir, filename.attrib['file'])
            self.image_filenames.append(img_path)
            self.crops.append(filename[0].attrib)

            lm = []
            for num in range(68):
                x = int(filename[0][num].attrib['x'])
                y = int(filename[0][num].attrib['y'])
                lm.append([x, y])
            self.landmarks.append(lm)

        self.landmarks = np.array(self.landmarks, dtype=np.float32)
        assert len(self.image_filenames) == len(self.landmarks)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        # đọc ảnh, nếu lỗi thì next
        while True:
            image = cv2.imread(self.image_filenames[index])
            if image is None:
                index = (index + 1) % len(self.image_filenames)
                continue
            break

        # chuyển về grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        landmarks = self.landmarks[index]
        bbox = self.crops[index]

        # tính crop theo bbox và scale factor
        x1 = int(bbox['left'])
        y1 = int(bbox['top'])
        w  = int(bbox['width'])
        h  = int(bbox['height'])
        new_w = int(w * self.bbox_scale_factor)
        new_h = int(h * self.bbox_scale_factor)
        x1 = max(0, x1 - (new_w - w) // 2)
        y1 = max(0, y1 - (new_h - h) // 2)
        x2 = min(image.shape[1], x1 + new_w)
        y2 = min(image.shape[0], y1 + new_h)
        cropped = image[y1:y2, x1:x2]

        # scale landmarks vào crop
        lm_scaled = landmarks - np.array([x1, y1], dtype=np.float32)

        # resize ảnh và landmarks
        resized = cv2.resize(cropped, self.target_size)
        sx = self.target_size[1] / cropped.shape[1]
        sy = self.target_size[0] / cropped.shape[0]
        lm_resized = np.stack([lm_scaled[:,0] * sx,
                               lm_scaled[:,1] * sy], axis=1).astype(np.float32)

        # transform ảnh (ToTensor + Normalize)
        if self.transform:
            # transform expects HxW or HxWxC; our image is HxW, so add a channel dim
            img_tensor = self.transform(resized[..., None])
        else:
            img_tensor = torch.tensor(resized, dtype=torch.float32).unsqueeze(0) / 255.0

        # flatten landmarks thành 1D tensor
        landmarks_tensor = torch.tensor(lm_resized.flatten(), dtype=torch.float32)

        return img_tensor, landmarks_tensor
