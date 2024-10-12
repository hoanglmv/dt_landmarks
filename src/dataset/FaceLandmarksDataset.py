import torch
from torch.utils.data import Dataset
import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image

class FaceLandmarksDataset(Dataset):
    def __init__(self, data_path: str = 'D:\lmvh\Detect_Facial_Landmarks\data_image\ibug_300W_large_face_landmark_dataset\labels_ibug_300W_train.xml',
                 transform=None, target_size=(224, 224), bbox_scale_factor=1.2, apply_augmentation=False):
        tree = ET.parse(data_path)
        root = tree.getroot()

        self.image_filenames = []
        self.landmarks = []
        self.crops = []
        self.transform = transform
        self.root_dir = 'D:\lmvh\Detect_Facial_Landmarks\data_image\ibug_300W_large_face_landmark_dataset'
        self.target_size = target_size
        self.bbox_scale_factor = bbox_scale_factor
        self.apply_augmentation = apply_augmentation  # Thêm cờ để bật tắt augment

        for filename in root[2]:
            self.image_filenames.append(os.path.join(self.root_dir, filename.attrib['file']))
            self.crops.append(filename[0].attrib)

            landmark = []
            for num in range(68):
                x_coordinate = int(filename[0][num].attrib['x'])
                y_coordinate = int(filename[0][num].attrib['y'])
                landmark.append([x_coordinate, y_coordinate])
            self.landmarks.append(landmark)

        self.landmarks = np.array(self.landmarks).astype('float32')
        assert len(self.image_filenames) == len(self.landmarks)

       

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        while True:
            image = cv2.imread(self.image_filenames[index])

            if image is None:
                index = (index + 1) % len(self.image_filenames)
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            landmarks = self.landmarks[index]
            bbox = self.crops[index]

            x1 = int(bbox['left'])
            y1 = int(bbox['top'])
            x2 = int(bbox['left']) + int(bbox['width'])
            y2 = int(bbox['top']) + int(bbox['height'])
            width = x2 - x1
            height = y2 - y1
            new_width = int(width * self.bbox_scale_factor)
            new_height = int(height * self.bbox_scale_factor)
            x1 = max(0, x1 - (new_width - width) // 2)
            y1 = max(0, y1 - (new_height - height) // 2)
            x2 = min(image.shape[1], x1 + new_width)
            y2 = min(image.shape[0], y1 + new_height)
            cropped_image = image[y1:y2, x1:x2]

            scaled_landmarks = []
            for landmark in landmarks:
                x_scaled = landmark[0] - x1
                y_scaled = landmark[1] - y1
                scaled_landmarks.append([x_scaled, y_scaled])

            scaled_landmarks = np.array(scaled_landmarks).astype('float32')
            resized_image = cv2.resize(cropped_image, self.target_size)

            scale_x = self.target_size[1] / cropped_image.shape[1]
            scale_y = self.target_size[0] / cropped_image.shape[0]
            final_landmarks = []
            for landmark in scaled_landmarks:
                x_resized = landmark[0] * scale_x
                y_resized = landmark[1] * scale_y
                final_landmarks.append([x_resized, y_resized])
            final_landmarks = np.array(final_landmarks).astype('float32')

            # Thêm augmentation cho ảnh và landmarks
            if self.apply_augmentation:
                # Convert image to PIL for augmentation
                resized_image_pil = Image.fromarray(resized_image)
                original_size = resized_image_pil.size

                # Apply random flip
                if np.random.rand() > 0.5:  # Xác suất lật ảnh 50%
                    resized_image_pil = resized_image_pil.transpose(Image.FLIP_LEFT_RIGHT)
                    final_landmarks[:, 0] = original_size[0] - final_landmarks[:, 0]  # Lật ngược tọa độ x

                # Apply random rotation
                angle = np.random.uniform(-30, 30)
                resized_image_pil = resized_image_pil.rotate(angle)

                # Tính ma trận xoay
                rotation_matrix = cv2.getRotationMatrix2D((original_size[0] / 2, original_size[1] / 2), angle, 1)
                # Thêm cột 1 vào final_landmarks để tính affine
                ones = np.ones(shape=(len(final_landmarks), 1))
                landmarks_augmented = np.hstack([final_landmarks, ones])
                final_landmarks = np.dot(landmarks_augmented, rotation_matrix.T)               
                # Update the image
                resized_image = np.array(resized_image_pil)

            # Nếu không có augment, chuyển image thành tensor
            if self.transform:
                resized_image = self.transform(resized_image)
            else:
                resized_image = torch.tensor(resized_image, dtype=torch.float32).unsqueeze(0) / 255.0

            landmarks_tensor = torch.tensor(final_landmarks.flatten(), dtype=torch.float32)

            return resized_image, landmarks_tensor
