import cv2
from src.dataset.FaceLandmarksDataset import FaceLandmarksDataset
dataset = FaceLandmarksDataset()
image, landmarks = dataset[0]  
print(image.shape)
print(landmarks)

