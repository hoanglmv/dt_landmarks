import sys
sys.path.append('D:/lmvh/Detect_Facial_Landmarks/src')

from pytorch_lightning import Trainer
from torch.utils.data import random_split, DataLoader
from src.dataset.FaceLandmarksDataset import FaceLandmarksDataset
from src.model.FaceLandmarksModel import FacialLandmarksModel  
from torchvision import transforms
from pytorch_lightning.callbacks import ModelCheckpoint

dataset = FaceLandmarksDataset(transform=transforms.ToTensor())
dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)



checkpoint_callback = ModelCheckpoint(
    dirpath='D:/lmvh/Detect_Facial_Landmarks/outputs',
    filename='facial_landmarks_model',
    save_top_k=1,
    monitor='val_loss',
    mode='min'
)

model = FacialLandmarksModel()
trainer = Trainer(max_epochs=20, accelerator='cpu', callbacks=[checkpoint_callback])

trainer.fit(model, train_loader, val_loader)
