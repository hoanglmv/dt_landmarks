import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchvision.models import resnet18

class FacialLandmarksModel(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super(FacialLandmarksModel, self).__init__()
        self.learning_rate = learning_rate
        
        self.backbone = resnet18(pretrained=True)
        
        
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.backbone.fc = nn.Linear(512, 136)
        
        self.criterion = nn.MSELoss()
        # Lưu lại hàm Loss
        self.loss_value = []

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        images, landmarks = batch
        predictions = self(images)
        loss = self.criterion(predictions, landmarks)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, landmarks = batch
        predictions = self(images)
        val_loss = self.criterion(predictions, landmarks)
        self.log('val_loss', val_loss)
        self.loss_value.append(val_loss)
        return val_loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
