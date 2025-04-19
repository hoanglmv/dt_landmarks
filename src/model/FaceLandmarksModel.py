import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchvision.models import resnet18

class FacialLandmarksModel(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.learning_rate = learning_rate
        
        # Backbone
        self.backbone = resnet18(pretrained=True)
        self.backbone.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        self.backbone.fc = nn.Linear(512, 136)
        
        self.criterion = nn.MSELoss()
        
        # Dùng để lưu loss từng batch trong epoch
        self.train_losses = []
        self.val_losses = []

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        images, landmarks = batch
        preds = self(images)
        loss = self.criterion(preds, landmarks)
        
        # lưu loss để tính trung bình ở on_train_epoch_end
        self.train_losses.append(loss.detach())
        
        # vẫn log nếu cần dùng TensorBoard hoặc CSV logger
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, landmarks = batch
        preds = self(images)
        val_loss = self.criterion(preds, landmarks)
        
        # lưu loss để tính trung bình ở on_validation_epoch_end
        self.val_losses.append(val_loss.detach())
        self.log('val_loss', val_loss, on_step=False, on_epoch=True)

    def on_train_epoch_end(self):
        # tính và in ra train loss trung bình sau mỗi epoch
        avg_train_loss = torch.stack(self.train_losses).mean()
        print(f"Epoch {self.current_epoch} ► Train loss: {avg_train_loss:.4f}")
        # reset danh sách
        self.train_losses.clear()

    def on_validation_epoch_end(self):
        # tương tự cho validation loss
        avg_val_loss = torch.stack(self.val_losses).mean()
        print(f"Epoch {self.current_epoch} ► Val   loss: {avg_val_loss:.4f}")
        self.val_losses.clear()

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)
