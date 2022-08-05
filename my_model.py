######
###### TODO: add normalizing layers , normalize the input
######

# import os
import torch
from torch import nn

import torch.nn.functional as F

# from torchvision import transforms
# from torchvision.datasets import MNIST
# from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from torchmetrics.functional import accuracy

IMAGE_SIZE = (28, 28)
# LR = 1e-5

class ModelFullyConvolutional(nn.Module):
    def __init__(self):
        super().__init__()
        # self.network = nn.Sequential()
        self.conv1 = nn.Conv2d(1, 3, 3)
        self.maxp1 = nn.MaxPool2d(2, stride=2)  # TODO what is the default stride value?
        self.conv2 = nn.Conv2d(3, 6, 3)
        self.maxp2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(6, 10, 5)
        #self.dropout1 = nn.Dropout(0.5)  # nn.Dropout2d(0.5)
        #self.dropout2 = nn.Dropout(0.5)  # nn.Dropout2d(0.5)
    def forward(self, x):
        a = self.conv1(x)
        a = self.maxp1(a)
        a = nn.ReLU()(a)
        #a = self.dropout1(a)
        a = self.conv2(a)
        a = self.maxp2(a)
        a = nn.ReLU()(a)
        #a = self.dropout2(a)
        a = self.conv3(a)
        return a

class SupervisedClassifierSystem(pl.LightningModule):
    def __init__(self, model:nn.Module, lr):
        super().__init__()
        self.model=model
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        
        ##b = F.softmax(y_hat, dim=1) * F.log_softmax(y_hat, dim=1)
        # b = F.softmax(y_hat, dim=1) 
        # penalty = (b.mean(dim=0)**2).mean()
        # b=b* F.log_softmax(y_hat, dim=1)
        # b = -1.0 * b.mean()+ 10* penalty
                
        b = nn.CrossEntropyLoss()(y_hat.squeeze(), y)
        self.log("train_loss", b)
        return b

    def validation_step(self,batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"val_acc": acc, "val_loss":loss}
        self.log_dict(metrics)
        return metrics

    def test_step(self,batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"test_acc": acc, "test_loss":loss}
        self.log_dict(metrics)
        return metrics

    def predict_step(self,batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self.model(x)
        return y_hat

    def _shared_eval_step(self, batch, batch_id):
        x, y = batch
        y_hat = self.model(x)
        loss = nn.CrossEntropyLoss()(y_hat.squeeze(), y)
        acc = accuracy(y_hat, y)

        
        return loss, acc

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer


class UnSupervisedClassifierSystem(pl.LightningModule):
    def __init__(self, model:nn.Module, lr):
        super().__init__()
        self.model=model
        self.lr = lr
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        
        b = F.softmax(y_hat, dim=1) * F.log_softmax(y_hat, dim=1)
        b = -1.0 * b.mean()
        # b = F.softmax(y_hat, dim=1) 
        # penalty = (b.mean(dim=0)**2).mean()
        # b=b* F.log_softmax(y_hat, dim=1)
        # b = -1.0 * b.mean()+ 10* penalty
                
        #b = nn.CrossEntropyLoss()(y_hat.squeeze(), y)
        self.log("train_loss", b)
        return b

    def validation_step(self,batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"val_acc": acc, "val_loss":loss}
        self.log_dict(metrics)
        return metrics

    def test_step(self,batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"test_acc": acc, "test_loss":loss}
        self.log_dict(metrics)
        return metrics

    def predict_step(self,batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self.model(x)
        return y_hat

    def _shared_eval_step(self, batch, batch_id):
        x, y = batch
        y_hat = self.model(x)
        loss = F.softmax(y_hat, dim=1) * F.log_softmax(y_hat, dim=1) # nn.CrossEntropyLoss()(y_hat.squeeze(), y)
        loss = -1.0 * loss.mean()
        acc = accuracy(y_hat, y)

        
        return loss, acc

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer

