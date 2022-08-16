#%%
import os
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from matplotlib import pyplot as plt
import pytorch_lightning as pl
import torch
from my_model import SupervisedClassifierSystem, UnSupervisedClassifierSystem, SemiSupervisedSystem , ModelFullyConvolutional

#%%
dataset = MNIST(os.getcwd(), download=True, train=True, transform=transforms.ToTensor())
tr_ds, vl_ds = torch.utils.data.random_split(dataset, (50000, 10000))
tr_sup_ds, tr_unsup_ds = torch.utils.data.random_split(tr_ds, (100, 50000-100))

train_sup_loader = DataLoader(tr_sup_ds, batch_size=5) #, shuffle=True)
train_unsup_loader = DataLoader(tr_unsup_ds, batch_size=2450) #, shuffle=True)
vali_loader = DataLoader(vl_ds, batch_size=64)

#%%
# model = ModelFullyConvolutional()
# supervisedClassifierSystem= SupervisedClassifierSystem(model,lr=1e-5)
# trainer = pl.Trainer(callbacks=[EarlyStopping(monitor="val_loss", mode="min")])
# trainer.fit(model=supervisedClassifierSystem, train_dataloaders=train_sup_loader, val_dataloaders= vali_loader)
# trainer.save_checkpoint("best_supervised_model.ckpt")

# %%

model_semiSup = ModelFullyConvolutional()
semiSupervisedClassifierSystem= SemiSupervisedSystem(model_semiSup,lr=1e-5)

trainer_semisup = pl.Trainer(callbacks=[EarlyStopping(monitor="val_loss", patience=10, mode="min")])
# trainer_semisup = pl.Trainer(\
#            limit_train_batches=10, limit_val_batches=10 ) #, max_epochs=5)
# trainer = pl.Trainer(callbacks=[EarlyStopping(monitor="val_loss", mode="min")], \
#            limit_train_batches=10, limit_val_batches=10 ) #, max_epochs=5)
#trainer = pl.Trainer(max_epochs=10)
trainer_semisup.fit(model=semiSupervisedClassifierSystem, \
        train_dataloaders={"sup": train_sup_loader, "unsup":train_unsup_loader}, \
             val_dataloaders= vali_loader)
trainer_semisup.save_checkpoint("best_semiSupervised_model.ckpt")
# %%
# continue
model_semiSup = ModelFullyConvolutional()
trainer_semisup = pl.Trainer()
semiSupervisedClassifierSystem = SemiSupervisedSystem.load_from_checkpoint(checkpoint_path="best_semiSupervised_model_2300epoch.ckpt",model = model_semiSup, lr=1e-5 )
trainer_semisup.fit(model=semiSupervisedClassifierSystem, \
        train_dataloaders={"sup": train_sup_loader, "unsup":train_unsup_loader}, \
             val_dataloaders= vali_loader)
trainer_semisup.save_checkpoint("best_semiSupervised_model.ckpt")

# %%

# # model2 = ModelFullyConvolutional()
# unSupervisedClassifierSystem= UnSupervisedClassifierSystem(model,lr=1e-6)
# # train_loader = DataLoader(tr_ds, batch_size=64) #, shuffle=True)
# # vali_loader = DataLoader(vl_ds, batch_size=64)

# trainer2 = pl.Trainer(callbacks=[EarlyStopping(monitor="val_loss", mode="min")], \
#            limit_train_batches=80, limit_val_batches=10 ) #, max_epochs=100)
# trainer2.fit(model=unSupervisedClassifierSystem, train_dataloaders=train_loader, val_dataloaders= vali_loader)



#%%
#===============================================
itr = iter(train_loader)
#%%
a = itr.next()
plt.imshow(a[0][0, 0, :, :].numpy())
#plt.show()

pred = model(a[0])

print("~~~~~~~~~~~~")
print(a[1])
am = torch.argmax(pred, dim=1)[:,0,0]
label_pred = [(a[1][i].numpy(), am[i].numpy()) for i in range(32) ]
print("-----------")
print(label_pred)
print ("==============")
#print(pred)
print(pred.shape, torch.argmax(pred))


# %%
itr = iter(vali_loader)
#%%
a = itr.next()
plt.imshow(a[0][0, 0, :, :].numpy())
#plt.show()

pred = model(a[0])

print("~~~~~~~~~~~~")
print(a[1])
am = torch.argmax(pred, dim=1)[:,0,0]
label_pred = [(a[1][i].numpy(), am[i].numpy()) for i in range(32) ]
print("-----------")
print(label_pred)
print ("==============")
#print(pred)
print(pred.shape, torch.argmax(pred))


# %%
