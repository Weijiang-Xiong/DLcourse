"""Trainer.

@author: Zhenye Na - https://github.com/Zhenye-Na
@reference: "End to End Learning for Self-Driving Cars", arXiv:1604.07316
"""

import os
import torch
import timeit

from utils import toDevice


class Trainer(object):
    """Trainer class."""

    def __init__(self,
                 dataroot,
                 ckptroot,
                 model,
                 device,
                 epochs,
                 criterion,
                 optimizer,
                 scheduler,
                 start_epoch,
                 trainloader,
                 validationloader):
        """Self-Driving car Trainer.

        Args:
            model: CNN model
            device: cuda or cpu
            epochs: epochs to training neural network
            criterion: nn.MSELoss()
            optimizer: optim.Adam()
            start_epoch: 0 or checkpoint['epoch']
            trainloader: training set loader
            validationloader: validation set loader

        """
        super(Trainer, self).__init__()

        self.dataType = dataroot.split('/')[-2]
        self.model = model
        self.device = device
        self.epochs = epochs
        self.ckptroot = ckptroot
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.start_epoch = start_epoch
        self.trainloader = trainloader
        self.validationloader = validationloader
        self.trainLoss = []
        self.validationLoss = []

    def train(self):
        """Training process."""
        self.model.to(self.device)
        for epoch in range(self.start_epoch, self.epochs + self.start_epoch):
            self.scheduler.step()

            startTime = timeit.default_timer()

            # Training
            train_loss = 0.0
            self.model.train()

            for local_batch, (centers, lefts, rights) in enumerate(self.trainloader):
                # Transfer to GPU
                centers, lefts, rights = toDevice(centers, self.device), toDevice(
                    lefts, self.device), toDevice(rights, self.device)

                # Model computations
                self.optimizer.zero_grad()
                datas = [centers, lefts, rights]
                for data in datas:
                    imgs, angles = data
                    # print("training image: ", imgs.shape)
                    outputs = self.model(imgs)
                    loss = self.criterion(outputs, angles.unsqueeze(1))
                    loss.backward()
                    self.optimizer.step()

                    train_loss += loss.data.item()

                if local_batch % 100 == 0:
                    # calculate loss at the beginning, if the dataset is large, then report loss every 100 batches
                    curTrainLoss = train_loss / (local_batch + 1)
                    print("Training Epoch: {} | Loss: {}".format(epoch, curTrainLoss))
                    self.trainLoss.append(curTrainLoss)

            # Validation
            self.model.eval()
            valid_loss = 0
            with torch.set_grad_enabled(False):
                for local_batch, (centers, lefts, rights) in enumerate(self.validationloader):
                    # Transfer to GPU
                    centers, lefts, rights = toDevice(centers, self.device), toDevice(
                        lefts, self.device), toDevice(rights, self.device)

                    # Model computations
                    self.optimizer.zero_grad()
                    datas = [centers, lefts, rights]
                    for data in datas:
                        imgs, angles = data
                        outputs = self.model(imgs)
                        loss = self.criterion(outputs, angles.unsqueeze(1))

                        valid_loss += loss.data.item()

                    if local_batch % 100 == 0:
                        curValLoss = valid_loss / (local_batch + 1)
                        print("Validation Loss: {}".format(curValLoss))
                        self.validationLoss.append(curValLoss)

            stopTime = timeit.default_timer()
            print('Time for single round (s):', stopTime-startTime)

            # Save model
            if epoch % 5 == 0 or epoch == self.epochs + self.start_epoch - 1:

                print("==> Save checkpoint ...")

                state = {
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                }

                self.save_checkpoint(state)
                if epoch == self.epochs + self.start_epoch - 1:
                    modelPath = self.ckptroot + self.dataType + '/'
                    with open(modelPath + 'trainLoss.txt', 'w') as f:
                        for item in self.trainLoss:
                            f.write("%s\n" % item)
                    with open(modelPath + 'valLoss.txt', 'w') as f:
                        for item in self.validationLoss:
                            f.write("%s\n" % item)


    def save_checkpoint(self, state):
        """Save checkpoint."""
        modelPath = self.ckptroot + self.dataType + '/'
        if not os.path.exists(modelPath):
            os.makedirs(modelPath)

        torch.save(state, modelPath + 'model-{}.h5'.format(state['epoch']))
