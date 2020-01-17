
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

from model import ModNet, NetworkNvidia
from trainer import Trainer
from utils import data_loader, load_data


def parse_args(dataPath):
    """Parse parameters.
    you can use python main.py --help in command line console to view all these information"""
    parser = argparse.ArgumentParser(description='Main pipeline for self-driving vehicles simulation using machine learning.')

    # directory
    parser.add_argument('--dataroot',     type=str,   default=dataPath,                         help='path to dataset')
    parser.add_argument('--ckptroot',     type=str,   default="./myProject/tempModels/",        help='path to checkpoint')

    # hyperparameters settings
    # be careful with num_workers=8, try set it to smaller numbers
    # (even 0) if memory runs up
    parser.add_argument('--lr',           type=float, default=1e-4,             help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,             help='weight decay (L2 penalty)')
    parser.add_argument('--batch_size',   type=int,   default=32,               help='training batch size')
    parser.add_argument('--num_workers',  type=int,   default=0,                help='the number of workers used in dataloader')
    parser.add_argument('--train_size',   type=float, default=0.8,              help='train validation set split ratio')
    parser.add_argument('--shuffle',      type=bool,  default=True,             help='whether shuffle data during training')

    # training settings
    parser.add_argument('--epochs',       type=int,   default=15,               help='number of epochs to train')
    parser.add_argument('--start_epoch',  type=int,   default=0,                help='pre-trained epochs')
    parser.add_argument('--resume',       type=bool,  default=False,            help='whether re-training from a interrupted model')
    parser.add_argument('--model_name',   type=str,   default="modified",       help='model architecture to use [default, modified]')

    # parse the arguments
    args = parser.parse_args()

    return args


def main():
    """Main pipeline."""
    # parse command line arguments
    pathSet = [
        "./myProject/trainData/Mild/",
        "./myProject/trainData/Aggressive/"
    ]
    for dataPath in pathSet:
        # parse argument
        args = parse_args(dataPath)

        print("Constructing data loaders...")
        # load training set and split
        trainset, valset = load_data(args.dataroot, args.train_size)

        trainloader, validationloader = data_loader(args.dataroot,trainset, valset,
        args.batch_size,args.shuffle,args.num_workers)

        # define model
        print("Initialize model ...")
        if args.model_name == "default":
            model = NetworkNvidia()
        elif args.model_name == "modified":
            model = ModNet()

        # define optimizer and loss function
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.MSELoss()

        # learning rate scheduler
        scheduler = MultiStepLR(optimizer, milestones=[30, 50], gamma=0.1)

        # resume
        if args.resume:
            print("Loading a model from checkpoint")
            # use pre-trained model
            checkpoint = torch.load("model.h5", map_location=lambda storage, loc: storage)

            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])

        # cuda or cpu
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Selected GPU: ", device)

        # training
        print("Training Neural Network...")
        trainer = Trainer(args.dataroot,args.ckptroot,model,device,args.epochs,criterion,optimizer,
        scheduler,args.start_epoch,trainloader,validationloader)
        trainer.train()


if __name__ == '__main__':
    main()
