import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split, DataLoader, Dataset
from torchvision import transforms
from torchvision import datasets

import pytorch_lightning as pl


class ReshapeTransform:
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        return torch.reshape(img, self.new_size)


class MNIST(pl.LightningDataModule):
    def __init__(self, batch_size, test_batch_size, root='./datasets',
                 task_list=list(range(10)), conv=False):
        super().__init__()
        # self.args = args
        self.root = root
        self.train_fraction = 1.
        if conv:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
            ])

        else:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize((0.1307,), (0.3081,)),
                ReshapeTransform((-1,))
            ])
        self.dims = (1, 28, 28)
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.task_list = task_list

    def prepare_data(self):
        datasets.MNIST(self.root, train=True, download=True)
        datasets.MNIST(self.root, train=False, download=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            dset = DataLoader(datasets.MNIST(self.root, train=True,
                                             transform=self.transforms), 100000)
            x_train, y_train = next(dset.__iter__())
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                              train_size=int(50000*self.train_fraction),
                                                              test_size=10000)

            mnist_train = ContDataset(x_train, y_train, transforms=None)
            mnist_train.set_task(self.task_list)
            self.train = mnist_train

            mnist_val = ContDataset(x_val, y_val, transforms=None)
            mnist_val.set_task(self.task_list)
            self.val = mnist_val
        if stage == 'test' or stage is None:
            dset = DataLoader(datasets.MNIST(self.root, train=False,
                                             transform=self.transforms), 100000)
            x_test, y_test = next(dset.__iter__())
            self.test = ContDataset(x_test, y_test, transforms=None)

    def train_dataloader(self):
        return DataLoader(self.train, self.batch_size, num_workers=6)

    def val_dataloader(self):
        return DataLoader(self.val, self.batch_size, num_workers=6)

    def test_dataloader(self):
        return DataLoader(self.test, self.test_batch_size, num_workers=6)


class FashionMNIST(pl.LightningDataModule):
    def __init__(self, batch_size, test_batch_size, root='./datasets', task_list=list(range(10)), conv=False):
        super().__init__()
        # self.args = args
        self.root = root
        self.train_fraction = 1.
        if conv:
            self.transforms = transforms.Compose([
                transforms.ToTensor()
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize((0.1307,), (0.3081,)),
                ReshapeTransform((-1,))
            ])
        self.dims = (1, 28, 28)
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.task_list = task_list

    def prepare_data(self):
        datasets.FashionMNIST(self.root, train=True, download=True)
        datasets.FashionMNIST(self.root, train=False, download=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            dset = DataLoader(datasets.FashionMNIST(self.root, train=True,
                                             transform=self.transforms), 100000)
            x_train, y_train = next(dset.__iter__())

            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                              train_size=int(50000*self.train_fraction),
                                                              test_size=10000)

            mnist_train = ContDataset(x_train, y_train, transforms=None)
            mnist_train.set_task(self.task_list)
            self.train = mnist_train

            mnist_val = ContDataset(x_val, y_val, transforms=None)
            mnist_val.set_task(self.task_list)
            self.val = mnist_val
        if stage == 'test' or stage is None:
            dset = DataLoader(datasets.FashionMNIST(self.root, train=False,
                                             transform=self.transforms), 100000)
            x_test, y_test = next(dset.__iter__())
            self.test = ContDataset(x_test, y_test, transforms=None)

    def train_dataloader(self):
        return DataLoader(self.train, self.batch_size, num_workers=6)

    def val_dataloader(self):
        return DataLoader(self.val, self.batch_size, num_workers=6)

    def test_dataloader(self):
        return DataLoader(self.test, self.test_batch_size, num_workers=6)


def load_dataset(args, **kwargs):
    conv = True
    if args.arc_type == 'mlp':
        conv = False
    if args.dataset_name == 'mnist':
        args.image_size = (1, 28, 28)
        data_module  = MNIST(args.batch_size, args.test_batch_size, task_list=args.task_list, conv=conv)
    elif args.dataset_name == 'fashion_mnist':
        args.image_size = (1, 28, 28)
        data_module = FashionMNIST(args.batch_size, args.test_batch_size, task_list=args.task_list, conv=conv)
    return args, data_module



