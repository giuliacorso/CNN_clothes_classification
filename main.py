import numpy as np
import torch
import time

from matplotlib import pyplot as plt

from argument_parser import get_conf
from cloth_dataset import ClothDataset
from torch.utils.data import DataLoader
from model import ClothModel

from train import train_function
import test


if __name__ == '__main__':
    args = get_conf()

    if args.phase == 'train':
        train_function(args)
    else:
        dataset_test = ClothDataset(args, phase='test')
        dataloader_test = DataLoader(dataset_test, batch_size=8, shuffle=True, num_workers=0)
        if args.confusion_matrix:
            test.create_confusion_matrix(args, dataloader_test)
        if args.real_image != '':
            test.test_real_image(args)
