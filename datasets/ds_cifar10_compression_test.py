from os.path import join

import cv2
import numpy as np
import torch
import torchvision.datasets as datasets
from config import DATA_FOLDER
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from datasets.ds_cifar10 import NORMALIZE

COMP_VALUES = [10 * x for x in range(1, 10)]


# Create a class that inherits from Dataset
class Cifar10JpegCompression(Dataset):
    def __init__(self):
        self.images = np.load(
            join(DATA_FOLDER, 'test_cifar10_jpeg_noise_image_stacks.npy')
        )
        self.labels = np.load(
            join(DATA_FOLDER, 'test_cifar10_jpeg_noise_labels.npy')
        )

        # use the same image transformations as in ds_cifar10
        self.f = transforms.Compose([
            transforms.ToTensor(),
            NORMALIZE,
        ])

    def __getitem__(self, idx):
        # return image and label at idx
        # shape: (10, 32, 32, 3) -> compression, height, width, channels
        # load the image with the proper compression rate
        # e.g., if compression rate == 0 -> x = X[0, ...]
        # e.g., if compression rate == 10 -> x = X[1, ...]
        # e.g., if compression rate == 100 -> x = X[-1, ...]
        X = self.images[idx, ...]
        label = self.labels[idx]

        image_stack = torch.stack(
            [self.f(X[i, ...]) for i in range(X.shape[0])], 0
        )

        return image_stack, torch.Tensor([label]).long()

    def __len__(self):
        # return the number of images
        return self.images.shape[0]


def create_dataset(is_train=False):
    print('Creating compression dataset ...')
    # load the original dataset from torchvision
    dataset = datasets.CIFAR10(
        root=DATA_FOLDER, train=is_train, transform=None)
    out = []
    labels_ = []
    # create new compressed images
    for image, label in tqdm(dataset):
        labels_.append(label)

        # from pillow image to np.array
        # uint8 image with 32x32x3
        image = np.array(image)

        # create a stack of all compressed images
        tmp = [image]  # original uncompressed image
        for compression in COMP_VALUES:
            # add compressed image
            tmp.append(create_compressed_image(image, compression))

        #  to output: add stacked (10, 32, 32, 3) tensor
        out.append(np.stack(tmp, 0))

    # to (10k, 10, 32, 32, 3) uint8
    out = np.stack(out, 0)
    prefix = 'train_' if is_train else 'test_'
    np.save(join(DATA_FOLDER, prefix + 'cifar10_jpeg_noise_image_stacks.npy'), out)

    labels_ = np.array(labels_)
    np.save(join(DATA_FOLDER, prefix + 'cifar10_jpeg_noise_labels.npy'), labels_)
    print('... done.')


def create_compressed_image(image, compression):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100 - compression]
    result, encimg = cv2.imencode('.jpg', image, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg


if __name__ == '__main__':
    create_dataset()
