"""
Code copied from:
https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/trainer.py
"""
import torchvision.datasets as datasets
from config import DATA_FOLDER
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.helpers import HiddenPrints
import warnings
# this is the imagenet normalization:
# https://github.com/pytorch/examples/blob/97304e232807082c2e7b54c597615dc0ad8f6173/imagenet/main.py#L197-L198
# cifar-10 normalization: [0.4914, 0.4822, 0.4465]; [0.2470, 0.2435, 0.2616]
NORMALIZE = transforms.Normalize(
    # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
)
warnings.warn('USING CIFAR-10 NORMALIZATION')


def get_data_loader(*, is_train, batch_size, num_workers=0, do_normalize=True, pin_memory=True):
    with HiddenPrints():
        dataset = get_dataset(is_train, do_normalize)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return data_loader


def get_dataset(is_train, do_normalize=True):
    """
    Provide the Cifar-10 dataset with data augmentation from
    He et al.'s Deep Residual learning:

    "We follow the simple data augmentation in [24] for training: 4 pixels
    are padded on each side,and a 32×32 crop is randomly sampled from
    the padded image or its horizontal flip. For testing, we only
    evaluate the single view of the original 32×32 image."

    The data are downloaded to DATA_FOLDER.

    is_train: bool
        return the Training dataset or validation dataset with the right
        data augmentation
    do_normalize: bool
        add imagenet normalization to pre-processing chain


    returns Pytorch Dataset


    """
    if is_train:
        transform = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),  # second number is padding
            transforms.ToTensor(),
        ]
        if do_normalize:
            transform.append(NORMALIZE)

        transform = transforms.Compose(transform)
        return datasets.CIFAR10(
            root=DATA_FOLDER, train=True, transform=transform, download=True
        )

    else:
        transform = [
            transforms.ToTensor(),
        ]
        if do_normalize:
            transform.append(NORMALIZE)

        transform = transforms.Compose(transform)

        return datasets.CIFAR10(
            root=DATA_FOLDER, train=False, transform=transform, download=True
        )


def _get_train_mean():
    import torch

    dataset = datasets.CIFAR10(
        root=DATA_FOLDER,
        train=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        ),
        download=True,
    )

    all_cifar10_train = torch.stack([x[0] for x in dataset], 0)

    print(all_cifar10_train.mean([0, 2, 3]), all_cifar10_train.std([0, 2, 3]))


if __name__ == "__main__":
    _get_train_mean()
