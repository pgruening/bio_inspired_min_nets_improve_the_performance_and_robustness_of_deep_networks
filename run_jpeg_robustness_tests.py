import argparse
from os.path import join

import numpy as np
import torch
from DLBio import pt_training
from DLBio.helpers import check_mkdir, search_rgx
from DLBio.pytorch_helpers import get_device
from tqdm import tqdm

from datasets.ds_cifar10_compression_test import Cifar10JpegCompression
from helpers import load_model

DATA_FOLDER_NAME = 'jpeg_model_predictions'


def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_folder', type=str, default=None)
    parser.add_argument('--base_folder', type=str, default=None)
    parser.add_argument('--device', type=int, default=None)

    return parser.parse_args()


def main(options):
    if options.device is not None:
        pt_training.set_device(options.device)

    pred_save_path = join(options.base_folder, DATA_FOLDER_NAME)
    compute_and_save_results(options.model_folder, pred_save_path)


def compute_and_save_results(folder, pred_save_path):
    """Loads a model from '[folder]/model.pt'. The model classifies ten
    different compression rates {0,...,90} of each image of the Cifar-10
    testset. 
    All predictions and the actual class are save as a numpy file. The file
    contains a matrix of shape (10000, 11). Each row corresponds to an image,
    the first column is the actual label, all subsequent columns are the model
    predictions at compression 0, 10, 20, ..., 90.

    Parameters
    ----------
    folder : string
        folder containing the model with filename model.pt
    pred_save_path : string
        folder to where the numpy file is saved
    """
    device = get_device()
    model = load_model(
        join(folder, 'opt.json'), device,
        new_model_path=join(folder, 'model.pt')
    )
    dataset = Cifar10JpegCompression()

    out = []
    for image, label in tqdm(dataset):
        with torch.no_grad():
            prediction = model(image.to(device))
        p_classes = prediction.max(1)[1]

        # save as (label, predictions)
        label = label.numpy()
        p_classes = p_classes.cpu().numpy()

        out.append(np.concatenate([label, p_classes]))

    out = np.stack(out, 0)

    name = folder.split('/')[-1]
    out_path = join(pred_save_path, name + '.npy')
    check_mkdir(out_path)
    np.save(out_path, out)


def compute_error_for_subset(X, idx):
    """Given the Prediction matrix (see compute_and_save_results), compute the
    test error for a given compression rate, defined by index. 

    Parameters
    ----------
    X : np.array of shape (10000, 11), first column is label
    idx : int
        which compression rate 0 = 0%, 1 = 10%, 2 = 20%, ..., 9 = 90%

    Returns
    -------
    float
        Error in percent (in [0, 100]).
    """
    # grab the label
    y = X[:, 0]
    # only consider the predictions
    X = X[:, 1:]
    acc = (y == X[:, idx]).mean()
    return 100. * (1. - acc)


def compute_change_prob(X, idx):
    """Given the Prediction matrix (see compute_and_save_results), compute the
    percentage of predictions that are different from prediction for the
    uncompressed image for a given compression rate, defined by index.

    Parameters
    ----------
    X : np.array of shape (10000, 11), first column is label
    idx : int
        which compression rate 0 = 0%, 1 = 10%, 2 = 20%, ..., 9 = 90%

    Returns
    -------
    float
        Percentage of predictions that differ from the prediction at
        compression rate 0% (in [0, 100]).
    """
    # only consider the predictions
    X = X[:, 1:]
    # compare uncompressed prediction to subset of the compressed predictions
    did_change = X[:, 0] != X[:, idx]
    return did_change.mean() * 100.


if __name__ == '__main__':
    main(get_options())
