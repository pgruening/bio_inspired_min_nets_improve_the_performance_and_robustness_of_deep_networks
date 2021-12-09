import argparse
from os.path import join

import numpy as np
import torch
from tqdm import tqdm

from datasets.ds_cifar10_compression_test import Cifar10JpegCompression
from DLBio import pt_training
from DLBio.helpers import check_mkdir, search_rgx
from DLBio.pytorch_helpers import get_device
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
        which compression rate 1 = 100%, 2 = 90%, 3 = 80%, ..., 10 = 10%

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
        which compression rate 1 = 100%, 2 = 90%, 3 = 80%, ..., 10 = 10%

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


def assert_correct_confusion_matrix(X, idx):
    """ 
    This functions test if the error delta and the POCP computations are sound.

    The POCP can be much higher compared to the error increase between the
    original predictions and the new predictions.

    This is because the POCP observes any change in classification instead
    of only observing which previously correct samples are now incorrect.

    Furthermore, a few previously incorrect examples can become correct.

    Parameters
    ----------
    X : np.array of shape (10000, 11), first column is label
    idx : int
        which compression rate 1 = 100%, 2 = 90%, 3 = 80%, ..., 10 = 10%

    """
    def _and(x, y):
        return np.logical_and(x, y)

    def _not(x):
        return np.logical_not(x)

    def almost_equal(x, y, eps=1e-6):
        return np.abs(x - y) < eps

    def compute_conf_mat(left, top, N):
        conf_matrix = np.zeros((2, 2))

        conf_matrix[0, 0] = _and(left, top).sum()
        conf_matrix[0, 1] = _and(left, _not(top)).sum()
        conf_matrix[1, 0] = _and(_not(left), top).sum()
        conf_matrix[1, 1] = _and(_not(left), _not(top)).sum()

        return conf_matrix / N

    num_images = X.shape[0]
    error = compute_error_for_subset(X.copy(), idx)
    accuracy = 1. - error / 100

    pocp = compute_change_prob(X.copy(), idx)
    perc_stays_same = 1. - pocp / 100

    label = X[:, 0]

    # remove label
    X = X[:, 1:]
    was_correct = X[:, 0] == label
    original_acc = (was_correct).mean()
    original_error = 100. * (1. - original_acc)

    stays = X[:, 0] == X[:, idx]
    now_correct = X[:, idx] == label

    # create table 1:
    #     now_correct y | n
    # stays: y | a | c
    # stays: n | b | d
    conf_mat_1 = compute_conf_mat(stays, now_correct, num_images)
    assert almost_equal(conf_mat_1[:, 0].sum(), accuracy)
    assert almost_equal(conf_mat_1[0, :].sum(), perc_stays_same)

    # create table 2:
    #     was_correct y | n
    # now_correct: y | a | c
    # now_correct: n | b | d
    conf_mat_2 = compute_conf_mat(now_correct, was_correct, num_images)
    assert almost_equal(conf_mat_2[:, 0].sum(), original_acc)
    assert almost_equal(conf_mat_2[0, :].sum(), accuracy)

    # orig - (now_false/was_right) + (now_right/was_false)
    delta = - conf_mat_2[1, 0] + conf_mat_2[0, 1]
    tmp = original_acc + delta
    assert almost_equal(tmp, accuracy)


if __name__ == '__main__':
    main(get_options())
