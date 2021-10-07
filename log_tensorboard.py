import glob
import json
import os
import traceback
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

from DLBio.helpers import load_json, check_mkdir


def log_tensorboard(workdir, tb_out, dataloaders=None, numBatches=1, model=None, show_labels=True, remove_old_events=True, input_shape=None):
    """ write data for visualization by tensorboard

    Start TensorBoard with 
            %tensorboard --logdir tb_out

    Parameters
    ----------
    workdir : str
            the working directory which should contain a 'log.json' file
            where the training values are stored
    tb_out : str
            out_path to store the tensorboard data in
    dataloaders : dict
            the dataloaders used in training for visualization of some image_batches
    numBatches : int, optional
            number of image batches to display, by default 1
    model : pytorch model, optional
            the model used in training, will be displayed as graph in TensorBoard, by default None
    show_labels : bool, optional
            show the corresponding labels in the grid, by default True
    """
    if remove_old_events:
        files_ = glob.glob(join(tb_out, 'events.out.tfevents.*'))
        for file in files_:
            print(f'remove file: {file} (Y for yes, any other key: no) ?')
            user_input = input()
            if user_input == 'Y':
                os.remove(file)

    log = load_json(join(workdir, 'log.json'))
    check_mkdir(tb_out)

    try:
        tb_writer = SummaryWriter(tb_out)

        if log is not None:
            for key, values in log.items():
                [tb_writer.add_scalar(key, v, ep)
                 for ep, v in enumerate(values)]

        if dataloaders is not None:
            for key, dl in dataloaders.items():
                if dl is not None:
                    di = iter(dl)
                    for i in range(numBatches):
                        images, labels = di.next()

                        if show_labels:
                            grid = label_images(images, labels)
                            tb_writer.add_figure(f"{key}_batch{i}", grid)
                        else:
                            grid = torchvision.utils.make_grid(images)
                            tb_writer.add_image(
                                f"{key}_batch{i}", to_uint(grid))
        else:
            images = torch.zeros(*input_shape)

        # load model if needed
        if model is not None:
            tb_writer.add_graph(model.cpu(), images.cpu())

        print(f'finished writing to {tb_out}')
    except:
        print("Exception thrown while logging to SummaryWriter:")
        traceback.print_exc()
    finally:
        tb_writer.flush()
        tb_writer.close()


def to_uint(tensor_image):
    tensor_image -= tensor_image.min()
    tensor_image /= tensor_image.max()
    tensor_image *= 255
    tensor_image = np.array(tensor_image.detach().cpu()).astype('uint8')

    return tensor_image


def label_images(images, labels):
    # add max 8 images to one row
    num_images = len(images)
    num_rows = num_images / 8
    if num_images % 8 > 0:
        num_rows += 1

    figure = plt.figure(figsize=(10, 10))
    for i in range(len(images)):
        plt.subplot(num_rows, 8, i + 1, title=labels[i].item())
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        img = images[i]
        img = to_uint(img)
        img = np.transpose(img, (1, 2, 0))
        plt.imshow(img)

    return figure
