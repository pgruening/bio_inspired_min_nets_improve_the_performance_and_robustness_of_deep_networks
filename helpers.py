import copy
import os
import sys

import torch
import torch.nn as nn
from DLBio import pt_training
from DLBio.helpers import dict_to_options
from DLBio.kwargs_translator import get_kwargs
from DLBio.pytorch_helpers import (ActivationGetter, get_device,
                                   load_model_with_opt, walk)

from datasets import data_getter
from models.model_getter import get_model


class HiddenPrints:
    # https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def get_data_loaders(options):
    if isinstance(options, dict):
        options = dict_to_options(options)

    # translate input string to dictionary
    if options.ds_kwargs is not None:
        kwargs = get_kwargs(options.ds_kwargs)
    else:
        kwargs = get_kwargs(None)

    data_loaders = data_getter.get_data_loaders(
        options.dataset, batch_size=options.bs,
        num_workers=options.nw,
        split_index=options.split_index,
        **kwargs
    )
    return data_loaders


def load_model(options, device, strict=True, new_model_path=None, map_location=None, from_par_gpu=False):
    if isinstance(options, dict):
        options = dict_to_options(options)

    def get_model_fcn(options, device):
        model_kwargs = get_kwargs(options.model_kw)
        return get_model(
            options.model_type,
            options.in_dim,
            options.out_dim,
            device,
            **model_kwargs
        )

    if new_model_path is not None:
        model_path = new_model_path
    else:
        model_path = options.model_path

    return load_model_with_opt(
        model_path,
        options,
        get_model_fcn,
        device,
        strict=strict,
        map_location=map_location,
        from_par_gpu=from_par_gpu
    )


def get_ordered_module_list(model, *, batch_input_size, device, use_only):
    class ActGetterwithGlobalList(ActivationGetter):
        def __init__(self, module, global_list, global_ctr, use_only):
            super(ActGetterwithGlobalList, self).__init__(module)
            self.list = global_list
            self.num_conv_layers = global_ctr
            self.use_only = tuple(use_only)

        def _hook_fcn(self, module, input_, output):
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                self.num_conv_layers[0] += 1

            elif isinstance(module, self.use_only):
                self.list.append((self.num_conv_layers[0], module))

    modules_ = walk(model)

    global_list = []
    global_ctr = [0]  # use list as a global object
    for m in modules_:
        ActGetterwithGlobalList(m, global_list, global_ctr, use_only)

    x = torch.zeros(*batch_input_size).to(device)
    model(x)

    return global_list


def predict_needed_gpu_memory(options, *, input_shape, device, factor=2.1):
    # changes to the options object here should not be saved to the actual object
    options = copy.deepcopy(options)

    if isinstance(options, dict):
        # initialize a new model
        options['model_path'] = None
        options = dict_to_options(options)

    if device is not None:
        pt_training.set_device(device, verbose=False)

    model = load_model(options, get_device())

    torch.cuda.reset_max_memory_allocated()
    for _ in range(3):
        x = torch.rand(*input_shape).to(get_device())
        model(x)

    fwd_memory_used = torch.cuda.max_memory_allocated()
    torch.cuda.empty_cache()

    # return as MegaByte
    return factor * fwd_memory_used / 1e6
