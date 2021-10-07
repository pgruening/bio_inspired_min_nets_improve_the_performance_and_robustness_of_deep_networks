import copy
from os.path import join

import sys
sys.path.append(
    '/nfshome/gruening/my_code/DLBio_repos/min_nets_svrhm_2021_code_submission'
)

import torch
from DLBio import kwargs_translator, pt_run_parallel
from DLBio.helpers import check_mkdir, load_json, search_rgx
from helpers import get_data_loaders, load_model, predict_needed_gpu_memory
from log_tensorboard import log_tensorboard

DEFAULT_KWARGS = {
    'comment': 'exp7: mobilenets with FP-Blocks',
    'lr': 0.1,
    'wd': 0.0001,
    'mom': 0.9,
    'bs': 128,
    'opt': 'SGD',
    'nw': 0,

    'train_interface': 'classification',

    # model / ds specific params
    'in_dim': 3,
    'out_dim': 10,

    # scheduling
    'epochs': 200,
    'lr_steps': 0,
    'fixed_steps': [100, 150],

    # dataset
    'dataset': 'cifar_10',

    # model saving: This time, no early stopping is used
    # just the model saved at the end!
    'sv_int': 0,
    # 'early_stopping': None,
    # 'es_metric': 'val_acc',
    'do_overwrite': None
}

EXE_FILE = 'run_training.py'
BASE_FOLDER = 'experiments/exp_10'

# only use the layer-start method
MODELS = [
    # 'CifarMinFP-LS',
    # 'CifarResNet',
    # 'CifarPyrResNet',
    'CifarMinFP-LS-RNBasic'
]

AVAILABLE_GPUS = [0, 1, 2, 3]
SEEDS = [9, 507, 723, 16, 744]

NUM_BLOCKS = [3, 5, 7, 9]
RUN_BIN_PACKING = True


class TrainingProcess(pt_run_parallel.ITrainingProcess):
    def __init__(self, **kwargs):
        self.start_time = -1
        self.device = -1
        self.mem_used = None

        self.__name__ = 'Exp. 10 training process'
        self.module_name = EXE_FILE
        self.kwargs = kwargs


def _run(param_generator):
    make_object = pt_run_parallel.MakeObject(TrainingProcess)
    if RUN_BIN_PACKING:
        pt_run_parallel.run_bin_packing(
            param_generator(), make_object,
            available_gpus=AVAILABLE_GPUS,
            log_file=join(BASE_FOLDER, 'parallel_train_log.txt')
        )
    else:
        pt_run_parallel.run(param_generator(), make_object,
                            available_gpus=AVAILABLE_GPUS
                            )


def run():
    default_kwargs = copy.deepcopy(DEFAULT_KWARGS)
    base_folder = join(BASE_FOLDER, 'exp_data')

    def param_generator():
        for p in _param_generator(default_kwargs, base_folder):
            yield p

    _run(param_generator)


def _param_generator(default_kwargs, base_folder, seeds=SEEDS, models=MODELS, num_blocks=NUM_BLOCKS):
    for seed in seeds:
        for model in models:
            for n in num_blocks:
                output = copy.deepcopy(default_kwargs)
                output['seed'] = seed
                output['folder'] = join(
                    base_folder, f'{model}_s{seed}_N{n}')

                output['model_type'] = model
                model_kw = {'q': [2], 'k': [3], 'N': [n]}
                output['model_kw'] = kwargs_translator.to_kwargs_str(
                    model_kw)

                output['mem_used'] = predict_needed_gpu_memory(
                    output, input_shape=(
                        output['bs'], output['in_dim'], 32, 32),
                    device=AVAILABLE_GPUS[0]
                )

                yield output


def one_epoch_test():
    output = copy.deepcopy(DEFAULT_KWARGS)
    base_folder = join(BASE_FOLDER, 'one_epoch_test')
    output['folder'] = base_folder

    output['epochs'] = 1
    output['comment'] = f"exp_10: one_run_tests"
    output['do_overwrite'] = None
    output['sv_int'] = 0
    output['seed'] = 0

    output.pop('early_stopping')
    output.pop('fixed_steps')

    def param_generator():
        for p in _param_generator(output, base_folder, seeds=[0], num_blocks=[3]):
            yield p

    _run(param_generator)


def one_model_test(model_type):
    output = copy.deepcopy(DEFAULT_KWARGS)
    base_folder = join(BASE_FOLDER, 'one_model_test')
    output['folder'] = base_folder

    output['epochs'] = 1
    output['comment'] = f"exp_10: one_model_test"
    output['do_overwrite'] = None
    output['sv_int'] = 0
    output['seed'] = 0

    output.pop('early_stopping')
    output.pop('fixed_steps')

    def param_generator():
        for p in _param_generator(output, base_folder, seeds=[0], models=[model_type], num_blocks=[3]):
            yield p

    _run(param_generator)


def _check_tensorboard(folder, out_name, *, add_images):
    tb_out = join(BASE_FOLDER, 'tboard', out_name)
    check_mkdir(tb_out)

    options = load_json(join(folder, 'opt.json'))
    assert options is not None, f'no options at: {folder}'

    model = load_model(options, 'cpu', map_location=torch.device(
        'cpu'), new_model_path=join(folder, 'model.pt')
    )

    if add_images:
        data_loaders = get_data_loaders(options)
    else:
        data_loaders = None

    log_tensorboard(folder, tb_out, data_loaders,
                    model=model, remove_old_events=True,
                    input_shape=(1, 3, 32, 32)
                    )


def check_tensorboard_one_epoch():
    folder_names_ = search_rgx(
        r'Cifar(Mobile)?Min(FP-(LS|ALL))?', join(BASE_FOLDER, 'one_epoch_test')
    )
    assert folder_names_

    for idx, folder_name in enumerate(folder_names_):
        out_name = join('one_epoch_test', folder_name)
        folder = join(BASE_FOLDER, 'one_epoch_test', folder_name)

        if idx == 0:
            add_images = True
        else:
            add_images = False
        _check_tensorboard(folder, out_name, add_images=add_images)


if __name__ == '__main__':
    # one_model_test('CifarMobileMin')
    # one_epoch_test()
    # check_tensorboard_one_epoch()

    run()
