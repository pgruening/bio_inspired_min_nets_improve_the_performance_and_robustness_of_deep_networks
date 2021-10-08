import copy
from os.path import join

import torch
from DLBio import kwargs_translator, pt_run_parallel
from DLBio.helpers import check_mkdir, load_json, search_rgx
from helpers import get_data_loaders, load_model, predict_needed_gpu_memory
from log_tensorboard import log_tensorboard

DEFAULT_KWARGS = {
    'comment': 'exp_11.1',
    # training params taken from densenet paper
    'lr': 0.1,
    'wd': 0.0001,
    'mom': 0.9,
    'bs': 64,
    'opt': 'SGD',
    # optimizer extras: use nesterov momentum
    'opt_kw': kwargs_translator.to_kwargs_str({'nesterov': [True]}),

    'train_interface': 'classification',
    # 'use_data_parallel': None,

    # model / ds specific params
    'in_dim': 3,
    'out_dim': 10,

    # scheduling
    'epochs': 300,
    'lr_steps': 0,
    'fixed_steps': [150, 225],

    # dataset
    'dataset': 'cifar_10',

    # model saving
    'sv_int': 0,
    # 'early_stopping': None,
    # 'es_metric': 'val_acc',
}

EXE_FILE = 'run_training.py'
BASE_FOLDER = 'experiments/exp_11_1'

# possible densenets: 'CifarDenseJOVFP', 'CifarDenseAbsReLU', 'CifarDenseNet'
MODEL = 'CifarDenseNet'

FP_BLOCKS = ['CifarDenseJOVFP', 'CifarDenseAbsReLU', 'CifarDenseMin']

MODELS = [MODEL, 'CifarDenseMin']
#MODELS = ['CifarDenseMin']
#MODELS = [MODEL]

DEPTHS_AND_GROWTH_RATES = list(zip(
    #['100', '250', '190'],
    #[12, 24, 40]
    # number of blocks: 3,9,16
    ['22', '58', '100'],
    [12, 12, 12]
))
EFFICIENT = True
COMPRESSION = .5

# use lists because the models do not fit on one gpu
#AVAILABLE_GPUS = [[0, 1], [2, 3]]
#AVAILABLE_GPUS = [0]
AVAILABLE_GPUS = [0, 1, 2, 3]
SEEDS = [9, 507, 723]


class TrainingProcess(pt_run_parallel.ITrainingProcess):
    def __init__(self, **kwargs):
        self.start_time = -1
        self.device = -1

        self.__name__ = 'Exp. 11.1 training process'
        self.module_name = EXE_FILE
        self.kwargs = kwargs
        self.mem_used = None


def _run(param_generator):
    make_object = pt_run_parallel.MakeObject(TrainingProcess)
    pt_run_parallel.run_bin_packing(
        param_generator(), make_object,
        available_gpus=AVAILABLE_GPUS,
        log_file=join(BASE_FOLDER, 'parallel_train_log.txt')
    )

    if False:
        pt_run_parallel.run(param_generator(), make_object,
                            available_gpus=AVAILABLE_GPUS,
                            do_not_check_free_gpus=True,
                            parallel_mode=False
                            )


def run():
    default_kwargs = copy.deepcopy(DEFAULT_KWARGS)
    base_folder = join(BASE_FOLDER, 'exp_data')

    def param_generator():
        yield from _param_generator(default_kwargs, base_folder)

    _run(param_generator)


def _param_generator(default_kwargs, base_folder, seeds=SEEDS):
    for seed in seeds:
        for model_type in MODELS:
            for (depth, growth_rate) in DEPTHS_AND_GROWTH_RATES:

                output = copy.deepcopy(default_kwargs)
                output['model_type'] = model_type
                output['seed'] = seed
                model_kw = {
                    # what stride ?!?
                    'stride': [1],
                    'growth_rate': [growth_rate],
                    'n_blocks': [depth],
                    'efficient': [EFFICIENT],
                    'compression': [COMPRESSION]
                }

                if model_type in FP_BLOCKS:
                    model_kw['k'] = [3]
                    model_kw['q'] = [2]

                output['folder'] = join(
                    base_folder,
                    f'{model_type}_k{growth_rate}_L{depth}_s{seed}'
                )
                output['model_kw'] = kwargs_translator.to_kwargs_str(model_kw)

                output['mem_used'] = predict_needed_gpu_memory(
                    output, input_shape=(
                        output['bs'], output['in_dim'], 32, 32),
                    device=AVAILABLE_GPUS[0]
                )
                yield output


def one_epoch_test(default_kwargs=DEFAULT_KWARGS):
    output = copy.deepcopy(default_kwargs)
    base_folder = join(BASE_FOLDER, 'one_epoch_test')
    output['folder'] = base_folder

    output['epochs'] = 1
    output['comment'] = f"exp_11.1: one_epoch_tests"
    output['do_overwrite'] = None
    output['sv_int'] = 0
    output['seed'] = 0

    output.pop('early_stopping')
    output.pop('fixed_steps')

    def param_generator():
        yield from _param_generator(output, base_folder, seeds=[0])

    _run(param_generator)


def _check_tensorboard(folder, out_name, *, add_images):
    tb_out = join(BASE_FOLDER, 'tboard', out_name)
    check_mkdir(tb_out)

    options = load_json(join(folder, 'opt.json'))
    assert options is not None, f'no options at: {folder}'

    model = load_model(options, 'cpu', map_location=torch.device(
        'cpu'), new_model_path=join(folder, 'model.pt')
    )

    data_loaders = get_data_loaders(options) if add_images else None
    log_tensorboard(folder, tb_out, data_loaders,
                    model=model, remove_old_events=True,
                    input_shape=(1, 3, 32, 32)
                    )


def check_tensorboard_one_epoch():
    rgx = r'CifarDense(AbsReLU|JOVFP|Net|Min)(.)*'
    folder_names_ = search_rgx(
        rgx, join(BASE_FOLDER, 'one_epoch_test')
    )
    assert folder_names_

    for idx, folder_name in enumerate(folder_names_):
        out_name = join('one_epoch_test', folder_name)
        folder = join(BASE_FOLDER, 'one_epoch_test', folder_name)

        add_images = idx == 0
        _check_tensorboard(folder, out_name, add_images=add_images)


def predict_size():
    for (L, k) in DEPTHS_AND_GROWTH_RATES:
        print(L, k)
        _predict_size(L, k)


def _predict_size(L, k):
    """
    this function does not make sense...
    from the LUA code
    for i = 1, N do 
         addLayer(model, nChannels, opt)
         nChannels = nChannels + opt.growthRate
    end
    !!!
    if bottleneck then N = N/2 end
    """
    k = int(k)
    L = int((int(L) - 4) / 3) // 2
    value = 0
    I = 0

    for _ in range(3):
        for _ in range(0, L - 1):
            I += k
            value += (1 * I * 4 * k) + (4 * k * k * 9)

    print(value / (10**6))


if __name__ == '__main__':
    # predict_size()
    #xxx = 0
    #    print('no routine selected')
    # one_epoch_test()
    # check_tensorboard_one_epoch()
    run()
