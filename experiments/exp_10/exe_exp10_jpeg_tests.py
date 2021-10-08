import subprocess
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import run_jpeg_robustness_tests as jpeg_test
from DLBio import pt_run_parallel
from DLBio.helpers import (MyDataFrame, check_mkdir, get_sub_dataframe,
                           load_json, search_rgx, set_plt_font_size)
from DLBio.kwargs_translator import get_kwargs
from experiments.eval_methods import create_jpeg_plot, create_table
from experiments.exp_11_1.exe_eval_exp_11_1 import LINESTYLES
from helpers import predict_needed_gpu_memory

RUN_PREDICTIONS = False

BASE_FOLDER = 'experiments/exp_10'
IMAGE_OUT = join(BASE_FOLDER, 'images')
check_mkdir(IMAGE_OUT)

MODEL_FOLDER = join(BASE_FOLDER, 'exp_data')
RGX = r'Cifar(MinFP-LS|PyrResNet|ResNet)(|-RNBasic)_s\d+_N\d+'
EXE_FILE = 'run_jpeg_robustness_tests.py'


#AVAILABLE_GPUS = [0, 1, 2, 3]
AVAILABLE_GPUS = [0]
USED_COLS = (
    ['model_type', 'seed', 'N'] +
    [f'er_{i}' for i in range(0, 10)] +
    [f'nc_{i}' for i in range(1, 10)]
)

COLORS = {
    'CifarMinFP-LS': 'b',
    'CifarPyrResNet': 'k',
}

LINESTYLES = {
    'CifarMinFP-LS': '-',
    'CifarPyrResNet': '--',
}

SUBSETS = {
    'PyrResNet': ['CifarMinFP-LS', 'CifarPyrResNet'],
}

NAMES = {
    'CifarMinFP-LS': 'Min-Net',
    'CifarPyrResNet': 'ResNet',

}

# ---------------------------------------------------------------------------
# -----------------EVALUATE SAVED PREDICTIONS--------------------------------
# ---------------------------------------------------------------------------


def run_eval():
    df = get_min_net_df()
    df = df.loc[:, USED_COLS]

    plot_kwargs = {
        'colors': COLORS,
        'image_out': IMAGE_OUT,
        'max_q': 5,
        'names': NAMES,
        'linestyles': LINESTYLES
    }

    table_kwargs = {
        'out_folder': IMAGE_OUT,
        'max_q': 9,
        'names': NAMES,
        'decimals': 1,
        'group_cols': ['Model', 'N']
    }

    for key, subset in SUBSETS.items():
        tmp = get_sub_dataframe(df, {'model_type': subset})

        create_scatter_plot(tmp, pref=key, **plot_kwargs)

        create_jpeg_plot(tmp, pref=key + '_nc', key='nc', **plot_kwargs)
        create_jpeg_plot(tmp, pref=key + '_er', key='er',
                         min_q=0, **plot_kwargs)

        create_table(df, pref='nc_', key='nc', **table_kwargs)
        create_table(df, pref='er_', key='er', min_q=0, **table_kwargs)

    create_jpeg_plot(df, pref='all_er', key='er', min_q=0, **plot_kwargs)
    create_jpeg_plot(df, pref='all_nc', key='nc', **plot_kwargs)


def create_scatter_plot(df, *, pref, names, linestyles, colors, image_out, min_q=1, max_q=9):
    set_plt_font_size(26)

    max_q = 4

    def get_data(data, key):
        return np.array([data[f'{key}_{i}'] for i in range(min_q, max_q)])

    for N in sorted(set(df['N'])):
        plt.figure(figsize=(12, 12))
        for mt in sorted(set(df['model_type'])):
            for s in sorted(set(df['seed'])):
                tmp = get_sub_dataframe(df, {
                    'model_type': mt,
                    'N': N,
                    'seed': s
                })

                plot_kwargs = {
                    'color': colors[mt],
                    'linewidth': 2,
                    'marker': 'd',
                    'markersize': 18,
                    'linestyle': linestyles[mt]
                }

                x = get_data(tmp, 'nc')
                y = get_data(tmp, 'er')
                plt.plot(x, y, **plot_kwargs)
                #plt.text(x[0], y[0], s)

        plt.xlabel('Perc. of changed predictions')
        plt.ylabel('Test error')

        # plt.legend()
        plt.grid()
        plt.tight_layout()

        plt.savefig(join(image_out, f'scatter_{pref}_N{N}.png'))
        plt.savefig(join(image_out, f'scatter_{pref}_N{N}.pdf'))
        plt.close()


def get_min_net_df():
    df = MyDataFrame()
    for folder in search_rgx(RGX, MODEL_FOLDER):
        model_path = join(MODEL_FOLDER, folder)
        pred_data_path = join(
            BASE_FOLDER, jpeg_test.DATA_FOLDER_NAME, folder + '.npy'
        )
        df = update(df, model_path, pred_data_path)

    return df.get_df()


def update(df, model_path, pred_data_path):
    pred_data = np.load(pred_data_path)
    options = load_json(join(model_path, 'opt.json'))
    log = load_json(join(model_path, 'log.json'))
    model_kwargs = get_kwargs(options['model_kw'])

    out = {
        'model_type': options['model_type'],
        'seed': options['seed'],
        'N': int(model_kwargs['N'][0]),
        'q': float(model_kwargs['q'][0]),
    }

    # compute the robustness metrics
    for i in range(1, 10):
        out[f'er_{i}'] = jpeg_test.compute_error_for_subset(pred_data, i)
        out[f'nc_{i}'] = jpeg_test.compute_change_prob(pred_data, i)

    # add the original error
    out['er_0'] = log['val_er'][-1]

    df.update(out)
    return df

# ---------------------------------------------------------------------------
# -----------------COMPUTE MODEL PREDICTIONS ON COMPRESSED DATA--------------
# ---------------------------------------------------------------------------


def create_numpy_files():
    make_object = pt_run_parallel.MakeObject(TrainingProcess)
    pt_run_parallel.run_bin_packing(
        param_generator(), make_object,
        available_gpus=AVAILABLE_GPUS,
        log_file=join(BASE_FOLDER, 'parallel_train_log.txt'),
        max_num_processes_per_gpu=3
    )


class TrainingProcess(pt_run_parallel.ITrainingProcess):
    def __init__(self, **kwargs):
        self.start_time = -1
        self.device = -1

        self.__name__ = 'Exp. 11.1 jpeg analysis'
        self.module_name = EXE_FILE
        self.kwargs = kwargs
        self.mem_used = None

    def __call__(self):
        subprocess.call([
            'python', EXE_FILE,
            '--model_folder', self.kwargs['folder'],
            '--base_folder', BASE_FOLDER
        ])


def param_generator():
    for folder in search_rgx(RGX, MODEL_FOLDER):
        output = {'folder': join(MODEL_FOLDER, folder)}
        options = load_json(join(output['folder'], 'opt.json'))
        assert options is not None
        output['mem_used'] = predict_needed_gpu_memory(
            options, input_shape=(10, 3, 32, 32),
            device=AVAILABLE_GPUS[0])
        yield output


if __name__ == "__main__":
    if RUN_PREDICTIONS:
        create_numpy_files()

    run_eval()
