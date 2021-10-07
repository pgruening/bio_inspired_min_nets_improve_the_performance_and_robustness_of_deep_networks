from DLBio.helpers import MyDataFrame, search_rgx, load_json, set_plt_font_size
import matplotlib.pyplot as plt
import numpy as np
from os.path import join

from experiments.exp_10.exe_eval_exp_10 import update
from experiments.exp_11_1.exe_eval_exp_11_1 import LINESTYLES

BASE_FOLDER = 'experiments/exp_10_1'
MODEL_FOLDER = join(BASE_FOLDER, 'exp_data')
RGX = r'CifarResNet_s\d+_N9'


def get_functions():
    return {
        'mean': lambda x: np.mean(x, 0),
        'median': lambda x: np.median(x, 0),
        'min': _get_min_curve,
        'max': _get_max_curve,
    }


COLORS = {
    'mean': 'k',
    'median': 'c',
    'max': 'r',
    'min': 'b'
}

LINESTYLES = {
    'mean': '-',
    'median': '-',
    'max': '-.',
    'min': '--'
}


def run():
    # train and test error curves
    tr_curves = []
    curves_test = []
    for folder in search_rgx(RGX, MODEL_FOLDER):
        tr_curves, curves_test = update(
            join(MODEL_FOLDER, folder),
            tr_curves, curves_test
        )

    create_plot(
        tr_curves, join(BASE_FOLDER, 'train_curve.pdf'),
        ylabel='Training error',
        ylim=[0., 14.]
    )
    create_plot(
        curves_test, join(BASE_FOLDER, 'test_curve.pdf'),
        ylabel='Test error',
        draw_line=6.97,
        ylim=[4., 18.]
    )
    create_plot(
        curves_test, join(BASE_FOLDER, 'test_curve.png'),
        ylabel='Test error',
        draw_line=6.97,
        ylim=[4., 18.]
    )

    data = np.stack(curves_test, 0)
    with open(join(BASE_FOLDER, 'results.txt'), 'w') as file:
        file.write((
            f'Median last: {round(np.median(data,0)[-1], 3)} \n' +
            f'Mean last: {round(np.mean(data,0)[-1], 3)} \n' +
            f'Std last: {round(np.std(data,0)[-1], 3)} \n' +
            f'Min last: {round(np.min(data,0)[-1], 3)} \n' +
            f'Max last: {round(np.max(data,0)[-1], 3)} \n' +
            f'Median min: {round(np.median(data,0).min(), 3)} \n'
        ))


def update(folder, tr_curves, curves_test):
    log = load_json(join(folder, 'log.json'))

    tr_curves.append(
        np.array(log['er'])
    )
    curves_test.append(
        np.array(log['val_er'])
    )
    return tr_curves, curves_test


def create_plot(data, path, *, ylabel, ylim, draw_line=None):
    plt.figure(figsize=(12, 8))
    set_plt_font_size(26)

    data = np.stack(data, 0)
    for key, f in get_functions().items():
        plot_kwargs = {
            'color': COLORS[key],
            'linestyle': LINESTYLES[key],
            'label': key,
            'linewidth': 3
        }
        plt.plot(f(data), **plot_kwargs)

    if draw_line is not None:
        plt.axhline(y=draw_line, color='k', linewidth=4, linestyle='dotted')

    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.ylim(ylim)

    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _get_min_curve(data):
    return _get_curve(data, np.argmin)


def _get_max_curve(data):
    return _get_curve(data, np.argmax)


def _get_curve(data, index_fcn):
    # return the curve where the mean value of the last 10 indices was minimal
    index = index_fcn(data[:, -10:].mean(1))
    return data[index, :]


if __name__ == '__main__':
    run()
