import copy
from os.path import isfile, join

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from DLBio.helpers import (MyDataFrame, check_mkdir, get_sub_dataframe,
                           load_json, search_rgx)
from DLBio.kwargs_translator import get_kwargs
from experiments.eval_methods import save_curve_plot
from experiments.exp_10.cfg_exp10 import COLORS, RGX
from experiments.exp_10.exe_exp10_jpeg_tests import NAMES
from experiments.exp_11_1.exe_eval_exp_11_1 import LINESTYLES

BASE_FOLDER = 'experiments/exp_10'
IMAGE_FOLDER = join(BASE_FOLDER, 'test_error_images')
check_mkdir(IMAGE_FOLDER)
MODEL_FOLDER = join(BASE_FOLDER, 'exp_data')


USED_METRIC = 'last_val_er'
AGGREGATOR = 'mean'
AGG = {
    'min_val_er': ('mean', 'std', 'min', 'max', 'median'),
    'last_val_er': ('mean', 'std', 'min', 'max', 'median'),
    'num_params': 'first',
}
COLORS['CifarMinFP-LS'] = 'b'
IDENT_KEYS = ['model_type', 'N', 'seed']

# models of interest
MOIS = ['CifarMinFP-LS', 'CifarPyrResNet']
LINESTYLES = {
    'CifarMinFP-LS': '-',
    'CifarPyrResNet': '--',
}
NAMES = {
    'CifarMinFP-LS': 'Min-Net',
    'CifarPyrResNet': 'ResNet',
}

SUBSETS = {
    'PyrResNet': ['CifarMinFP-LS', 'CifarPyrResNet'],
}


def run():
    folders_ = search_rgx(RGX, MODEL_FOLDER)
    assert folders_

    df = MyDataFrame()

    for folder in folders_:
        df = update(df, join(MODEL_FOLDER, folder))

    df = df.get_df().sort_values(by=USED_METRIC, ignore_index=True)
    df.to_csv(join(BASE_FOLDER, 'results.csv'))

    print(df.to_markdown())

    df_grouped = df.groupby(by=['model_type', 'N'], as_index=False).agg(
        AGG).sort_values(by=(USED_METRIC, 'mean'))
    df_grouped.to_csv(join(BASE_FOLDER, 'grouped_results.csv'))

    with open(join(BASE_FOLDER, 'results.md'), 'w') as file:
        file.write(df_grouped.to_markdown())

    save_curve_plot(
        df_grouped, BASE_FOLDER, colors_=COLORS, pref='data_'
    )

    # models of interest
    for pref, subset in SUBSETS.items():
        tmp = get_sub_dataframe(df_grouped, {'model_type': subset})
        colors = copy.deepcopy(COLORS)
        if pref != 'all':
            colors['CifarMinFP-LS-RNBasic'] = 'b'
            colors['CifarResNet'] = 'k'
        save_curve_plot(
            tmp, IMAGE_FOLDER, colors_=colors, val_key=USED_METRIC,
            ylabel='Test error after 200 epochs', pref=pref + '_cifar_resnet_',
            linestyles_=LINESTYLES, aggregator=AGGREGATOR,
            new_name=NAMES)


def get_sub_table(df, **kwargs):
    df = df.copy()
    for k, v in kwargs.items():
        df = df[df[k] == v]

    return df


def update(df, folder):
    tmp = get_results(folder)
    if tmp is not None:
        df.update(tmp)
    return df


def get_results(folder):
    opt = load_json(join(folder, 'opt.json'))
    log = load_json(join(folder, 'log.json'))
    if log is None:
        return None
    if log['epoch'][-1] < 180:
        return None

    model_specs = load_json(join(folder, 'model_specs.json'))
    model_kwargs = get_kwargs(opt['model_kw'])

    return {
        'model_type': opt['model_type'],
        'min_val_er': min(log['val_er']),
        'last_val_er': log['val_er'][-1],
        'seed': opt['seed'],
        'N': int(model_kwargs.get('N', ['-'])[-1]),
        'num_params': int(model_specs['num_params']),
    }


if __name__ == '__main__':
    run()
