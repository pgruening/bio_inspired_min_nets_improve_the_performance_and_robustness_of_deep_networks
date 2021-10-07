from DLBio.helpers import load_json, search_rgx, MyDataFrame
from experiments.eval_methods import save_curve_plot
from os.path import join
from DLBio.kwargs_translator import get_kwargs

BASE_FOLDER = 'experiments/exp_11_1'
MODEL_FOLDER = join(BASE_FOLDER, 'exp_data')
RGX = r'CifarDense(Min|Net)_k\d+_L\d+_s\d+'

LAST_VAL_ERROR = 'last_val_er'
AGG = {
    LAST_VAL_ERROR: ('mean', 'std', 'min', 'max'),
    'min_val_er': ('mean', 'std', 'min', 'max')
}

COLORS = {
    'CifarDenseNet': 'k',
    'CifarDenseMin': 'b'
}

LINESTYLES = {
    'CifarDenseNet': '--',
    'CifarDenseMin': '-'
}


def run():
    folders_ = search_rgx(RGX, MODEL_FOLDER)
    assert folders_

    df = MyDataFrame()
    for folder in folders_:
        folder = join(MODEL_FOLDER, folder)
        df = update(df, folder)

    df = df.get_df().sort_values(LAST_VAL_ERROR)
    df.to_csv(join(BASE_FOLDER, 'all_results.csv'))

    df_grouped = df.groupby([
        'model_type', 'k', 'num_params', 'N'
    ], as_index=False).agg(AGG).sort_values((LAST_VAL_ERROR, 'mean'))
    df_grouped.to_csv(join(BASE_FOLDER, 'grouped_results.csv'))
    save_as_latex(df_grouped.round(2), join(
        BASE_FOLDER, 'tex_grouped_results.txt'))

    for key in ['last_val_er', 'min_val_er']:
        save_curve_plot(
            df_grouped, BASE_FOLDER, val_key=key,
            ylabel='Test error after 300 epochs', pref='cifar_densenet_',
            colors_=COLORS, linestyles_=LINESTYLES
        )


def save_as_latex(df, path):
    with open(path, 'w') as file:
        file.write(df.to_latex())


def update(df, folder):
    log = load_json(join(folder, 'log.json'))
    opt = load_json(join(folder, 'opt.json'))
    model_specs = load_json(join(folder, 'model_specs.json'))
    model_kwargs = get_kwargs(opt['model_kw'])

    if log['epoch'][-1] < 200:
        return df

    out = {
        'model_type': opt['model_type'],
        'k': int(model_kwargs['growth_rate'][0]),
        'N': int(model_kwargs['n_blocks'][0]),
        'min_val_er': min(log['val_er']),
        LAST_VAL_ERROR: log['val_er'][-1],
        'num_params': model_specs['num_params'],
        'epoch': log['epoch'][-1]
    }

    df.update(out)
    return df


if __name__ == '__main__':
    run()
