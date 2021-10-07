import warnings
from os.path import isfile, join

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from DLBio.helpers import check_mkdir, load_json, set_plt_font_size, get_sub_dataframe


def save_curve_plot(
    df, out_folder, val_key='min_val_er', colors_=None,
    pref='', ylim=None, name_key='model_type', ylabel=None,
    legend_order=None, plot_std=True, aggregator='mean', plot_fill=True,
        linestyles_=None):

    set_plt_font_size(26)
    _, ax = plt.subplots(1, figsize=(15, 15))
    for key in set(df['model_type']):
        tmp = df[df['model_type'] == key].copy()
        tmp = tmp.sort_values(by='N')
        x = np.array(tmp['num_params'] // 1000)
        y = np.array(tmp[(val_key, aggregator)])
        z = np.array(tmp[(val_key, 'std')])
        vmin = np.array(tmp[(val_key, 'min')])
        vmax = np.array(tmp[(val_key, 'max')])

        name = tmp.iloc[0, :][name_key][0]

        # TODO set hue order
        plot_kwargs = {
            'linewidth': 4,
            'label': name,
            'marker': 'd',
            'markersize': 14,
        }
        fill_kwargs = {
            'alpha': .2
        }
        if colors_ is not None:
            plot_kwargs['color'] = colors_[key]
            fill_kwargs['color'] = colors_[key]

        if linestyles_ is not None:
            plot_kwargs['linestyle'] = linestyles_[key]

        if plot_std:
            plt.errorbar(
                x, y, z, **plot_kwargs
            )
        else:
            plt.plot(x, y, **plot_kwargs)

        if plot_fill:
            if x.ndim == 2:
                assert x.shape[1] == 1
                x = x[:, 0]

            plt.fill_between(x, vmin, vmax, **fill_kwargs)

    plt.xlabel('Number of Parameters (k)')
    if ylabel is None:
        plt.ylabel(val_key)
    else:
        plt.ylabel(ylabel)

    plt.legend()

    if legend_order is not None:
        # sort legends according to
        handles, labels = ax.get_legend_handles_labels()
        assert len(labels) == len(legend_order), f'{legend_order} vs. {labels}'
        tmp = [labels.index(x) for x in legend_order]
        new_handles = [handles[i] for i in tmp]
        # sort both labels and handles by labels
        ax.legend(new_handles, legend_order)

    plt.grid()
    if ylim is not None:
        plt.ylim(ylim)

    plt.tight_layout()
    plt.savefig(join(out_folder, pref + val_key + '.png'))
    plt.savefig(join(out_folder, pref + val_key + '.pdf'))
    plt.close()


def create_jpeg_plot(df, *, pref, key, colors, linestyles, image_out, names, min_q=1, max_q=10):
    set_plt_font_size(26)

    def get_y(data):
        return np.array([data[f'{key}_{i}'] for i in range(min_q, max_q)])

    for N in sorted(set(df['N'])):
        plt.figure(figsize=(12, 12))
        for mt in sorted(set(df['model_type'])):
            tmp = get_sub_dataframe(df, {
                'model_type': mt,
                'N': N
            })

            plot_kwargs = {
                'label': names[mt],
                'color': colors[mt],
                'linewidth': 4,
                'marker': 'd',
                'markersize': 14,
                'linestyle': linestyles[mt]
            }
            fill_kwargs = {
                'alpha': .2,
                'color': colors[mt]
            }
            x = 100. - 10. * np.arange(min_q, max_q)
            plt.errorbar(
                x, get_y(tmp.mean()), get_y(tmp.std()), **plot_kwargs
            )
            plt.fill_between(x, get_y(tmp.min()),
                             get_y(tmp.max()), **fill_kwargs)

        plt.gca().invert_xaxis()
        plt.xlabel('Quality Percentage')
        plt.ylabel({
            'er': 'Test error', 'nc': 'Percentage of changed predictions'
        }[key])

        plt.legend()
        plt.grid()
        plt.tight_layout()

        plt.savefig(join(image_out, f'{pref}_N{N}.png'))
        plt.savefig(join(image_out, f'{pref}_N{N}.pdf'))
        plt.close()


def create_table(df, *, pref, key, out_folder, names, group_cols, min_q=1, max_q=9, decimals=2):
    # rename model_type
    df['Model'] = [names[r['model_type']] for _, r in df.iterrows()]
    used_cols = group_cols + [f'{key}_{q}' for q in range(min_q, max_q + 1)]
    grouped_df = df.copy().groupby(
        group_cols, as_index=False
    ).mean()[used_cols].round(decimals)

    # remove key prefix for cols
    grouped_df = grouped_df.rename(columns={
        f'{key}_{q}': f'{int(100 - 10.*q)}%' for q in range(min_q, max_q + 1)
    })

    with open(join(out_folder, pref + 'result_table.txt'), 'w') as file:
        file.write(grouped_df.to_latex(index=False))


def create_bar_plots_per_size(df, out_folder, normalize_depth=False, n_bins=-1, colors=None, ylim=None):

    for N in set(df['N']):
        tmp = df[df['N'] == N].copy()

        if normalize_depth:
            for mtype in set(tmp['model_type']):
                where = tmp['model_type'] == mtype
                x = tmp['depth'][where]
                b = x.max()
                a = x.min()
                x_normed = (x - a) / (b - a)
                x_binned = (n_bins * x_normed).round()
                tmp.loc[where, 'depth'] = x_binned

        kwargs = {}
        if colors is not None:
            kwargs['palette'] = colors

        sns.barplot(
            data=tmp, x='depth', y='percent_dead',
            hue='model_type', **kwargs
        )
        if ylim is not None:
            plt.ylim(ylim)

        if normalize_depth:
            plt.xlabel('Bin')

        plt.legend()
        plt.grid()
        plt.title(f'N: {N}')

        plt.savefig(join(out_folder, f'barplot_N{N}.png'))
        plt.close()
