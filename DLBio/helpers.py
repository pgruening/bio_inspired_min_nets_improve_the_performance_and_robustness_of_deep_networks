import datetime
import inspect
import json
import os
import re
import shutil
import warnings
from collections import namedtuple
from datetime import datetime
from os.path import isfile, join, splitext

import matplotlib
import numpy as np
import pandas as pd

SRC_COPY_FOLDER = ''
TB_LOG_FOLDER = ''


def check_mkdir(directory_or_file, is_dir=False):
    if is_dir:
        # NOTE: useful if the directory has a point in the foldername
        directory = directory_or_file
    else:
        directory = _get_directory(directory_or_file)

    if not os.path.isdir(directory):
        os.makedirs(directory)


def save_options(file_path, options):
    warnings.warn(
        'helpers.save_options is deprecated and has been moved to pytorch_helpers'
    )
    if not hasattr(options, "__dict__"):
        out_dict = dict(options._asdict())
    else:
        out_dict = options.__dict__

    # add the current time to the output
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y-%H:%M:%S")

    out_dict.update({
        'start_time': dt_string
    })

    with open(file_path, 'w') as file:
        json.dump(out_dict, file)


def load_json(file_path):
    if not isfile(file_path):
        return None

    with open(file_path, 'r') as file:
        out = json.load(file)
    return out


def get_sub_dataframe(df, cols_and_vals):
    df = df.copy()
    where_and = np.ones(df.shape[0]) > 0
    for key, values in cols_and_vals.items():
        if isinstance(values, list):
            _where_or = np.ones(df.shape[0]) == 0
            for v in values:
                tmp = np.array(df[key] == v)
                _where_or = np.logical_or(_where_or, tmp)
        else:
            v = values
            _where_or = np.array(df[key] == v)

        where_and = np.logical_and(where_and, _where_or)

    return df[where_and]


def copy_source(out_folder, max_num_files=100, do_not_copy_folders=None):
    """Copies the source files of the current working dir to out_folder

    Parameters
    ----------
    out_folder : str
        where to copy the files
    max_num_files : int, optional
        makes sure not to copy files in endless loop, by default 100
    do_not_copy_folders : list of str, optional
        list of folder names that are not supposed to be copied, by default None
    """
    if do_not_copy_folders is None:
        warnings.warn('No folders excluded from source_copy! Are you sure?')
        do_not_copy_folders = []

    do_not_copy_folders += ['__pycache__']

    out_f = out_folder.split('/')[-1]
    out_folder = join(out_folder, 'src_copy')
    print(f'Source copy to folder: {out_folder}')
    ctr = 0
    for root, _, files_ in os.walk('.'):

        # remove the base folder:
        # e.g. './models' or '.' -> grab anything behind that
        current_folder = re.match(r'^(.$|.\/)(.*)', root).group(2)

        # NOTE: only looks at the root folder, maybe adjustments are needed
        # at some point
        do_continue = False
        for x in current_folder.split('/'):
            if x in do_not_copy_folders:
                do_continue = True
                break
        if do_continue:
            continue

        # do not copy anything that is on the path to the output folder (out_f)
        if out_f in root.split('/'):
            continue

        # make sure there is no never-ending copy loop!!
        tmp = [1 for x in current_folder.split('/') if x in ['src_copy']]
        if tmp:
            continue

        files_ = [x for x in files_ if splitext(x)[-1] in ['.py']]
        if not files_:
            continue
        else:
            print(f'src_copy from folder: {root}')

        for file in files_:
            dst = file
            if current_folder != '':
                dst = join(current_folder, file)

            dst = join(out_folder, dst)
            src = join(root, file)

            check_mkdir(dst)
            shutil.copy(src, dst)

            ctr += 1
            assert ctr < max_num_files, 'too many files copied.'


def search_in_all_subfolders(rgx, folder, search_which='files', match_on_full_path=False, depth=None):
    # TODO: rename to find
    def is_rgx_match(rgx, x):
        return bool(re.fullmatch(rgx, x))
    outputs = []
    assert os.path.isdir(folder), f'folder not found: {folder}'

    for root, dirs_, files_ in os.walk(folder):
        if depth is not None:
            # Don't search to deep into the folder tree
            # remove base path
            tmp = root[len(folder):]
            # count folders
            current_depth = len(tmp.split('/')) - 1
            if current_depth > depth:
                continue

        if search_which == 'files':
            to_search = files_
        elif search_which == 'dirs':
            to_search = dirs_
        elif search_which == 'all':
            to_search = files_ + dirs_
        else:
            raise ValueError(f'unknown search_type: {search_which}')

        if not to_search:
            continue

        if match_on_full_path:
            tmp = [
                join(root, x) for x in to_search if is_rgx_match(rgx, join(root, x))
            ]
        else:
            tmp = [join(root, x) for x in to_search if is_rgx_match(rgx, x)]

        outputs += tmp

    return outputs


def search_rgx(rgx, path):
    return [x for x in os.listdir(path) if bool(re.match(rgx, x))]


def get_from_module(py_module, bool_fcn):
    """Check all objects of a python module and return all for which the
    bool function returns true

    Parameters
    ----------
    py_module : python module
        import x -> pass x
    bool_fcn : function that returns a boolean
        y = fcn(x) -> x = object found in py_module, y in [True, False]

    Returns
    -------
    list of tuples (name (str), object)
    """
    cls_members = inspect.getmembers(py_module, bool_fcn)
    return cls_members


def get_subfolders(base_folder):
    # TODO: rename to find
    return next(os.walk(base_folder))[1]


def get_parent_folder(folder):
    return '/'.join(folder.split('/')[:-1])


def dict_to_options(opt_dict):
    """Transforms a dictionary into an options object,
    similar to the object created by the ArgumentParser. 

    Parameters
    ----------
    opt_dict : dict

    Returns
    -------
    object
        "key: value" -> object.key == value
    """
    Options = namedtuple('Options', opt_dict.keys())

    return Options(**opt_dict)


class MyDataFrame():
    def __init__(self, verbose=0):
        self.x = dict()
        self.max_num_items = 0
        self.verbose = verbose

    def update(self, in_dict, add_missing_values=False, missing_val=np.nan):
        for k, v in in_dict.items():

            if isinstance(v, list):
                warnings.warn(f'Input for {k} is list, consider add_col.')

            if k not in list(self.x.keys()):
                if self.verbose > 0:
                    print(f'added {k}')
                # case 1: df just intialized
                if self.max_num_items == 0:
                    self.x[k] = [v]
                else:
                    # case 2: entire new key is added
                    if add_missing_values:
                        # fill with missing values to current num items
                        self.x[k] = [missing_val] * self.max_num_items
                        self.x[k].append(v)

            else:
                self.x[k].append(v)

        if add_missing_values:
            self._add_missing(missing_val)

    def _add_missing(self, missing_val):
        self._update()
        for k in self.x.keys():
            if self.verbose > 1 and len(self.x[k]) < self.max_num_items:
                print(f'add missing: {k}')

            while len(self.x[k]) < self.max_num_items:
                self.x[k].append(missing_val)

    def _update(self):
        self.max_num_items = max([len(v) for v in self.x.values()])

    def add_col(self, key, col):
        self.x[key] = col

    def get_df(self, cols=None):
        assert self._check_same_lenghts()
        return pd.DataFrame(self.x, columns=cols)

    def _check_same_lenghts(self):
        len_vals = {k: len(v) for k, v in self.x.items()}
        if len(set(len_vals.values())) > 1:
            print(len_vals)
            return False

        return True


def set_plt_font_size(font_size):
    # font = {'family' : 'normal',
    #    'weight' : 'bold',
    #    'size'   : 22}
    font = {'size': font_size}
    matplotlib.rc('font', **font)


class ToBin():
    def __init__(self, n):
        self.n = n

    def __call__(self, arr):
        if isinstance(arr, int):
            return np.stack(self._to_bin(arr), 0)
        assert arr.ndim == 1
        out = [self._to_bin(int(x)) for x in list(arr)]
        return np.stack(out, 0)

    def _to_bin(self, x):
        return np.array([float(s) for s in self._bin(x)])

    def _bin(self, x):
        return format(x, 'b').zfill(self.n)


def get_dataframe_from_row(df, index):
    row = dict(df.iloc[index])
    return pd.DataFrame({k: [v] for k, v in row.items()})


def get_parent_folder(directory_or_file):
    directory = _get_directory(directory_or_file)
    return directory.split('/')[-1]


def _get_directory(directory_or_file):
    if os.path.splitext(directory_or_file)[-1] != '':
        directory = '/'.join(directory_or_file.split('/')[:-1])
    else:
        directory = directory_or_file
    return directory
