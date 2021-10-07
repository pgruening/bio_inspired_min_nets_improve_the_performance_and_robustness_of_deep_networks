import os
from os.path import join
import shutil
from DLBio.helpers import check_mkdir
import re

SAVED_MODELS = 'saved_models'
REMOVE_MODELS = False


def run():
    for root, dirs, files in os.walk('.'):
        remove_dir(root, dirs, '__pycache__')
        remove_dir(root, dirs, 'src_copy')
        #save_and_remove_models(root, files)


def remove_dir(root, dirs, name):
    cache_dirs = [d for d in dirs if d == name]
    if cache_dirs:
        for d in cache_dirs:
            print(join(root, d))
            shutil.rmtree(join(root, d))


def save_and_remove_models(root, files):
    model_files = [f for f in files if f == 'model.pt']
    if not model_files:
        return

    for file in model_files:
        parent_folder = root.split('/')[-1]
        experiment_folder = root.split('/')[2]
        assert bool(re.match(r'exp_\d+(.*)', experiment_folder))

        source = join(root, file)
        target = join(SAVED_MODELS, experiment_folder, parent_folder, file)
        check_mkdir(target)
        shutil.copy(source, target)
        print((
            'Moved: \n' +
            f'{source} \n' +
            '-> \n' +
            f'{target} \n'
        ))

        if REMOVE_MODELS:
            pass


if __name__ == '__main__':
    run()
