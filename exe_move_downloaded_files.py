import re
import shutil
from os import walk
from os.path import isdir, join
from config import DOWNLOADED_MODELS, DOWNLOADED_PREDICTIONS


def run():
    for root, _, files in walk(DOWNLOADED_MODELS):
        model_files = [f for f in files if f == 'model.pt']
        if not model_files:
            continue

        for file in model_files:
            experiment_folder = root.split('/')[1]
            parent_folder = root.split('/')[2]
            assert bool(re.match(r'exp_\d+(.*)', experiment_folder))

            source = join(root, file)
            tar_dir = join(
                'experiments', experiment_folder, 'exp_data', parent_folder
            )
            assert isdir(tar_dir), tar_dir
            target = join(tar_dir, file)

            #shutil.copy(source, target)
            print((
                'Moved: \n' +
                f'{source} \n' +
                '-> \n' +
                f'{target} \n'
            ))


if __name__ == '__main__':
    run()
