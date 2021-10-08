import re
import shutil
from os import walk
from os.path import isdir, join
from DLBio.helpers import check_mkdir
from config import DOWNLOADED_MODELS, DOWNLOADED_PREDICTIONS

PRED_RGX = (
    r'Cifar(Dense(Net|Min)|PyrResNet|MinFP-LS)_(s\d+_N\d+|k\d+_L\d+_s\d+).npy'
)

DO_COPY = True


def run():
    copy_predictions()
    copy_models()


def copy_predictions():
    def match(f):
        return bool(re.match(PRED_RGX, f))

    for root, _, files in walk(DOWNLOADED_PREDICTIONS):
        pred_files = [f for f in files if match(f)]
        if not pred_files:
            continue

        experiment_folder = root.split('/')[1]
        assert bool(re.match(r'exp_\d+(.*)', experiment_folder))

        for file in pred_files:
            source = join(root, file)
            tar_dir = join(
                'experiments', experiment_folder, 'jpeg_model_predictions'
            )
            check_mkdir(tar_dir)
            target = join(tar_dir, file)
            copy(source, target)


def copy_models():
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

            copy(source, target)


def copy(source, target):
    if DO_COPY:
        shutil.copy(source, target)

    print((
        'Moved: \n' +
        f'{source} \n' +
        '-> \n' +
        f'{target} \n'
    ))


if __name__ == '__main__':
    run()
