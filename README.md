# Code Submission for: Bio-inspired Min-Nets Improve the Performance and Robustness of Deep Networks

## Run with docker

To build a docker environment, change dir to 'docker' and run:
```
bash docker_build.sh
```
Move back to the parent folder and run
```
bash docker_run.sh
```
to start the docker environment. Note that this environments uses Pytorch 1.8 and Cuda 11.1.

## Run with virtualenv

In a new [virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) run
```
pip install -r venv_requirements.txt
```

The PYTHONPATH needs to be updated. Append this to your '.bashrc':
```
export PYTHONPATH=path/to/this/repository:$PYTHONPATH
```

Furthermore, you need to specify the variable 'DATA_FOLDER' in 'config.py'.

## Running the experiments

Python modules meant to be executed include the prefix 'exe_' in their name. The experiments of the paper are located in the experiments folder. Note that each module is meant to be started from the repositories folder, e.g. to evaluate experiment 10 run:

```
python experiments/exp_10/exe_train_exp_10.py
```

Here is a list of all experiments:

### exp_10: ResNet and PyrblockNet experiment on Cifar-10 classification and JPEG compression

* 'exe_train_exp_10.py': retrain the networks on Cifar-10. Creates log-files in 'experiments/exp_10/exp_data'
* 'exe_eval_exp_10.py': runs classification evaluation
* 'exe_exp10_jpeg_tests.py': evaluates the predictions in saved in 'jpeg_model_predictions'. To again compute the predictions, run this module with the flag 'RUN_PREDICTIONS' set to 'True'
### exp_10_1: Additional variance experiment on Cifar-10 classification

* 'exe_train_exp_10_1.py': trains 12 ResNet (N=9) models on Cifar-10.
* 'exe_eval_exp_10_1.py': computes the training curve evaluation.

### exp_11_1: DenseNet experiment on Cifar-10 classification and JPEG compression

* 'exe_train_exp_11_1.py': retrain the networks on Cifar-10. Creates log-files in 'experiments/exp_11_1/exp_data'
* 'exe_eval_exp_11_1.py': runs classification evaluation
* 'exe_exp10_jpeg_tests.py': evaluates the predictions in saved in 'jpeg_model_predictions'. To again compute the predictions, run this module with the flag 'RUN_PREDICTIONS' set to 'True'

## Other modules:

* 'run_training.py': Module to train a CNN. Is started in any 'exe_train'-file as a subprocess.
* 'run_jpeg_robustness_test.py': Module to create predictions on compressed data.
* 'datasets/ds_ds_cifar10_compression_test.py': run this module to create 10 compressed Cifar-10 test-sets. The data are save to 'DATA_FOLDER' in the 'config.py' file

All models are located in the 'model' folder, the dataset code is in the 'datasets' folder.

## Downloading the models

All models are save as a zip file on [google drive](). To add the to the respective experiment folders, extract the zip file, change the path variable 'DOWNLOADED_MODELS' in 'exe_move_models_to_exp_folder.py' and run this module. 


git archive -o min_nets_latest.zip HEAD