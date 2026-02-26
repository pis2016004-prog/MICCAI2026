#!/bin/sh
cd /drive/Brats2017/MICCAI_FINAL

DATASET_PATH=../DATASET/DATASET_Lungs/DATASET_Lungs

export PYTHONPATH=./
export RESULTS_FOLDER=./output_lung
export unetr_pp_preprocessed=../DATASET/DATASET_Lungs/DATASET_Lungs/unetr_pp_raw/unetr_pp_raw_data/Task06_Lung
export unetr_pp_raw_data_base=../DATASET/DATASET_Lungs/DATASET_Lungs/unetr_pp_raw

python unetr_pp/run/run_training.py 3d_fullres unetr_pp_trainer_lung 6 0 --continue_training
