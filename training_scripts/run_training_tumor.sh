#!/bin/sh

cd /drive/Brats2017/MICCAI_FINAL
DATASET_PATH=../DATASET_Tumor


export PYTHONPATH=./
export RESULTS_FOLDER=./output_tumormic
export unetr_pp_preprocessed=../DATASET/DATASET_Tumor/DATASET_Tumor/unetr_pp_raw/unetr_pp_raw_data/Task03_tumor
export unetr_pp_raw_data_base=../DATASET/DATASET_Tumor/unetr_pp_raw
python unetr_pp/run/run_training.py 3d_fullres unetr_pp_trainer_tumor 3 0  --continue_training
