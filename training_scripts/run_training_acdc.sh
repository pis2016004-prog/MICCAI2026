#!/bin/sh
cd /drive/Brats2017/MICCAI_FINAL
DATASET_PATH=../DATASET/DATASET_Acdc
# force mamba-ssm to use reference (PyTorch) kernels, not the CUDA extension
export CAUSAL_CONV1D_FORCE_FALLBACK=1
export MAMBA_FORCE_FALLBACK=1

export PYTHONPATH=./
export RESULTS_FOLDER=./acdc_MICCAIFGA
export unetr_pp_preprocessed=../DATASET/DATASET_Acdc/unetr_pp_raw/unetr_pp_raw_data/Task01_ACDC
#"$DATASET_PATH"/unetr_pp_raw/unetr_pp_raw_data/Task01_ACDC
export unetr_pp_raw_data_base=../DATASET/DATASET_Acdc/unetr_pp_raw

python unetr_pp/run/run_training.py 3d_fullres unetr_pp_trainer_acdc 1 0  --continue_training 

