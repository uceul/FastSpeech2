#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2

pythonCMD=/home/mbehr/anaconda3/envs/fs/bin/python
$pythonCMD synthesize.py --restore_step 900000 --mode stdin -p config/LJSpeech/preprocess.yaml -m config/LJSpeech/model.yaml -t config/LJSpeech/train.yaml
