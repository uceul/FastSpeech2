#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=5

python train.py -p experiments/halabi_default_01/config/preprocess.yaml -m experiments/halabi_default_01/config/model.yaml -t experiments/halabi_default_01/config/train.yaml
