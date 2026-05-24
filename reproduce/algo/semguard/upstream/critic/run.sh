#!/bin/bash
PORT=$1
GPUID=$2
export CUDA_VISIBLE_DEVICES=$GPUID


IFS=',' read -ra elements <<< "$GPUID"
gpucount=${#elements[@]}
echo "The number of GPUs: $gpucount"


accelerate launch --num_processes $gpucount  --main_process_port $PORT --config_file ./acc_config.yaml ./run.py
