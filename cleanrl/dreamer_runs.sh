#!/bin/bash
NUM_CORES=$(nproc --all)
export MKL_NUM_THREADS=$NUM_CORES OMP_NUM_THREADS=$NUM_CORES

###############################################
# region: Atari Pong                          #

    # region: Baseline experiments
    # ## Seed 1
    export CUDA_VISIBLE_DEVICES=0
    (sleep 1s && python dreamer_atari.py \
        --track --capture-video \
        --env-id "PongNoFrameskip-v4" \
        --env-grayscale False \
        --total-timesteps 2500000 \
        --batch-size 50 --batch-length 50 \
        --train-every 16 \
        --exp-name "dreamer_B_50_T_50_trnev_16" \
        --seed 1 \
    ) & # >& /dev/null &
    # endregion: Baseline experiments

# endregion: Atari Pong                       #
###############################################
