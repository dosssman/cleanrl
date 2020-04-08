#!/bin/bash
NUM_CORES=$(nproc --all)
export MKL_NUM_THREADS=$NUM_CORES OMP_NUM_THREADS=$NUM_CORES

# Simple benchmark over a few envs of PyBullet
# # for gym_id in HopperBulletEnv-v0 HalfCheetahBulletEnv-v0; do
# # for gym_id in AntBulletEnv-v0 HumanoidBulletEnv-v0; do
# for gym_id in Walker2DBulletEnv-v0; do
#   for seed in {1..3}; do
#     (sleep 0.3 && nohup python td3.py \
#     --prod-mode \
#     --wandb-project-name cleanrl.td3 \
#     --wandb-entity dosssman \
#     --gym-id $gym_id \
#     --seed $seed \
#     --eval
#     ) >& /dev/null &
#   done
# done

# Experimenting with different type of noise. Also have simple DDPG baseline
# export CUDA_VISIBLE_DEVICES=0
# # for gym_id in HopperBulletEnv-v0; do
# # for gym_id in AntBulletEnv-v0; do
# # for gym_id in HalfCheetahBulletEnv-v0; do
# # for gym_id in HumanoidBulletEnv-v0; do
# for gym_id in Walker2DBulletEnv-v0; do
#   for seed in {1..2}; do
#     # Default Noise: normal
#     (sleep 0.3 && nohup python td3.py \
#       --prod-mode \
#       --wandb-project-name cleanrl.td3.noise_types \
#       --wandb-entity dosssman \
#       --gym-id $gym_id \
#       --seed $seed \
#       --eval
#     ) >& /dev/null &
#
#     # OU Noise
#     (sleep 0.3 && nohup python td3.py \
#       --prod-mode \
#       --wandb-project-name cleanrl.td3.noise_types \
#       --wandb-entity dosssman \
#       --gym-id $gym_id \
#       --seed $seed \
#       --eval \
#       --noise-type ou
#     ) >& /dev/null &
#
#     # Adaptime Parameter Space Noise Epxloration
#     (sleep 0.3 && nohup python td3.py \
#       --prod-mode \
#       --wandb-project-name cleanrl.td3.noise_types \
#       --wandb-entity dosssman \
#       --gym-id $gym_id \
#       --seed $seed \
#       --eval \
#       --noise-type adapt-param
#     ) >& /dev/null &
#   done
# done
# export CUDA_VISIBLE_DEVICES=

# Investigating the effect of update delays
# export CUDA_VISIBLE_DEVICES=0
# # for gym_id in HopperBulletEnv-v0; do
# # for gym_id in HumanoidBulletEnv-v0; do
# # for gym_id in HalfCheetahBulletEnv-v0; do
# for gym_id in Walker2DBulletEnv-v0; do
# # for gym_id in Humanoid-v2; do
#   for seed in {1..2}; do
#     ## Baseline with 2 as update delay
#     (sleep 0.3 && nohup python td3.py \
#       --gym-id $gym_id \
#       --seed $seed \
#       --prod-mode \
#       --wandb-project-name cleanrl.td3.update_delay_exp \
#       --wandb-entity dosssman
#     ) >& /dev/null &
#
#     ## Different update delays
#     for update_delay in 3 4 5 7 10; do
#       (sleep 0.3 && nohup python td3.py \
#         --gym-id $gym_id \
#         --seed $seed \
#         --update-delay $update_delay \
#         --prod-mode \
#         --wandb-project-name cleanrl.td3.update_delay_exp \
#         --wandb-entity dosssman
#       ) >& /dev/null &
#     done
#   done
# done
# export CUDA_VISIBLE_DEVICES=
