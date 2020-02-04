#!/bin/bash
NUM_CORES=$(nproc --all)
export MKL_NUM_THREADS=$NUM_CORES OMP_NUM_THREADS=$NUM_CORES
export CUDA_VISIBLE_DEVICES=0

# Investigating effect of the KL break
# for targetkl in 0.015 0.02 0.035 0.04 0.05 0.064 0.075 0.08 0.09 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
#   (sleep 0.3 && nohup python ppo_continuous_v.py \
#   --prod-mode True \
#   --wandb-project-name cleanrl.ppo \
#   --wandb-entity dosssman \
#   --target-kl $targetkl \
#   --total-timesteps 1000000
#   ) >& /dev/null &
# done

# Investigating effect of the "Small buffer" effect
# for episode_length in 1000 1500 2000 2500 3000 4000 5000 6000 7500 9000 10000; do
#   (sleep 0.3 && nohup python ppo_continuous_gae.py \
#   --prod-mode True \
#   --wandb-project-name cleanrl.ppo \
#   --wandb-entity dosssman \
#   --total-timesteps 1000000 \
#   --episode_length $episode_length
#   ) >& /dev/null &
# done

# Investifating on other envs
# for seed in {0..2}; do
#   # Hopper Bullet Env
#   (sleep 0.3 && nohup python ppo_continuous_gae_2.py \
#   --prod-mode True \
#   --wandb-project-name cleanrl.ppo_cont.HopperBulletEnv \
#   --wandb-entity dosssman \
#   --total-timesteps 1000000 \
#   --gym-id HopperBulletEnv-v0 \
#   --seed $seed
#   ) >& /dev/null &
#
#   (sleep 0.3 && nohup python ppo_continuous_v.py \
#   --prod-mode True \
#   --wandb-project-name cleanrl.ppo_cont.HopperBulletEnv \
#   --wandb-entity dosssman \
#   --total-timesteps 1000000 \
#   --gym-id HopperBulletEnv-v0 \
#   --seed $seed
#   ) >& /dev/null &
#
#   # Walker2D
#   (sleep 0.3 && nohup python ppo_continuous_gae_2.py \
#   --prod-mode True \
#   --wandb-project-name cleanrl.ppo_cont.Walker2DBulletEnv \
#   --wandb-entity dosssman \
#   --total-timesteps 1000000 \
#   --gym-id Walker2DBulletEnv-v0 \
#   --seed $seed
#   ) >& /dev/null &
#
#   (sleep 0.3 && nohup python ppo_continuous_v.py \
#   --prod-mode True \
#   --wandb-project-name cleanrl.ppo_cont.Walker2DBulletEnv \
#   --wandb-entity dosssman \
#   --total-timesteps 1000000 \
#   --gym-id Walker2DBulletEnv-v0 \
#   --seed $seed
#   ) >& /dev/null &
#
#   # HalfCheetah-v0
#   (sleep 0.3 && nohup python ppo_continuous_gae_2.py \
#   --prod-mode True \
#   --wandb-project-name cleanrl.ppo_cont.HalfCheetahBulletEnv \
#   --wandb-entity dosssman \
#   --total-timesteps 1000000 \
#   --gym-id HalfCheetahBulletEnv-v0 \
#   --seed $seeddone

#   ) >& /dev/null &
#
#   (sleep 0.3 && nohup python ppo_continuous_v.py \
#   --prod-mode True \
#   --wandb-project-name cleanrl.ppo_cont.HalfCheetahBulletEnv \
#   --wandb-entity dosssman \
#   --total-timesteps 1000000 \
#   --gym-id HalfCheetahBulletEnv-v0 \
#   --seed $seed
#   ) >& /dev/null &
#
#   # HumanoidBUllet Env
#   (sleep 0.3 && nohup python ppo_continuous_gae_2.py \
#   --prod-mode True \
#   --wandb-project-name cleanrl.ppo_cont.HumanoidBulletEnv \
#   --wandb-entity dosssman \
#   --total-timesteps 1000000 \
#   --gym-id HumanoidBulletEnv-v0 \
#   --seed $seed
#   ) >& /dev/null &
#
#   (sleep 0.3 && nohup python ppo_continuous_v.py \
#   --prod-mode True \
#   --wandb-project-name cleanrl.ppo_cont.HumanoidBulletEnv \
#   --wandb-entity dosssman \
#   --total-timesteps 1000000 \
#   --gym-id HumanoidBulletEnv-v0 \
#   --seed $seed
#   ) >& /dev/null &
# done

# Daddy look, No KL ...
# for seed in {0..2}; do
#     # Hopper Bullet Env
#     (sleep 0.3 && nohup python ppo_cont_gae_nokl.py \
#     --prod-mode True \
#     --wandb-project-name cleanrl.ppo_cont.HopperBulletEnv \
#     --wandb-entity dosssman \
#     --total-timesteps 1000000 \
#     --gym-id HopperBulletEnv-v0 \
#     --seed $seed
#     ) >& /dev/null &
#
#     # Walker2D
#     (sleep 0.3 && nohup python ppo_cont_gae_nokl.py \
#     --prod-mode True \
#     --wandb-project-name cleanrl.ppo_cont.Walker2DBulletEnv \
#     --wandb-entity dosssman \
#     --total-timesteps 1000000 \
#     --gym-id Walker2DBulletEnv-v0 \
#     --seed $seed
#     ) >& /dev/null &
#
#     # HalfCheetah-v0
#     (sleep 0.3 && nohup python ppo_cont_gae_nokl.py \
#     --prod-mode True \
#     --wandb-project-name cleanrl.ppo_cont.HalfCheetahBulletEnv \
#     --wandb-entity dosssman \
#     --total-timesteps 1000000 \
#     --gym-id HalfCheetahBulletEnv-v0 \
#     --seed $seed
#     ) >& /dev/null &
#
#     # HumanoidBUllet Env
#     (sleep 0.3 && nohup python ppo_cont_gae_nokl.py \
#     --prod-mode True \
#     --wandb-project-name cleanrl.ppo_cont.HumanoidBulletEnv \
#     --wandb-entity dosssman \
#     --total-timesteps 1000000 \
#     --gym-id HumanoidBulletEnv-v0 \
#     --seed $seed
#     ) >& /dev/null &
# done

# State Obs Normalization and no norm test
# for seed in {1..5}; do
  # # HopperBulletEnv
  # (sleep 0.3 && nohup python ppo_cont_gae_normenv.py \
  # --prod-mode True \
  # --wandb-project-name cleanrl.ppo_gae.norm_obs \
  # --wandb-entity dosssman \
  # --total-timesteps 1000000 \
  # --gym-id HopperBulletEnv-v0 \
  # --seed $seed \
  # --nokl
  # ) >& /dev/null &
  #
  # (sleep 0.3 && nohup python ppo_cont_gae_normenv.py \
  # --prod-mode True \
  # --wandb-project-name cleanrl.ppo_gae.norm_obs \
  # --wandb-entity dosssman \
  # --total-timesteps 1000000 \
  # --gym-id HopperBulletEnv-v0 \
  # --seed $seed \
  # --norm-obs \
  # --nokl
  # ) >& /dev/null &
  # # End HopperBulletEnv
  #
  # # Walker2D
  # (sleep 0.3 && nohup python ppo_cont_gae_normenv.py \
  # --prod-mode True \
  # --wandb-project-name cleanrl.ppo_gae.norm_obs \
  # --wandb-entity dosssman \
  # --total-timesteps 1000000 \
  # --gym-id Walker2DBulletEnv-v0 \
  # --seed $seed \
  # --nokl
  # ) >& /dev/null &
  #
  # (sleep 0.3 && nohup python ppo_cont_gae_normenv.py \
  # --prod-mode True \
  # --wandb-project-name cleanrl.ppo_gae.norm_obs \
  # --wandb-entity dosssman \
  # --total-timesteps 1000000 \
  # --gym-id Walker2DBulletEnv-v0 \
  # --seed $seed \
  # --norm-obs \
  # --nokl
  # ) >& /dev/null &
  # # End Walker2D
  #
  # # HalfCheetahBulletEnv
  # (sleep 0.3 && nohup python ppo_cont_gae_normenv.py \
  # --prod-mode True \
  # --wandb-project-name cleanrl.ppo_gae.norm_obs \
  # --wandb-entity dosssman \
  # --total-timesteps 1000000 \
  # --gym-id HalfCheetahBulletEnv-v0 \
  # --seed $seed \
  # --nokl
  # ) >& /dev/null &
  #
  # (sleep 0.3 && nohup python ppo_cont_gae_normenv.py \
  # --prod-mode True \
  # --wandb-project-name cleanrl.ppo_gae.norm_obs \
  # --wandb-entity dosssman \
  # --total-timesteps 1000000 \
  # --gym-id HalfCheetahBulletEnv-v0 \
  # --seed $seed \
  # --norm-obs \
  # --nokl
  # ) >& /dev/null &
  # # End HalfCheetahBulletEnv
  #
  # # HumanoidBulletEnv
  # (sleep 0.3 && nohup python ppo_cont_gae_normenv.py \
  # --prod-mode True \
  # --wandb-project-name cleanrl.ppo_gae.norm_obs \
  # --wandb-entity dosssman \
  # --total-timesteps 1000000 \
  # --gym-id HumanoidBulletEnv-v0 \
  # --seed $seed \
  # --nokl
  # ) >& /dev/null &
  #
  # (sleep 0.3 && nohup python ppo_cont_gae_normenv.py \
  # --prod-mode True \
  # --wandb-project-name cleanrl.ppo_gae.norm_obs \
  # --wandb-entity dosssman \
  # --total-timesteps 1000000 \
  # --gym-id HumanoidBulletEnv-v0 \
  # --seed $seed \
  # --norm-obs \
  # --nokl
  # ) >& /dev/null &
  # # End HumanoidBulletEnv
# done

# State Obs Normalization and no norm test with KL-based early stop version
# for seed in {1..5}; do
  # # HopperBulletEnv
  # (sleep 0.3 && nohup python ppo_cont_gae_normenv.py \
  # --prod-mode True \
  # --wandb-project-name cleanrl.ppo_gae.norm_obs_withkl \
  # --wandb-entity dosssman \
  # --total-timesteps 1000000 \
  # --gym-id HopperBulletEnv-v0 \
  # --seed $seed
  # ) >& /dev/null &
  #
  # (sleep 0.3 && nohup python ppo_cont_gae_normenv.py \
  # --prod-mode True \
  # --wandb-project-name cleanrl.ppo_gae.norm_obs_withkl \
  # --wandb-entity dosssman \
  # --total-timesteps 1000000 \
  # --gym-id HopperBulletEnv-v0 \
  # --seed $seed \
  # --norm-obs
  # ) >& /dev/null &
  # # End HopperBulletEnv

  # # Walker2D
  # (sleep 0.3 && nohup python ppo_cont_gae_normenv.py \
  # --prod-mode True \
  # --wandb-project-name cleanrl.ppo_gae.norm_obs_withkl \
  # --wandb-entity dosssman \
  # --total-timesteps 1000000 \
  # --gym-id Walker2DBulletEnv-v0 \
  # --seed $seed
  # ) >& /dev/null &
  #
  # (sleep 0.3 && nohup python ppo_cont_gae_normenv.py \
  # --prod-mode True \
  # --wandb-project-name cleanrl.ppo_gae.norm_obs_withkl \
  # --wandb-entity dosssman \
  # --total-timesteps 1000000 \
  # --gym-id Walker2DBulletEnv-v0 \
  # --seed $seed \
  # --norm-obs
  # ) >& /dev/null &
  # # End Walker2D

  # # HalfCheetahBulletEnv
  # (sleep 0.3 && nohup python ppo_cont_gae_normenv.py \
  # --prod-mode True \
  # --wandb-project-name cleanrl.ppo_gae.norm_obs_withkl \
  # --wandb-entity dosssman \
  # --total-timesteps 1000000 \
  # --gym-id HalfCheetahBulletEnv-v0 \
  # --seed $seed
  # ) >& /dev/null &
  #
  # (sleep 0.3 && nohup python ppo_cont_gae_normenv.py \
  # --prod-mode True \
  # --wandb-project-name cleanrl.ppo_gae.norm_obs_withkl \
  # --wandb-entity dosssman \
  # --total-timesteps 1000000 \
  # --gym-id HalfCheetahBulletEnv-v0 \
  # --seed $seed \
  # --norm-obs
  # ) >& /dev/null &
  # # End HalfCheetahBulletEnv

  # # HumanoidBulletEnv
  # (sleep 0.3 && nohup python ppo_cont_gae_normenv.py \
  # --prod-mode True \
  # --wandb-project-name cleanrl.ppo_gae.norm_obs_withkl \
  # --wandb-entity dosssman \
  # --total-timesteps 1000000 \
  # --gym-id HumanoidBulletEnv-v0 \
  # --seed $seed
  # ) >& /dev/null &
  #
  # (sleep 0.3 && nohup python ppo_cont_gae_normenv.py \
  # --prod-mode True \
  # --wandb-project-name cleanrl.ppo_gae.norm_obs_withkl \
  # --wandb-entity dosssman \
  # --total-timesteps 1000000 \
  # --gym-id HumanoidBulletEnv-v0 \
  # --seed $seed \
  # --norm-obs
  # ) >& /dev/null &
  # # End HumanoidBulletEnv
# done

# PPO Continuous and GAE with KL
# for targetkl in 0.015 0.02 0.027 0.035 0.04 0.05 0.064 0.075 0.08 0.09 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
#   # HopperBulletEnv
#   (sleep 0.3 && nohup python ppo_continuous_v.py \
#   --prod-mode True \
#   --wandb-project-name cleanrl.ppo_cont.kl_exp \
#   --wandb-entity dosssman \
#   --target-kl $targetkl \
#   --total-timesteps 1000000 \
#   --gym-id HopperBulletEnv-v0
#   ) >& /dev/null &
#
#   (sleep 0.3 && nohup python ppo_continuous_gae_2.py \
#   --prod-mode True \
#   --wandb-project-name cleanrl.ppo_cont.kl_exp \
#   --wandb-entity dosssman \
#   --target-kl $targetkl \
#   --total-timesteps 1000000 \
#   --gym-id HopperBulletEnv-v0
#   ) >& /dev/null &
# done

# KL vs NOKL for PPO with just V, GAE and respective NOKL versions
# for seed in 3 4 5; do
#   # Hopper Bullet Env With KL
#   (sleep 0.3 && nohup python ppo_continuous_gae_2.py \
#   --prod-mode True \
#   --wandb-project-name cleanrl.ppo_cont.kl_vs_nokl \
#   --wandb-entity dosssman \
#   --total-timesteps 1000000 \
#   --gym-id HopperBulletEnv-v0 \
#   --seed $seed
#   ) >& /dev/null &
#
#   (sleep 0.3 && nohup python ppo_continuous_v.py \
#   --prod-mode True \
#   --wandb-project-name cleanrl.ppo_cont.kl_vs_nokl \
#   --wandb-entity dosssman \
#   --total-timesteps 1000000 \
#   --gym-id HopperBulletEnv-v0 \
#   --seed $seed
#   ) >& /dev/null &
#
#   # Hopper Bullet Env Without KL const.
#   (sleep 0.3 && nohup python ppo_continuous_gae_2.py \
#   --prod-mode True \
#   --wandb-project-name cleanrl.ppo_cont.kl_vs_nokl \
#   --wandb-entity dosssman \
#   --total-timesteps 1000000 \
#   --gym-id HopperBulletEnv-v0 \
#   --seed $seed \
#   --nokl
#   ) >& /dev/null &
#
#   (sleep 0.3 && nohup python ppo_continuous_v.py \
#   --prod-mode True \
#   --wandb-project-name cleanrl.ppo_cont.kl_vs_nokl \
#   --wandb-entity dosssman \
#   --total-timesteps 1000000 \
#   --gym-id HopperBulletEnv-v0 \
#   --seed $seed \
#   --nok
#   ) >& /dev/null &
#
#   # HalfCheetah-v0 with and without KL
#   (sleep 0.3 && nohup python ppo_continuous_gae_2.py \
#   --prod-mode True \
#   --wandb-project-name cleanrl.ppo_cont.kl_vs_nokl \
#   --wandb-entity dosssman \
#   --total-timesteps 1000000 \
#   --gym-id HalfCheetahBulletEnv-v0 \
#   --seed $seed
#   ) >& /dev/null &
#
#   (sleep 0.3 && nohup python ppo_continuous_v.py \
#   --prod-mode True \
#   --wandb-project-name cleanrl.ppo_cont.kl_vs_nokl \
#   --wandb-entity dosssman \
#   --total-timesteps 1000000 \
#   --gym-id HalfCheetahBulletEnv-v0 \
#   --seed $seed
#   ) >& /dev/null &
#
#   (sleep 0.3 && nohup python ppo_continuous_gae_2.py \
#   --prod-mode True \
#   --wandb-project-name cleanrl.ppo_cont.kl_vs_nokl \
#   --wandb-entity dosssman \
#   --total-timesteps 1000000 \
#   --gym-id HalfCheetahBulletEnv-v0 \
#   --seed $seed \
#   --nokl
#   ) >& /dev/null &
#
#   (sleep 0.3 && nohup python ppo_continuous_v.py \
#   --prod-mode True \
#   --wandb-project-name cleanrl.ppo_cont.kl_vs_nokl \
#   --wandb-entity dosssman \
#   --total-timesteps 1000000 \
#   --gym-id HalfCheetahBulletEnv-v0 \
#   --seed $seed \
#   --nokl
#   ) >& /dev/null &
#
#   # Walker2D  with and without KL
#   (sleep 0.3 && nohup python ppo_continuous_gae_2.py \
#   --prod-mode True \
#   --wandb-project-name cleanrl.ppo_cont.kl_vs_nokl \
#   --wandb-entity dosssman \
#   --total-timesteps 1000000 \
#   --gym-id Walker2DBulletEnv-v0 \
#   --seed $seed
#   ) >& /dev/null &
#
#   (sleep 0.3 && nohup python ppo_continuous_v.py \
#   --prod-mode True \
#   --wandb-project-name cleanrl.ppo_cont.kl_vs_nokl \
#   --wandb-entity dosssman \
#   --total-timesteps 1000000 \
#   --gym-id Walker2DBulletEnv-v0 \
#   --seed $seed
#   ) >& /dev/null &
#
#   (sleep 0.3 && nohup python ppo_continuous_gae_2.py \
#   --prod-mode True \
#   --wandb-project-name cleanrl.ppo_cont.kl_vs_nokl \
#   --wandb-entity dosssman \
#   --total-timesteps 1000000 \
#   --gym-id Walker2DBulletEnv-v0 \
#   --seed $seed \
#   --nokl
#   ) >& /dev/null &
#
#   (sleep 0.3 && nohup python ppo_continuous_v.py \
#   --prod-mode True \
#   --wandb-project-name cleanrl.ppo_cont.kl_vs_nokl \
#   --wandb-entity dosssman \
#   --total-timesteps 1000000 \
#   --gym-id Walker2DBulletEnv-v0 \
#   --seed $seed \
#   --nokl
#   ) >& /dev/null &
#
#   # HumanoidBUllet Env
#   (sleep 0.3 && nohup python ppo_continuous_gae_2.py \
#   --prod-mode True \
#   --wandb-project-name cleanrl.ppo_cont.kl_vs_nokl \
#   --wandb-entity dosssman \
#   --total-timesteps 1000000 \
#   --gym-id HumanoidBulletEnv-v0 \
#   --seed $seed
#   ) >& /dev/null &
#
#   (sleep 0.3 && nohup python ppo_continuous_v.py \
#   --prod-mode True \
#   --wandb-project-name cleanrl.ppo_cont.kl_vs_nokl \
#   --wandb-entity dosssman \
#   --total-timesteps 1000000 \
#   --gym-id HumanoidBulletEnv-v0 \
#   --seed $seed
#   ) >& /dev/null &
#
#   (sleep 0.3 && nohup python ppo_continuous_gae_2.py \
#   --prod-mode True \
#   --wandb-project-name cleanrl.ppo_cont.kl_vs_nokl \
#   --wandb-entity dosssman \
#   --total-timesteps 1000000 \
#   --gym-id HumanoidBulletEnv-v0 \
#   --seed $seed \
#   --nokl
#   ) >& /dev/null &
#
#   (sleep 0.3 && nohup python ppo_continuous_v.py \
#   --prod-mode True \
#   --wandb-project-name cleanrl.ppo_cont.kl_vs_nokl \
#   --wandb-entity dosssman \
#   --total-timesteps 1000000 \
#   --gym-id HumanoidBulletEnv-v0 \
#   --seed $seed \
#   --nokl
#   ) >& /dev/null &
# done

# # LR Annealing experiment: Without KL Upper bound
# for seed in {1..3}; do
#   # HopperBulletEnv
#   # Baselines without Anneal
#   (sleep 0.3 && nohup python ppo_cont_gae_normenv.py \
#   --prod-mode True \
#   --wandb-project-name cleanrl.ppo_gae.anneal_lr \
#   --wandb-entity dosssman \
#   --total-timesteps 1000000 \
#   --gym-id HopperBulletEnv-v0 \
#   --seed $seed \
#   --nokl
#   ) >& /dev/null &
#
#   # With Anneal
#   (sleep 0.3 && nohup python ppo_cont_gae_normenv.py \
#   --prod-mode True \
#   --wandb-project-name cleanrl.ppo_gae.anneal_lr \
#   --wandb-entity dosssman \
#   --total-timesteps 1000000 \
#   --gym-id HopperBulletEnv-v0 \
#   --seed $seed \
#   --nokl \
#   --anneal-lr
#   ) >& /dev/null &
#   # End HopperBulletEnv
#
#   # Walker2D
#   (sleep 0.3 && nohup python ppo_cont_gae_normenv.py \
#   --prod-mode True \
#   --wandb-project-name cleanrl.ppo_gae.anneal_lr \
#   --wandb-entity dosssman \
#   --total-timesteps 1000000 \
#   --gym-id Walker2DBulletEnv-v0 \
#   --seed $seed \
#   --nokl
#   ) >& /dev/null &
#
#   (sleep 0.3 && nohup python ppo_cont_gae_normenv.py \
#   --prod-mode True \
#   --wandb-project-name cleanrl.ppo_gae.anneal_lr \
#   --wandb-entity dosssman \
#   --total-timesteps 1000000 \
#   --gym-id Walker2DBulletEnv-v0 \
#   --seed $seed \
#   --nokl \
#   --anneal-lr
#   ) >& /dev/null &
#   # End Walker2D
#
#   # HalfCheetahBulletEnv
#   (sleep 0.3 && nohup python ppo_cont_gae_normenv.py \
#   --prod-mode True \
#   --wandb-project-name cleanrl.ppo_gae.anneal_lr \
#   --wandb-entity dosssman \
#   --total-timesteps 1000000 \
#   --gym-id HalfCheetahBulletEnv-v0 \
#   --seed $seed \
#   --nokl
#   ) >& /dev/null &
#
#   (sleep 0.3 && nohup python ppo_cont_gae_normenv.py \
#   --prod-mode True \
#   --wandb-project-name cleanrl.ppo_gae.anneal_lr \
#   --wandb-entity dosssman \
#   --total-timesteps 1000000 \
#   --gym-id HalfCheetahBulletEnv-v0 \
#   --seed $seed \
#   --nokl \
#   --anneal-lr
#   ) >& /dev/null &
#   # End HalfCheetahBulletEnv
#
#   # HumanoidBulletEnv
#   (sleep 0.3 && nohup python ppo_cont_gae_normenv.py \
#   --prod-mode True \
#   --wandb-project-name cleanrl.ppo_gae.anneal_lr \
#   --wandb-entity dosssman \
#   --total-timesteps 1000000 \
#   --gym-id HumanoidBulletEnv-v0 \
#   --seed $seed \
#   --nokl
#   ) >& /dev/null &
#
#   (sleep 0.3 && nohup python ppo_cont_gae_normenv.py \
#   --prod-mode True \
#   --wandb-project-name cleanrl.ppo_gae.anneal_lr \
#   --wandb-entity dosssman \
#   --total-timesteps 1000000 \
#   --gym-id HumanoidBulletEnv-v0 \
#   --seed $seed \
#   --nokl \
#   --anneal-lr
#   ) >& /dev/null &
#   # End HumanoidBulletEnv
# done
# # End LR Annealing experiment: Without KL Upper bound

# LR Annealing experiment: With KL Upper bound
for seed in {1..3}; do
  # HopperBulletEnv
  # Baselines without Anneal
  (sleep 0.3 && nohup python ppo_cont_gae_normenv.py \
  --prod-mode True \
  --wandb-project-name cleanrl.ppo_gae.anneal_lr \
  --wandb-entity dosssman \
  --total-timesteps 1000000 \
  --gym-id HopperBulletEnv-v0 \
  --seed $seed
  ) >& /dev/null &

  # With Anneal
  (sleep 0.3 && nohup python ppo_cont_gae_normenv.py \
  --prod-mode True \
  --wandb-project-name cleanrl.ppo_gae.anneal_lr \
  --wandb-entity dosssman \
  --total-timesteps 1000000 \
  --gym-id HopperBulletEnv-v0 \
  --seed $seed \
  --anneal-lr
  ) >& /dev/null &
  # End HopperBulletEnv

  # Walker2D
  (sleep 0.3 && nohup python ppo_cont_gae_normenv.py \
  --prod-mode True \
  --wandb-project-name cleanrl.ppo_gae.anneal_lr \
  --wandb-entity dosssman \
  --total-timesteps 1000000 \
  --gym-id Walker2DBulletEnv-v0 \
  --seed $seed
  ) >& /dev/null &

  (sleep 0.3 && nohup python ppo_cont_gae_normenv.py \
  --prod-mode True \
  --wandb-project-name cleanrl.ppo_gae.anneal_lr \
  --wandb-entity dosssman \
  --total-timesteps 1000000 \
  --gym-id Walker2DBulletEnv-v0 \
  --seed $seed \
  --anneal-lr
  ) >& /dev/null &
  # End Walker2D

  # HalfCheetahBulletEnv
  (sleep 0.3 && nohup python ppo_cont_gae_normenv.py \
  --prod-mode True \
  --wandb-project-name cleanrl.ppo_gae.anneal_lr \
  --wandb-entity dosssman \
  --total-timesteps 1000000 \
  --gym-id HalfCheetahBulletEnv-v0 \
  --seed $seed
  ) >& /dev/null &

  (sleep 0.3 && nohup python ppo_cont_gae_normenv.py \
  --prod-mode True \
  --wandb-project-name cleanrl.ppo_gae.anneal_lr \
  --wandb-entity dosssman \
  --total-timesteps 1000000 \
  --gym-id HalfCheetahBulletEnv-v0 \
  --seed $seed \
  --anneal-lr
  ) >& /dev/null &
  # End HalfCheetahBulletEnv

  # HumanoidBulletEnv
  (sleep 0.3 && nohup python ppo_cont_gae_normenv.py \
  --prod-mode True \
  --wandb-project-name cleanrl.ppo_gae.anneal_lr \
  --wandb-entity dosssman \
  --total-timesteps 1000000 \
  --gym-id HumanoidBulletEnv-v0 \
  --seed $seed
  ) >& /dev/null &

  (sleep 0.3 && nohup python ppo_cont_gae_normenv.py \
  --prod-mode True \
  --wandb-project-name cleanrl.ppo_gae.anneal_lr \
  --wandb-entity dosssman \
  --total-timesteps 1000000 \
  --gym-id HumanoidBulletEnv-v0 \
  --seed $seed \
  --anneal-lr
  ) >& /dev/null &
  # End HumanoidBulletEnv
done
# End LR Annealing experiment: With KL Upper bound
