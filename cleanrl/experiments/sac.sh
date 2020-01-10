#!/bin/bash
NUM_CORES=$(nproc --all)
export MKL_NUM_THREADS=$NUM_CORES OMP_NUM_THREADS=$NUM_CORES
export CUDA_VISIBLE_DEVICES=1

for seed in 1
do
    # Continuous soft actor critic with entropy autotuning and without
    # (sleep 0.3 && nohup python sac_continuous.py \
    # --seed $seed \
    # --gym-id HopperBulletEnv-v0 \
    # --total-timesteps 1000000 \
    # --episode-length 1000 \
    # --wandb-project-name cleanrl.sac \
    # --wandb-entity dosssman \
    # --autotune \
    # --prod-mode True
    # ) >& /dev/null &
    #
    # (sleep 0.3 && nohup python sac_continuous.py \
    # --seed $seed \
    # --gym-id HopperBulletEnv-v0 \
    # --total-timesteps 1000000 \
    # --episode-length 1000 \
    # --wandb-project-name cleanrl.sac \
    # --wandb-entity dosssman \
    # --prod-mode True
    # ) >& /dev/null &
    #
    # # Discrete Action Space
    # (sleep 0.3 && nohup python sac_disc.py \
    # --seed $seed \
    # --gym-id CartPole-v0 \
    # --total-timesteps 1000000 \
    # --episode-length 1000 \
    # --wandb-project-name cleanrl.sac \
    # --wandb-entity dosssman \
    # --prod-mode True
    # ) >& /dev/null &
    #
    # (sleep 0.3 && nohup python sac_disc.py \
    # --seed $seed \
    # --gym-id CartPole-v1 \
    # --total-timesteps 1000000 \
    # --episode-length 1000 \
    # --wandb-project-name cleanrl.sac \
    # --wandb-entity dosssman \
    # --prod-mode True
    # ) >& /dev/null &
    #
    # (sleep 0.3 && nohup python sac_disc.py \
    # --seed $seed \
    # --gym-id LunarLander-v2 \
    # --total-timesteps 1000000 \
    # --episode-length 1000 \
    # --wandb-project-name cleanrl.sac \
    # --wandb-entity dosssman \
    # --prod-mode True
    # ) >& /dev/null &
    #
    # (sleep 0.3 && nohup python sac_disc.py \
    # --seed $seed \
    # --gym-id MountainCar-v2 \
    # --total-timesteps 1000000 \
    # --episode-length 1000 \
    # --wandb-project-name cleanrl.sac \
    # --wandb-entity dosssman \
    # --prod-mode True
    # ) >& /dev/null &

    # SAC with V Target
    (sleep 0.3 && nohup python sac_continuous_vtarg.py \
    --seed $seed \
    --gym-id HopperBulletEnv-v0 \
    --total-timesteps 1000000 \
    --episode-length 1000 \
    --wandb-project-name cleanrl.sac \
    --wandb-entity dosssman \
    --prod-mode True
    ) >& /dev/null &
done
