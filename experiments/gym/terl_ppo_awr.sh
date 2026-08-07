##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl_sb3/license.html
#
##############################################################################
#!/bin/bash

# Check if environment name and seed are provided
if [ -z "$1" ]; then
    echo "Usage: $0 <env_name> [seed]"
    exit 1
fi

ENVS=(
    'LunarLanderContinuous-v2'
    'Pendulum-v1'
    'MountainCarContinuous-v0'
)


ENV_NAME=$1
SEED=""
if [ "$2" ]; then
    SEED=$2
fi


if [ "$ENV_NAME" == "MountainCarContinuous-v0" ]; then
    COMMAND="python scripts/train.py --algo_type=hybrid_xgb \
        --ppo_lr=0.02 \
        --awr_beta=0.05 \
        --max_depth=10 \
        --batch_size=64 \
        --buffer_size=50000 \
        --device=cuda \
        --ent_coef=0 \
        --env_name=$ENV_NAME \
        --env_type=gym \
        --gamma=0.9999 \
        --gradient_steps=150 \
        --grow_policy=oblivious \
        --max_policy_grad_norm=150 \
        --num_envs=1 \
        --policy_lr=0.0003938480943840346 \
        --reward_mode=gae \
        --total_n_steps=1000000 \
        --train_freq=2000 \
        --value_lr=0.08317628198321741 \
        --wrapper=None"

# elif [ "$ENV_NAME" == "Pendulum-v1" ]; then
#     COMMAND="python scripts/train.py --algo_type=hybrid_xgb \
#         --ppo_lr=0.001 \
#         --awr_beta=0.005 \
#         --max_depth=10 \
#         --device=cuda \
#         --env_name=$ENV_NAME \
#         --env_type=gym \
#         --gae_lambda=0.9 \
#         --gamma=0.999 \
#         --gradient_steps=50 \
#         --grow_policy=oblivious \
#         --log_std_init=-2 \
#         --log_std_lr=0.0005318970196570411 \
#         --normalize_advantage=True \
#         --num_envs=1 \
#         --policy_lr=0.00378772024172414 \
#         --total_n_steps=1000000 \
#         --train_freq=1000 \
#         --value_lr=0.07353926885096224 \
#         --wrapper=None \
#         --wrapper_kwargs=None"

#CLAUDE
# elif [ "$ENV_NAME" == "Pendulum-v1" ]; then
#     COMMAND="python scripts/train.py --algo_type=hybrid_xgb \
#         --ppo_lr=0.03 \
#         --awr_beta=0.005 \
#         --awr_update_freq=2000 \
#         --awr_buffer_size=4000 \
#         --max_depth=10 \
#         --device=cuda \
#         --env_name=$ENV_NAME \
#         --env_type=gym \
#         --gae_lambda=0.9 \
#         --gamma=0.999 \
#         --gradient_steps=50 \
#         --grow_policy=oblivious \
#         --log_std_init=-2 \
#         --log_std_lr=0.0005318970196570411 \
#         --normalize_advantage=True \
#         --num_envs=1 \
#         --policy_lr=0.00378772024172414 \
#         --total_n_steps=1000000 \
#         --train_freq=1000 \
#         --value_lr=0.07353926885096224 \
#         --wrapper=None \
#         --wrapper_kwargs=None"

#sweep.py
elif [ "$ENV_NAME" == "Pendulum-v1" ]; then
    COMMAND="python scripts/train.py --algo_type=hybrid_xgb \
        --env_name=$ENV_NAME \
        --env_type=gym \
        --ppo_lr=0.001 \
        --awr_beta=0.005 \
        --max_depth=10 \
        --total_n_steps=1000000 \
        --awr_update_freq=50000 \
        --awr_buffer_size=50000"


# elif [ "$ENV_NAME" == "LunarLanderContinuous-v2" ]; then
#     COMMAND="python scripts/train.py --algo_type=hybrid_xgb \
#         --ppo_lr=0.02 \
#         --awr_beta=0.01 \
#         --max_depth=10 \
#         --batch_size=1024 \
#         --buffer_size=50000 \
#         --device=cuda \
#         --ent_coef=0 \
#         --env_name=$ENV_NAME \
#         --env_type=gym \
#         --gamma=0.9999
#         --gradient_steps=150 \
#         --num_envs=1 \
#         --policy_lr=0.05073960309815198 \
#         --reward_mode=gae \
#         --total_n_steps=1500000 \
#         --train_freq=2000 \
#         --value_lr=0.10526702677422366 \
#         --wrapper=normalize"

#sweep.py
elif [ "$ENV_NAME" == "LunarLanderContinuous-v2" ]; then
    COMMAND="python scripts/train.py --algo_type=hybrid_xgb \
        --env_name=$ENV_NAME \
        --env_type=gym \
        --ppo_lr=0.02 \
        --awr_beta=0.01 \
        --max_depth=10 \
        --total_n_steps=1000000 \
        --awr_update_freq=50000 \
        --awr_buffer_size=50000"


else
    echo "Unknown environment name: $ENV_NAME"
    echo "Valid environments are:"
    for ENV in "${ENVS[@]}"; do
        echo "  $ENV"
    done
    exit 1
fi

# Add the seed argument if provided
if [ ! -z "$SEED" ]; then
    COMMAND+=" --seed=$SEED"
fi

# Run the command
eval $COMMAND
