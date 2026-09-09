#!/bin/bash
#
# TERL (hybrid_xgb): AWR ensemble rebuilds + on-policy leaf updates.
#
# --xgb_eta is XGBoost shrinkage and must NOT be left to --learning_rate:
# that is the neural-net step size (3e-4 in defaults.yaml), and at eta=3e-4 a
# 100-round ensemble reproduces only ~3% of its regression target, so the policy
# mean stays near zero and the agent acts purely through exploration noise.
#
#   MountainCarContinuous-v0
#     gamma=0.9999          Only real reward is +100 at the goal ~100 steps out;
#                           at 0.99 it discounts away and the -0.1a^2 action cost
#                           dominates.
#     exploration_rho=0.9   Independent noise averages out over a trajectory.
#                           Measured over 5 seeds: 0/5 solved at rho=0, 5/5 at
#                           rho=0.8-0.9.  Avoid rho>=0.99.
#
#   Pendulum-v1
#     max_depth=8           Depth 5 leaves the actor barely state-dependent
#                           (policy mean varying by 0.33 over an action range of
#                           4).  Depth 8 raised that and improved return from
#                           -972 to -627.  Depth 10 was worse (-818).
#     awr_target_ess=0.1    A broad AWR fit averages good and bad actions
#                           together and produces a timid policy; sharpening it
#                           moved return from -972 to -857 on its own.
#     carry_variance=True   Rebuilds happen every ~2 iterations here, and each
#                           one overwrote sigma^2 with fresh AWR residuals,
#                           discarding what the policy gradient had learned -
#                           visible as std oscillating with no trend.  Carrying
#                           it across rebuilds gave -516 (3 seeds: -516, -433,
#                           -551) and removed the late-training drift.
#     var_update=log        The additive variance step is scaled by
#                           (var_max - var_min), i.e. by the action range.  Once
#                           sigma^2 anneals below that, the step is comparable to
#                           sigma^2 itself (measured: 0.12 against a variance of
#                           0.11-0.30), so the variance can only oscillate.
#                           Stepping log(sigma^2) makes the step a fixed fraction
#                           of the current variance: -434 vs -516 at 60k, and
#                           -467 at 150k (3 seeds: -434, -467, -499).
#     exploration_rho=0     Dense reward, no sustained-push structure; measured
#                           neutral here.
#
#   LunarLanderContinuous
#     ent_coef=0.01         sigma^2 is the residual spread of the AWR fit, so as
#                           the policy becomes deterministic the residuals shrink
#                           and sigma^2 shrinks with them - a self-reinforcing
#                           collapse.  Measured: std falling monotonically
#                           0.28 -> 0.147 while return peaked and then declined.
#                           An entropy bonus resists it.  Seed-dependent: seed 0
#                           went -37 -> -2 at 160k, seed 1 still declined late
#                           (-61).  ent_coef=0.03 was worse than 0.01.
#                           var_mode=uncertainty holds std stable (0.39 -> 0.42,
#                           confirming the mechanism) but did not improve return.
#
# ppo_lr is a FRACTION OF THE ACTION RANGE per micro-update (leaf steps are
# Adam-normalised), not an SGD learning rate.

if [ -z "${1:-}" ]; then
    echo "Usage: $0 <env_name> [seed]"
    exit 1
fi

ENV_NAME=$1
SEED=${2:-}

COMMON="--algo_type=hybrid_xgb \
    --env_type=gym \
    --env_name=$ENV_NAME \
    --use_ppo_clip \
    --device=cpu \
    --num_envs=4 \
    --wrapper=None \
    --xgb_eta=0.05 \
    --ppo_lr=0.03 \
    --n_estimators=100 \
    --n_steps=256 \
    --obs_dependent_std=True"

if [ "$ENV_NAME" == "MountainCarContinuous-v0" ]; then
    COMMAND="python scripts/train.py $COMMON \
        --total_n_steps=1000000 \
        --gamma=0.9999 \
        --gae_lambda=0.95 \
        --exploration_rho=0.9 \
        --max_depth=5 \
        --awr_update_freq=4000 \
        --awr_buffer_size=20000 \
        --awr_target_ess=0.3"

# elif [ "$ENV_NAME" == "Pendulum-v1" ]; then
#     COMMAND="python scripts/train.py $COMMON \
#         --total_n_steps=1000000 \
#         --gamma=0.999 \
#         --gae_lambda=0.9 \
#         --exploration_rho=0.0 \
#         --max_depth=8 \
#         --awr_update_freq=2000 \
#         --awr_buffer_size=4000 \
#         --awr_target_ess=0.1 \
#         --carry_variance=True \
#         --var_update=log"

elif [ "$ENV_NAME" == "Pendulum-v1" ]; then
    COMMAND="python scripts/train.py --algo_type=hybrid_xgb \
        --env_type=gym \
        --env_name=$ENV_NAME \
        --device=cpu \
        --use_ppo_clip \
        --num_envs=4 \
        --wrapper=None \
        --obs_dependent_std=True \
        --gamma=0.999 \
        --gae_lambda=0.8 \
        --ppo_lr=0.07386839765272894 \
        --critic_lr=0.009417400570710336 \
        --xgb_eta=0.08200051762683408 \
        --awr_target_ess=0.07445209475266414 \
        --n_estimators=200 \
        --max_depth=5 \
        --awr_update_freq=4000 \
        --awr_buffer_size=20000 \
        --n_steps=256 \
        --n_ppo_epochs=1 \
        --ent_coef=2.545288293007366e-07"

# elif [ "$ENV_NAME" == "LunarLanderContinuous-v2" ]; then
#     COMMAND="python scripts/train.py $COMMON \
#         --total_n_steps=1000000 \
#         --gamma=0.999 \
#         --gae_lambda=0.95 \
#         --exploration_rho=0.0 \
#         --max_depth=10 \
#         --awr_target_ess=0.3 \
#         --awr_buffer_size=20000 \
#         --awr_update_freq=2000 \
#         --ent_coef=0.01"

# #from OLD
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
    exit 1
fi

if [ -n "$SEED" ]; then
    COMMAND+=" --seed=$SEED"
fi

echo "$COMMAND"
eval "$COMMAND"