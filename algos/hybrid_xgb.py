import numpy as np
import json
import xgboost as xgb
import torch as th
import pandas as pd
import time
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.logger import configure
from collections import deque

# def get_leafs(node):
#     """Recursively extract leaf weights from XGBoost JSON dump."""
#     leaves = {}
#     if 'leaf' in node:
#         leaves[node['nodeid']] = {"leaf_weight": node['leaf']}
#     elif 'children' in node:
#         for child in node['children']:
#             leaves.update(get_leafs(child))
#     return leaves

def get_leafs(tree: dict):
    """Get all leafs of a tree.

    Parameters
    ----------
    tree : dict
        Tree.

    Returns
    -------
    leafs : dict
        Dist of leafs.
    """

    leafs = {}
    stack = [tree]
    while stack:
        node = stack.pop()
        try:
            stack.append(node['children'][0])
            stack.append(node['children'][1])
        except:
            leafs[node['nodeid']] = node #if node['nodeid'] != 0 else return leafs  

    return leafs

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

class XGBoostTreeEngine:
    """Core math engine: 100% Vectorized NumPy logic for real-time Tree RL."""
    def __init__(self, action_dim, max_depth, n_estimators, ppo_lr, awr_beta, action_low, action_high, clip_ratio=0.2):
        self.action_dim = action_dim
        self.n_estimators = n_estimators
        self.ppo_lr = ppo_lr
        self.awr_beta = awr_beta
        self.clip_ratio = clip_ratio

        self.action_low = action_low
        self.action_high = action_high
        
        self.actor_model = None
        self.critic_model = None
        self.fast_leaf_array = None  # NEW: Dense NumPy memory bank

        self.actor_leaves_per_tree = None
        self.critic_leaves_per_tree = None
        
        #self.actor_params = {'objective': 'multi:softprob', 'num_class': action_dim, 'max_depth': max_depth, 'tree_method': 'hist', 'base_score': 0, 'eta': 0.3}
        self.actor_params = {'objective': 'reg:squarederror', 'max_depth': max_depth, 'tree_method': 'hist', 'base_score': 0, 'eta': 0.01}
        self.critic_params = {'objective': 'reg:squarederror', 'max_depth': max_depth, 'tree_method': 'hist', 'base_score': 0, 'eta': 0.01}
        self.actor_lr = self.actor_params['eta']
        self.critic_lr = self.critic_params['eta']

        # NEW: Log standard deviation for Gaussian exploration
        self.log_std = np.full(self.action_dim, 0.0, dtype=np.float32)

    def _prepare_dmatrix(self, X, label=None, weight=None):
        enable_cat = False
        
        if X.dtype == object or X.dtype.kind in {'S', 'U'}:
            if X.ndim == 1: X = X.reshape(-1, 1)
            
            if isinstance(X.flat[0], bytes):
                X = np.char.decode(X.astype('S'), 'utf-8')
                
            # --- NEW: Try to cast to standard floats first! ---
            try:
                # If this works, it's purely numeric data (like MountainCar)
                X = X.astype(np.float32)
            except ValueError:
                # If it throws a ValueError, it contains strings (like MiniGrid)
                df = pd.DataFrame(X)
                for col in df.columns:
                    df[col] = df[col].astype('category')
                X = df
                enable_cat = True
            # --------------------------------------------------

        # If weight logic is needed later, uncomment it here, otherwise keep it clean
        return xgb.DMatrix(X, label=label, enable_categorical=enable_cat)

    def train_macro_awr(self, states, actions, advantages, returns):
        returns = returns.reshape((len(returns),1))
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        weights = np.clip(np.exp(advantages / self.awr_beta), 0, 20.0)
        
        dtrain_act = self._prepare_dmatrix(states, label=actions, weight=weights)
        self.actor_model = xgb.train(self.actor_params, dtrain_act, num_boost_round=self.n_estimators)
        
        dtrain_crit = self._prepare_dmatrix(states, label=returns)
        self.critic_model = xgb.train(self.critic_params, dtrain_crit, num_boost_round=self.n_estimators)

        #from TEUCB
        #train critic
        leaf_scores_crit = self.critic_model.get_dump(with_stats=True, dump_format='json')
        json_trees_crit = [json.loads(leaf_scores_crit[i]) for i in range(self.n_estimators)]
        
        self.critic_leaves_per_tree = []
        for tree in json_trees_crit:
            self.critic_leaves_per_tree.append(get_leafs(tree))
            
        # Now do the residual math
        X = self._prepare_dmatrix(states)
        returns_flat = returns.flatten()
        
        individual_preds_crit = self.critic_model.predict(X, pred_leaf=True)
        leaf_of_data_crit = np.array(individual_preds_crit)
        
        crit_residuals_list = []
        
        for i in range(self.n_estimators):
            if i == 0:
                # For the first tree, the prediction is just the base score!
                pred_before_tree = np.full(len(returns_flat), self.critic_params['base_score'])
            else:
                pred_before_tree = self.critic_model.predict(X, iteration_range=(0, i+1)).flatten()
                
            # Target minus prediction BEFORE this tree
            residual = returns_flat - pred_before_tree
            crit_residuals_list.append(residual)
            
        crit_residuals_array = np.column_stack(crit_residuals_list)

        # Calculate your custom Pandas Means and Variances!
        for i in range(self.n_estimators):
            df = pd.DataFrame({'leaf': leaf_of_data_crit[:, i], 'residual': crit_residuals_array[:, i]})
            group = df.groupby(by='leaf').agg(['count', 'mean', 'var'])

            for row_idx, row in group.iterrows():
                self.critic_leaves_per_tree[i][row_idx]["row_idx"] = row_idx
                self.critic_leaves_per_tree[i][row_idx]["leaf_count"] = row['residual']['count']
                
                # Trust XGBoost's structural weight! 
                native_weight = self.critic_leaves_per_tree[i][row_idx].get("leaf_weight", 0.0)
                self.critic_leaves_per_tree[i][row_idx]["leaf_mean"] = native_weight
                
                # Keep YOUR custom variance for exploration!
                var = row['residual']['var']
                if pd.isna(var): var = 0.0
                self.critic_leaves_per_tree[i][row_idx]["leaf_variance"] = var * self.critic_lr**2


        #train actor
        # --- 1. CONTINUOUS ACTOR LEAF EXTRACTION ---
        total_actor_trees = self.n_estimators * self.action_dim
        leaf_scores = self.actor_model.get_dump(with_stats=True, dump_format='json')
        
        json_trees = [json.loads(leaf_scores[i]) for i in range(total_actor_trees)]
        
        self.actor_leaves_per_tree = []
        for idx, tree in enumerate(json_trees):
            self.actor_leaves_per_tree.append(get_leafs(tree))

        actions_flat = actions.astype(np.float32).reshape(-1, self.action_dim)
        
        individual_preds_act = self.actor_model.predict(X, pred_leaf=True)
        leaf_of_data_act = np.array(individual_preds_act)
        
        act_residuals_list = []
        
        for i in range(total_actor_trees):
            iteration = i // self.action_dim  
            dim_idx = i % self.action_dim   
            
            if iteration == 0:
                # Tree 0 predicts based on the base_score
                pred_before_tree = np.full(len(actions_flat), self.actor_params['base_score'])
            else:
                # Predict strictly BEFORE the current boosting round
                pred = self.actor_model.predict(X, iteration_range=(0, iteration+1)).reshape(-1, self.action_dim)
                pred_before_tree = pred[:, dim_idx]
                
            # The residual the tree was actually trying to fit!
            residual = actions_flat[:, dim_idx] - pred_before_tree
            act_residuals_list.append(residual)
            
        act_residuals_array = np.column_stack(act_residuals_list)

        for i in range(total_actor_trees):
            df = pd.DataFrame({'leaf': leaf_of_data_act[:, i], 'residual': act_residuals_array[:, i]})
            group = df.groupby(by='leaf').agg(['count', 'mean', 'var'])

            for row_idx, row in group.iterrows():
                self.actor_leaves_per_tree[i][row_idx]["row_idx"] = row_idx
                self.actor_leaves_per_tree[i][row_idx]["leaf_count"] = row['residual']['count']
                
                # YOUR CUSTOM MATH!
                self.actor_leaves_per_tree[i][row_idx]["leaf_mean"] = row['residual']['mean'] * self.actor_lr
                
                var = row['residual']['var']
                if pd.isna(var): var = 0.0
                self.actor_leaves_per_tree[i][row_idx]["leaf_variance"] = var * self.actor_lr**2



        # # Extract leaves into a Lightning-Fast NumPy Matrix
        # json_trees = [json.loads(t) for t in self.actor_model.get_dump(dump_format='json')]
        # leaves_per_tree = [get_leafs(t) for t in json_trees]
        
        # total_trees = self.n_estimators * self.action_dim
        # max_leaf_id = max([max(d.keys()) for d in leaves_per_tree if d] + [0])
        
        # # Build a dense array: Shape (total_trees, max_leaf_id)
        # self.fast_leaf_array = np.zeros((total_trees, max_leaf_id + 1))
        # for tree_idx, tree_dict in enumerate(leaves_per_tree):
        #     for leaf_id, leaf_data in tree_dict.items():
        #         self.fast_leaf_array[tree_idx, leaf_id] = leaf_data["leaf_weight"]

    # def _predict_logits_batch(self, leaf_assignments):
    #     """Instantaneous Inference: No Python Loops."""
    #     batch_size = leaf_assignments.shape[0]
    #     tree_indices = np.arange(self.n_estimators * self.action_dim)
        
    #     # NumPy Advanced Indexing: Fetches all leaf weights simultaneously
    #     weights = self.fast_leaf_array[tree_indices, leaf_assignments]
        
    #     # Reshape to (batch, estimators, classes) and sum
    #     return weights.reshape(batch_size, self.n_estimators, self.action_dim).sum(axis=1)

    def train_micro_ppo(self, states, actions, old_log_probs, advantages):
        if self.actor_model is None: return
        
        # Normalize advantages to keep tree gradients stable
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 1. Get the leaf assignments. Shape: (batch_size, total_trees)
        leaf_assignments = np.array(self.actor_model.predict(self._prepare_dmatrix(states), pred_leaf=True)).astype(int)
        
        batch_size = len(states)
        total_trees = self.n_estimators * self.action_dim
        
        # 2. Re-calculate the current means (mu)
        mu = np.zeros((batch_size, self.action_dim))
        
        for b in range(batch_size):
            for i in range(total_trees):
                leaf_id = leaf_assignments[b, i]
                dim_idx = i % self.action_dim
                
                leaf_data = self.actor_leaves_per_tree[i][leaf_id]
                
                # Get Mean
                leaf_val = leaf_data.get("leaf_mean", leaf_data.get("leaf_weight", 0.0))
                mu[b, dim_idx] += leaf_val
                
        mu += self.actor_params['base_score']
        
        # 🟢 THE FIX: Use the global, trainable Variance!
        # (This entirely replaces your old leaf_variance loops)
        var_sum = np.exp(self.log_std) ** 2
        
        # 3. Calculate Gaussian Gradients for Mu
        actions = actions.astype(np.float32).reshape(-1, self.action_dim)
        advantages = advantages.reshape(-1, 1) 
        
        # Mathematical Gradient of a Gaussian Log-Prob: advantage * (action - mu) / variance
        raw_grad_mu = advantages * (actions - mu) / var_sum
        
        # Safety constraint: Prevent exploding gradients
        grad_mu = np.clip(raw_grad_mu, -10.0, 10.0)
        
        # Divide the step size evenly across all trees in the ensemble
        step_sizes = (self.ppo_lr * grad_mu) / self.n_estimators
        
        # 4. Apply the gradients directly to your custom Python dictionaries!
        for b in range(batch_size):
            for i in range(total_trees):
                leaf_id = leaf_assignments[b, i]
                dim_idx = i % self.action_dim
                
                # Ensure the key exists before we add to it (Fallback protection)
                if "leaf_mean" not in self.actor_leaves_per_tree[i][leaf_id]:
                    self.actor_leaves_per_tree[i][leaf_id]["leaf_mean"] = self.actor_leaves_per_tree[i][leaf_id].get("leaf_weight", 0.0)
                
                # Gradient Ascent: Push the leaf mean in the direction of the gradient
                self.actor_leaves_per_tree[i][leaf_id]["leaf_mean"] += step_sizes[b, dim_idx]

        # 🟢 5. UPDATE THE GLOBAL STANDARD DEVIATION (Exploration)
        # Mathematical Gradient for variance: advantage * (((action - mu)^2 / var) - 1)
        grad_log_std = advantages * (((actions - mu) ** 2) / var_sum - 1.0)
        
        # Average the std gradient across the batch and apply a tiny clip for stability
        mean_grad_log_std = np.clip(grad_log_std.mean(axis=0), -0.5, 0.5)
        
        self.log_std += self.ppo_lr * mean_grad_log_std
        
        # Keep a floor so it never entirely collapses (Prevents math errors later)
        self.log_std = np.maximum(self.log_std, -2.0)

        #-----
                
        # # 5. Update the global Standard Deviation (Exploration Parameter)
        # # Gradient for variance: advantage * (((action - mu)^2 / var) - 1)
        # grad_log_std = advantages * (((actions - mu) ** 2) / var - 1.0)
        
        # # Average the std gradient across the batch and apply a tiny clip for stability
        # mean_grad_log_std = np.clip(grad_log_std.mean(axis=0), -0.5, 0.5)
        
        # self.log_std += self.ppo_lr * mean_grad_log_std

        # self.log_std = np.maximum(self.log_std, -1.2)

        # if self.actor_model is None: return
        
        # leaf_assignments = np.array(self.actor_model.predict(self._prepare_dmatrix(states), pred_leaf=True)).astype(int)
        # current_probs = softmax(self._predict_logits_batch(leaf_assignments))
        
        # actions = actions.astype(int)
        # targets = np.eye(self.action_dim)[actions]
        # grads = advantages[:, None] * (targets - current_probs) 
        # step_sizes = (self.ppo_lr * grads / self.n_estimators)
        
        # # 100% Vectorized Gradient Descent
        # total_trees = self.n_estimators * self.action_dim
        # tree_indices = np.arange(total_trees)
        
        # # Expand step sizes to map cleanly to the tree indexing
        # expanded_steps = np.tile(step_sizes, self.n_estimators)
        
        # flat_trees = np.tile(tree_indices, len(states))
        # flat_leaves = leaf_assignments.flatten()
        # flat_updates = expanded_steps.flatten()
        
        # # Instantly applies all batch gradients directly to memory!
        # np.add.at(self.fast_leaf_array, (flat_trees, flat_leaves), flat_updates)

        #return

        # X = xgb.DMatrix(x.T, feature_types=self.feature_types, enable_categorical=self.enable_categorical)

        # #get leaf indices
        # leaf_preds = np.array(self.model.predict(X, pred_leaf=True))

        # #get residual and update leaf values
        # residual = y - self.base_score  #first residual
        # for i in range(len(self.leaves_per_tree)):
        #     pred = self.model.predict(X, iteration_range=(0,i+1))

        #     self.leaves_per_tree[i][leaf_preds[0][i]]["leaf_mean"] = ( (self.leaves_per_tree[i][leaf_preds[0][i]]["leaf_mean"] * 
        #                                                             self.leaves_per_tree[i][leaf_preds[0][i]]["leaf_count"] + 
        #                                                             residual * self.lr) / (self.leaves_per_tree[i][leaf_preds[0][i]]["leaf_count"] + 1) )[0][0]
        #     self.leaves_per_tree[i][leaf_preds[0][i]]["leaf_variance"] = ( (self.leaves_per_tree[i][leaf_preds[0][i]]["leaf_variance"] *
        #                                                                 (self.leaves_per_tree[i][leaf_preds[0][i]]["leaf_count"] - 1) + 
        #                                                                 (pred - residual)**2 * self.lr**2) / (self.leaves_per_tree[i][leaf_preds[0][i]]["leaf_count"]) )[0][0]
        #     self.leaves_per_tree[i][leaf_preds[0][i]]["leaf_count"] += 1

        #     residual = y - pred

    def pick_action(self, obs):
        if self.actor_model is None:
            # Dynamically sample from the exact physical limits of the environment
            acts = np.random.uniform(self.action_low, self.action_high, size=(len(obs), self.action_dim))
            
            # Dynamically calculate the uniform log prob based on the action range
            range_size = self.action_high - self.action_low
            log_probs = np.sum(-np.log(range_size)) * np.ones(len(obs))
            
            # 🟢 THE FIX: Return 3 values here too! (acts, raw_acts, log_probs)
            return acts, acts, log_probs
            
        leaf_assignments = np.array(self.actor_model.predict(self._prepare_dmatrix(obs), pred_leaf=True)).astype(int)
        
        batch_size = len(obs)
        total_trees = self.n_estimators * self.action_dim
        
        mu = np.zeros((batch_size, self.action_dim))
        var_sum = np.zeros((batch_size, self.action_dim))
        
        for b in range(batch_size):
            for i in range(total_trees):
                leaf_id = leaf_assignments[b, i]
                dim_idx = i % self.action_dim  
                
                leaf_data = self.actor_leaves_per_tree[i][leaf_id]
                
                leaf_val = leaf_data.get("leaf_mean", leaf_data.get("leaf_weight", 0.0))
                mu[b, dim_idx] += leaf_val
                
                leaf_var = leaf_data.get("leaf_variance", 1e-4)
                var_sum[b, dim_idx] += leaf_var
                
        mu += self.actor_params['base_score']

        # var_sum = np.clip(var_sum, 1e-4, 5.0) 
        # std = np.sqrt(var_sum)
        
        # # Sample the raw Gaussian
        # raw_actions = np.random.normal(loc=mu, scale=std)
        
        # # Math requires the raw unclipped actions
        # log_probs = -0.5 * (((raw_actions - mu) ** 2) / var_sum + np.log(var_sum) + np.log(2 * np.pi))
        # log_probs = log_probs.sum(axis=1) 
        
        # # NEW: Clip using the dynamic environment bounds!
        # actions = np.clip(raw_actions, self.action_low, self.action_high) 
        # #print('actions, log_probs:', actions, log_probs)
        # return actions, log_probs

        #test 26-04-17
        
        # 🟢 THE FIX: Use global exploration std
        std = np.exp(self.log_std)
        var_sum = std ** 2 
        
        # Sample the raw Gaussian
        raw_actions = np.random.normal(loc=mu, scale=std)
        
        log_probs = -0.5 * (((raw_actions - mu) ** 2) / var_sum + np.log(var_sum) + np.log(2 * np.pi))
        log_probs = log_probs.sum(axis=1) 
        
        actions = np.clip(raw_actions, self.action_low, self.action_high) 
        
        # 🟢 RETURN RAW ACTIONS for the PPO Buffer!
        return actions, raw_actions, log_probs

    def predict_value(self, obs):
        if self.critic_model is None: return np.zeros(len(obs))
        return self.critic_model.predict(self._prepare_dmatrix(obs))


class Hybrid_XGB:
    """The SB3-Compatible Wrapper"""
    def __init__(self, env, awr_update_freq=10000, awr_buffer_size=15000, 
                 n_steps=256, gamma=0.99, gae_lambda=0.95, ppo_lr=0.01, awr_beta=0.05, 
                 max_depth=6, n_estimators=500, **kwargs):
        self.env = env
        self.n_envs = env.num_envs
        self.n_steps = n_steps
        self.awr_update_freq = awr_update_freq
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        self.action_dim = env.action_space.n if hasattr(env.action_space, 'n') else env.action_space.shape[0]
        
        # --- NEW: Extract dynamic continuous bounds ---
        if hasattr(env.action_space, 'low'):
            self.action_low = env.action_space.low
            self.action_high = env.action_space.high
        else:
            # Fallback just in case
            self.action_low = np.full(self.action_dim, -1.0)
            self.action_high = np.full(self.action_dim, 1.0)
            
        self.obs_dim = env.observation_space.shape
        self.obs_dtype = object 
        
        # Pass the bounds into the Engine!
        self.engine = XGBoostTreeEngine(self.action_dim, max_depth, n_estimators, ppo_lr, awr_beta, self.action_low, self.action_high)
        
        self.rollout_buffer = RolloutBuffer(n_steps, env.observation_space, env.action_space, 
                                            device='cpu', gamma=gamma, gae_lambda=gae_lambda, n_envs=self.n_envs)
        
        # Use np.empty with dtype=object to create an array of generic Python pointers
        self.ppo_obs_buffer = np.empty((n_steps, self.n_envs) + self.obs_dim, dtype=object)
        
        self.awr_buffer = {
            'obs': np.empty((awr_buffer_size,) + self.obs_dim, dtype=object),
            'actions': np.zeros((awr_buffer_size, self.action_dim), dtype=np.float32),
            'returns': np.zeros(awr_buffer_size, dtype=np.float32),    # NEW: Store Returns
            'advantages': np.zeros(awr_buffer_size, dtype=np.float32), # NEW: Store Advantages
            'ptr': 0, 'size': 0, 'max_size': awr_buffer_size
        }
        self.num_timesteps = 0

        self.tensorboard_log = kwargs.get('tensorboard_log', None)
        self.verbose = kwargs.get('verbose', 1)
        self.logger = None
        self.ep_info_buffer = deque(maxlen=5)

    def set_logger(self, logger):
        """Allows train.py to attach the SB3 terminal/tensorboard logger."""
        self.logger = logger

    # Note: We add tb_log_name="HYBRID_XGB" to match standard SB3 signatures
    def learn(self, total_timesteps, callback=None, log_interval=1, tb_log_name="HYBRID_XGB", **kwargs):
        
        # --- NEW: Use SB3's exact native logger to match AWR_XGB perfectly ---
        if self.logger is None:
            from stable_baselines3.common.utils import configure_logger
            self.logger = configure_logger(self.verbose, self.tensorboard_log, tb_log_name, reset_num_timesteps=True)
        # ---------------------------------------------------------------------
        
        obs = self.env.reset()
        episode_starts = np.ones(self.n_envs, dtype=bool)
        
        iteration = 0
        import time
        start_time = time.time()
        
        print("🚀 Starting Hybrid XGBoost Training Loop...")
        while self.num_timesteps < total_timesteps:
            self.rollout_buffer.reset()
            iteration += 1
            
            # --- 1. COLLECT DATA ---
            for step in range(self.n_steps):
                # 🟢 Catch the raw_actions
                actions, raw_actions, log_probs = self.engine.pick_action(obs)
                values = self.engine.predict_value(obs)
                next_obs, rewards, dones, infos = self.env.step(actions)
                # actions, log_probs = self.engine.pick_action(obs)
                # values = self.engine.predict_value(obs)
                # next_obs, rewards, dones, infos = self.env.step(actions)
                
                # Extract Episode Rewards for Logging!
                for info in infos:
                    if "episode" in info:
                        self.ep_info_buffer.append(info["episode"])
                        
                # STORE REAL OBS in our safe custom buffer
                self.ppo_obs_buffer[step] = obs
                
                # FEED DUMMY OBS to SB3 just so it does the math without crashing
                                # 🟢 STORE RAW ACTIONS in the buffer!
                dummy_obs = np.zeros_like(obs, dtype=np.float32)
                self.rollout_buffer.add(dummy_obs, raw_actions, rewards, episode_starts, th.tensor(values), th.tensor(log_probs))

                obs = next_obs
                episode_starts = dones
                self.num_timesteps += self.n_envs
            
            # --- 2. MICRO-UPDATE (PPO) ---
            last_values = self.engine.predict_value(obs)
            self.rollout_buffer.compute_returns_and_advantage(last_values=th.tensor(last_values), dones=dones)
            
            # Extract perfectly calculated arrays
            flat_obs = self.ppo_obs_buffer.reshape(-1, *self.obs_dim)
            flat_actions = self.rollout_buffer.actions.reshape(-1, self.action_dim)
            flat_old_log_probs = self.rollout_buffer.log_probs.reshape(-1)
            flat_returns = self.rollout_buffer.returns.reshape(-1)
            flat_advantages = self.rollout_buffer.advantages.reshape(-1)
            
            # --- NEW: Dump the computed arrays into the AWR Buffer! ---
            n_samples = len(flat_obs)
            for i in range(n_samples):
                idx = (self.awr_buffer['ptr'] + i) % self.awr_buffer['max_size']
                self.awr_buffer['obs'][idx] = flat_obs[i]
                self.awr_buffer['actions'][idx] = flat_actions[i]
                self.awr_buffer['returns'][idx] = flat_returns[i]
                self.awr_buffer['advantages'][idx] = flat_advantages[i]
                
            self.awr_buffer['ptr'] = (self.awr_buffer['ptr'] + n_samples) % self.awr_buffer['max_size']
            self.awr_buffer['size'] = min(self.awr_buffer['size'] + n_samples, self.awr_buffer['max_size'])
            # ----------------------------------------------------------

            self.engine.train_micro_ppo(flat_obs, flat_actions, flat_old_log_probs, flat_advantages)
                
            # --- 3. MACRO-UPDATE (AWR) ---
            if (self.num_timesteps % self.awr_update_freq) < (self.n_envs * self.n_steps):
                valid_size = self.awr_buffer['size']
                if valid_size > 500:
                    print(f"[Timestep {self.num_timesteps}] Triggering AWR Macro-Update...")
                    m_obs = self.awr_buffer['obs'][:valid_size]
                    m_act = self.awr_buffer['actions'][:valid_size]
                    
                    # Simply pull the perfectly calculated arrays from memory!
                    m_adv = self.awr_buffer['advantages'][:valid_size]
                    m_ret = self.awr_buffer['returns'][:valid_size] 
                    
                    self.engine.train_macro_awr(m_obs, m_act, m_adv, m_ret)

            if self.logger and iteration % log_interval == 0:
                fps = int(self.num_timesteps / (time.time() - start_time))
                self.logger.record("time/iterations", iteration)
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time.time() - start_time))
                self.logger.record("time/total_timesteps", self.num_timesteps)
                
                # --- NEW: Log the exact number of active trees! ---
                if self.engine.actor_model is not None:
                    self.logger.record("trees/actor_count", self.engine.n_estimators * self.action_dim)
                    self.logger.record("trees/critic_count", self.engine.n_estimators)
                else:
                    self.logger.record("trees/actor_count", 0)
                    self.logger.record("trees/critic_count", 0)
                # --------------------------------------------------
                
                if len(self.ep_info_buffer) > 0:
                    self.logger.record("rollout/ep_rew_mean", np.mean([ep["r"] for ep in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean", np.mean([ep["l"] for ep in self.ep_info_buffer]))
                
                # Push everything to the terminal!
                self.logger.dump(step=self.num_timesteps)

        return self