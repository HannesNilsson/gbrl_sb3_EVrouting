import numpy as np
import json
import xgboost as xgb
import torch as th
import pandas as pd
import time
from stable_baselines3.common.buffers import RolloutBuffer
from collections import deque
 
 
def get_leafs(tree: dict) -> dict:
    """Iterative leaf extraction from an XGBoost JSON-dump tree node."""
    leafs = {}
    stack = [tree]
    while stack:
        node = stack.pop()
        try:
            stack.append(node['children'][0])
            stack.append(node['children'][1])
        except KeyError:
            leafs[node['nodeid']] = node
    return leafs
 
 
class XGBoostTreeEngine:
    """
    Core math engine: vectorized NumPy inference and gradient updates.
 
    Architecture
    ------------
    * Actor  – one XGBoost multi-output regressor; get_dump yields
               n_estimators * action_dim trees (one tree per output per round).
               Tree index i handles action dimension  i % action_dim.
    * Critic – standard single-output regressor.
 
    Fast arrays (built after every AWR macro-update)
    ------------------------------------------------
    fast_leaf_means_arr  : (total_trees, max_leaf_id+1)  float32
        Current policy mean per leaf.  Updated in-place by micro-PPO.
    native_leaf_weights_arr : (total_trees, max_leaf_id+1)  float32
        XGBoost's own leaf weights, used only for residual reconstruction
        in train_macro_awr (avoids n_estimators*action_dim predict calls).
    """
 
    def __init__(self, action_dim, max_depth, n_estimators, ppo_lr, awr_beta,
                 action_low, action_high, clip_ratio=0.2):
        self.action_dim      = action_dim
        self.n_estimators    = n_estimators
        self.ppo_lr          = ppo_lr
        self.awr_beta        = awr_beta
        self.clip_ratio      = clip_ratio
        self.action_low      = action_low
        self.action_high     = action_high
 
        self.actor_model  = None
        self.critic_model = None
 
        self.actor_leaves_per_tree  = None
        self.critic_leaves_per_tree = None
 
        # Dense memory banks (allocated after first AWR update)
        self.fast_leaf_means_arr    = None   # (total_trees, max_lid+1)
        self.native_leaf_weights_arr = None  # (total_trees, max_lid+1)
        self._max_leaf_id           = 0
        self._total_trees           = 0      # n_estimators * action_dim
        self._dim_indices           = None   # (total_trees,)  i % action_dim
 
        self.actor_params = {
            'objective':  'reg:squarederror',
            'max_depth':  max_depth,
            'tree_method':'hist',
            'base_score': 0,
            'eta':        0.01,
            'verbosity':  0,
        }
        self.critic_params = {
            'objective':  'reg:squarederror',
            'max_depth':  max_depth,
            'tree_method':'hist',
            'base_score': 0,
            'eta':        0.01,
            'verbosity':  0,
        }
        self.actor_lr  = self.actor_params['eta']
        self.critic_lr = self.critic_params['eta']
 
 
    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
 
    def _prepare_dmatrix(self, X, label=None, weight=None):
        enable_cat = False
        if X.dtype == object or X.dtype.kind in {'S', 'U'}:
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            if isinstance(X.flat[0], bytes):
                X = np.char.decode(X.astype('S'), 'utf-8')
            try:
                X = X.astype(np.float32)
            except ValueError:
                df = pd.DataFrame(X)
                for col in df.columns:
                    df[col] = df[col].astype('category')
                X = df
                enable_cat = True
        # NOTE: weight is accepted in the signature for call-site compatibility
        # but intentionally not forwarded to DMatrix, matching the original.
        return xgb.DMatrix(X, label=label, enable_categorical=enable_cat)
 
    def _build_fast_arrays(self):
        """
        Convert actor_leaves_per_tree dicts into dense numpy arrays for
        O(1) vectorized lookup.  Called at the end of every AWR update.
        """
        total_trees = self.n_estimators * self.action_dim
        self._total_trees = total_trees
        self._dim_indices = np.arange(total_trees, dtype=np.int32) % self.action_dim
 
        max_lid = max(
            (max(d.keys()) for d in self.actor_leaves_per_tree if d),
            default=0
        )
        self._max_leaf_id = max_lid
 
        means   = np.zeros((total_trees, max_lid + 1), dtype=np.float32)
        native  = np.zeros((total_trees, max_lid + 1), dtype=np.float32)
        # Aleatoric variance per (tree, leaf): var(residuals in leaf) * lr^2.
        # Summed across trees in pick_action to give a per-dim, per-observation
        # exploration width — more expressive than a single global log_std.
        vars_   = np.full((total_trees, max_lid + 1), 1e-4, dtype=np.float32)
 
        for i, tree_dict in enumerate(self.actor_leaves_per_tree):
            for lid, ldata in tree_dict.items():
                means[i, lid]  = ldata.get('leaf_mean',
                                 ldata.get('leaf_weight', 0.0))
                native[i, lid] = ldata.get('leaf', 0.0)
                vars_[i, lid]  = max(ldata.get('leaf_variance', 1e-4), 1e-4)
 
        self.fast_leaf_means_arr     = means
        self.native_leaf_weights_arr = native
        self.fast_leaf_vars_arr      = vars_
 
        # Per-tree update weights: inversely proportional to each tree's mean
        # absolute leaf value.  Early boosting rounds fit the full signal and
        # tend to have large leaf values; later rounds fit small residuals and
        # have small leaf values.  Normalizing by magnitude puts all trees on
        # the same scale before applying the gradient.
        # Weights are normalized to sum to 1 so the total update magnitude
        # across the ensemble equals ppo_lr * grad_mu — same as before.
        mean_abs = np.zeros(total_trees, dtype=np.float32)
        for i, tree_dict in enumerate(self.actor_leaves_per_tree):
            vals = np.array([
                ldata.get('leaf_mean', ldata.get('leaf_weight', 0.0))
                for ldata in tree_dict.values()
            ], dtype=np.float32)
            mean_abs[i] = np.mean(np.abs(vals)) if len(vals) > 0 else 0.0
 
        raw_w = 1.0 / (mean_abs + 1e-8)
        self._tree_weights = (raw_w / raw_w.sum()).astype(np.float32)  # (total_trees,)
 
    # ------------------------------------------------------------------
    # Macro update  (AWR)
    # ------------------------------------------------------------------
 
    def train_macro_awr(self, states, actions, advantages, returns):
        returns_1d  = returns.flatten()
        advantages  = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        weights     = np.clip(np.exp(advantages / self.awr_beta), 0, 20.0).flatten()
        actions_flat = actions.astype(np.float32).reshape(-1, self.action_dim)
        N = len(states)
 
        # --- Train models ---
        dtrain_act  = self._prepare_dmatrix(states, label=actions_flat, weight=weights)
        self.actor_model = xgb.train(self.actor_params, dtrain_act,
                                     num_boost_round=self.n_estimators)
 
        dtrain_crit = self._prepare_dmatrix(states, label=returns_1d, weight=weights)
        self.critic_model = xgb.train(self.critic_params, dtrain_crit,
                                      num_boost_round=self.n_estimators)
 
        X_eval = self._prepare_dmatrix(states)
 
        # ---- CRITIC leaf stats ----
        dump_crit = self.critic_model.get_dump(with_stats=True, dump_format='json')
        json_trees_crit = [json.loads(t) for t in dump_crit]
        self.critic_leaves_per_tree = [get_leafs(t) for t in json_trees_crit]
 
        leaf_preds_crit = self.critic_model.predict(X_eval, pred_leaf=True).astype(int)  # (N, n_est)
 
        # Build native critic leaf-weight array so we can reconstruct cumulative
        # predictions without calling predict(iteration_range=...) n_estimators times.
        crit_max_lid = max(
            (max(d.keys()) for d in self.critic_leaves_per_tree if d), default=0
        )
        native_crit = np.zeros((self.n_estimators, crit_max_lid + 1), dtype=np.float32)
        for i, tree_dict in enumerate(self.critic_leaves_per_tree):
            for lid, ldata in tree_dict.items():
                native_crit[i, lid] = ldata.get('leaf', 0.0)
 
        # gathered_crit[n, i] = native leaf weight for sample n in tree i  -> (N, n_est)
        gathered_crit = native_crit[np.arange(self.n_estimators), leaf_preds_crit]
 
        # Match original: pred_at tree i = predict(iteration_range=(0, i+1))
        #   i=0  -> base_score only (XGBoost special-cases first tree)
        #   i>0  -> base_score + cumsum of trees 0..i  (inclusive)
        cumsum_crit = np.cumsum(gathered_crit, axis=1)             # (N, n_est)
        pred_at_crit = np.empty_like(cumsum_crit)
        pred_at_crit[:, 0]  = self.critic_params['base_score']
        pred_at_crit[:, 1:] = self.critic_params['base_score'] + cumsum_crit[:, 1:]
 
        residuals_crit = returns_1d[:, np.newaxis] - pred_at_crit  # (N, n_est)
 
        for i in range(self.n_estimators):
            df = pd.DataFrame({'leaf': leaf_preds_crit[:, i], 'res': residuals_crit[:, i]})
            grp = df.groupby('leaf')['res'].agg(['count', 'mean', 'var'])
            for lid, row in grp.iterrows():
                d = self.critic_leaves_per_tree[i][lid]
                d['leaf_count']    = row['count']
                d['leaf_mean']     = d.get('leaf_weight', 0.0)
                var = row['var'] if not pd.isna(row['var']) else 0.0
                d['leaf_variance'] = var * self.critic_lr ** 2
 
        # ---- ACTOR leaf stats (vectorized residual reconstruction) ----
        total_trees = self.n_estimators * self.action_dim
        dump_act  = self.actor_model.get_dump(with_stats=True, dump_format='json')
        json_trees_act = [json.loads(t) for t in dump_act]
        self.actor_leaves_per_tree = [get_leafs(t) for t in json_trees_act]
 
        # pred_leaf → (N, total_trees); each column = leaf id for that dump-tree
        leaf_preds_act = self.actor_model.predict(X_eval, pred_leaf=True).astype(int)
 
        # Build native weight lookup from the just-parsed dump trees.
        # We need this to reconstruct cumulative predictions without calling
        # model.predict() in a loop (which was the expensive part).
        tmp_max_lid = max(
            (max(d.keys()) for d in self.actor_leaves_per_tree if d), default=0
        )
        tmp_native = np.zeros((total_trees, tmp_max_lid + 1), dtype=np.float32)
        for i, tree_dict in enumerate(self.actor_leaves_per_tree):
            for lid, ldata in tree_dict.items():
                tmp_native[i, lid] = ldata.get('leaf', 0.0)
 
        # Gather native leaf weights for every (sample, tree) → (N, total_trees)
        tree_idx = np.arange(total_trees)
        gathered_native = tmp_native[tree_idx, leaf_preds_act]  # (N, total_trees)
 
        # Reshape to (N, n_estimators, action_dim): axis-2 = dim, axis-1 = round
        # Tree layout: [dim0_r0, dim1_r0, …, dim(D-1)_r0, dim0_r1, …]
        # → C-order reshape puts last axis fastest → [b, round, dim]  ✓
        gathered_3d = gathered_native.reshape(N, self.n_estimators, self.action_dim)
 
        # The original calls predict(iteration_range=(0, r+1)) for round r > 0,
        # meaning the prediction *includes* round r (not just up to r-1).
        # Concretely:
        #   r = 0  → pred = base_score                   (special-cased)
        #   r > 0  → pred = base_score + cumsum[0..r]    (inclusive of round r)
        cumsum_3d = np.cumsum(gathered_3d, axis=1)           # (N, n_est, D)
        pred_at_3d = np.empty_like(cumsum_3d)
        pred_at_3d[:, 0, :]  = self.actor_params['base_score']          # r=0: base only
        pred_at_3d[:, 1:, :] = self.actor_params['base_score'] + cumsum_3d[:, 1:, :]  # r>0
 
        # residuals_3d[b, r, d] = actions[b, d] - pred_at[b, r, d]
        residuals_3d = actions_flat[:, np.newaxis, :] - pred_at_3d  # (N, n_est, D)
 
        # Write stats back into the leaf dicts
        for i in range(total_trees):
            round_idx = i // self.action_dim
            dim_idx   = i % self.action_dim
 
            leaf_col = leaf_preds_act[:, i]
            res_col  = residuals_3d[:, round_idx, dim_idx]
 
            df = pd.DataFrame({'leaf': leaf_col, 'res': res_col})
            grp = df.groupby('leaf')['res'].agg(['count', 'mean', 'var'])
            for lid, row in grp.iterrows():
                d = self.actor_leaves_per_tree[i][lid]
                d['leaf_count'] = row['count']
                d['leaf_mean']  = row['mean'] * self.actor_lr
                var = row['var'] if not pd.isna(row['var']) else 0.0
                d['leaf_variance'] = var * self.actor_lr ** 2
 
        # Compile dense fast arrays for inference and micro-updates
        self._build_fast_arrays()
 
    # ------------------------------------------------------------------
    # Micro update  (PPO)  — fully vectorized
    # ------------------------------------------------------------------
 
    def train_micro_ppo(self, states, actions, old_log_probs, advantages):
        if self.actor_model is None:
            return
 
        advantages   = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        actions_flat = actions.astype(np.float32).reshape(-1, self.action_dim)
        advantages   = advantages.reshape(-1, 1)
        N            = len(states)
        T            = self._total_trees
 
        # ---- 1. Leaf assignments ----
        leaf_assignments = self.actor_model.predict(
            self._prepare_dmatrix(states), pred_leaf=True
        ).astype(int)                                    # (N, T)
        # Clip defensively against any leaf ID that exceeds our array size
        np.clip(leaf_assignments, 0, self._max_leaf_id,
                out=leaf_assignments)
 
        # ---- 2. Vectorized mu reconstruction ----
        # gathered_means[n, t] = fast_leaf_means_arr[t, leaf_assignments[n, t]]
        tree_idx      = np.arange(T)                     # (T,)
        gathered      = self.fast_leaf_means_arr[tree_idx, leaf_assignments]  # (N, T)
        # Sum across rounds for each dim: reshape (N, n_estimators, action_dim)
        mu = gathered.reshape(N, self.n_estimators, self.action_dim).sum(axis=1)  # (N, D)
        mu += self.actor_params['base_score']
 
        # ---- 3. Gather per-leaf variances and compute gradients ----
        gathered_vars = self.fast_leaf_vars_arr[tree_idx, leaf_assignments]  # (N, T)
        var_sum = np.clip(
            gathered_vars.reshape(N, self.n_estimators, self.action_dim).sum(axis=1),
            1e-4, 5.0,
        )                                                                     # (N, D)
 
        # Mean gradient: advantage * (action - mu) / var  [chain rule, additive ensemble]
        raw_grad_mu = advantages * (actions_flat - mu) / var_sum
        grad_mu     = np.clip(raw_grad_mu, -10.0, 10.0)
        step_mu     = self.ppo_lr * grad_mu                                   # (N, D)
 
        # Variance gradient: ∂log π/∂var = 0.5 * advantage * ((a-mu)²/var² - 1/var)
        # Each tree's var contributes additively to var_sum, so ∂var_sum/∂var_t = 1
        # and the chain rule gives the same expression for every contributing tree.
        grad_var  = 0.5 * advantages * (
            (actions_flat - mu) ** 2 / var_sum ** 2 - 1.0 / var_sum
        )
        step_var  = self.ppo_lr * grad_var                                    # (N, D)
 
        # ---- 4. Scatter mean and variance updates via per-tree bincount ----
        step_mu_per_tree  = step_mu[:, self._dim_indices]  * self._tree_weights[np.newaxis, :]
        step_var_per_tree = step_var[:, self._dim_indices] * self._tree_weights[np.newaxis, :]
 
        max_lid_p1 = self._max_leaf_id + 1
        for t in range(T):
            lids = leaf_assignments[:, t]
            mean_update = np.bincount(lids, weights=step_mu_per_tree[:, t],
                                      minlength=max_lid_p1).astype(np.float32)
            var_update  = np.bincount(lids, weights=step_var_per_tree[:, t],
                                      minlength=max_lid_p1).astype(np.float32)
            self.fast_leaf_means_arr[t] += mean_update[:max_lid_p1]
            self.fast_leaf_vars_arr[t]  += var_update[:max_lid_p1]
 
        # Enforce aleatoric variance floor — prevents the distribution collapsing
        np.maximum(self.fast_leaf_vars_arr, 1e-4, out=self.fast_leaf_vars_arr)
 
    # ------------------------------------------------------------------
    # Inference — fully vectorized
    # ------------------------------------------------------------------
 
    def pick_action(self, obs):
        if self.actor_model is None:
            acts      = np.random.uniform(self.action_low, self.action_high,
                                          size=(len(obs), self.action_dim))
            log_probs = np.full(len(obs), np.sum(-np.log(self.action_high - self.action_low)))
            return acts, acts, log_probs
 
        N = len(obs)
        T = self._total_trees
 
        leaf_assignments = self.actor_model.predict(
            self._prepare_dmatrix(obs), pred_leaf=True
        ).astype(int)                                    # (N, T)
        np.clip(leaf_assignments, 0, self._max_leaf_id,
                out=leaf_assignments)
 
        tree_idx = np.arange(T)
        gathered = self.fast_leaf_means_arr[tree_idx, leaf_assignments]  # (N, T)
        mu       = gathered.reshape(N, self.n_estimators, self.action_dim).sum(axis=1)
        mu      += self.actor_params['base_score']       # (N, D)
 
        # Sum per-tree aleatoric variances for each action dimension.
        # Each tree handles one dim (dim_indices[t] = t % action_dim), so
        # reshaping (N, T) → (N, n_estimators, action_dim) and summing axis 1
        # correctly accumulates variance across boosting rounds per dimension.
        gathered_vars = self.fast_leaf_vars_arr[tree_idx, leaf_assignments]  # (N, T)
        var = np.clip(
            gathered_vars.reshape(N, self.n_estimators, self.action_dim).sum(axis=1),
            1e-4, 5.0,
        )                                                # (N, D)
        std = np.sqrt(var)
 
        raw_actions = np.random.normal(loc=mu, scale=std)
        log_probs   = (
            -0.5 * (((raw_actions - mu) ** 2) / var
                    + np.log(var)
                    + np.log(2 * np.pi))
        ).sum(axis=1)                                    # (N,)
 
        actions = np.clip(raw_actions, self.action_low, self.action_high)
        return actions, raw_actions, log_probs
 
    def predict_value(self, obs):
        if self.critic_model is None:
            return np.zeros(len(obs))
        return self.critic_model.predict(self._prepare_dmatrix(obs))
 
 
# ---------------------------------------------------------------------------
# SB3-Compatible Wrapper
# ---------------------------------------------------------------------------
 
class Hybrid_XGB:
    """SB3-compatible training loop."""
 
    def __init__(self, env, awr_update_freq=10000, awr_buffer_size=15000,
                 n_steps=256, gamma=0.99, gae_lambda=0.95, ppo_lr=0.01,
                 awr_beta=0.05, max_depth=6, n_estimators=500, **kwargs):
        self.env      = env
        self.n_envs   = env.num_envs
        self.n_steps  = n_steps
        self.awr_update_freq = awr_update_freq
        self.gamma       = gamma
        self.gae_lambda  = gae_lambda
 
        self.action_dim = (
            env.action_space.n
            if hasattr(env.action_space, 'n')
            else env.action_space.shape[0]
        )
 
        if hasattr(env.action_space, 'low'):
            self.action_low  = env.action_space.low
            self.action_high = env.action_space.high
        else:
            self.action_low  = np.full(self.action_dim, -1.0)
            self.action_high = np.full(self.action_dim, 1.0)
 
        self.obs_dim   = env.observation_space.shape
        self.obs_dtype = object
 
        self.engine = XGBoostTreeEngine(
            self.action_dim, max_depth, n_estimators, ppo_lr, awr_beta,
            self.action_low, self.action_high,
        )
 
        self.rollout_buffer = RolloutBuffer(
            n_steps, env.observation_space, env.action_space,
            device='cpu', gamma=gamma, gae_lambda=gae_lambda, n_envs=self.n_envs,
        )
 
        # Separate obs buffer (dtype=object for XGBoost; SB3 gets zeros)
        self.ppo_obs_buffer = np.empty(
            (n_steps, self.n_envs) + self.obs_dim, dtype=object
        )
 
        self.awr_buffer = {
            'obs':        np.empty((awr_buffer_size,) + self.obs_dim, dtype=object),
            'actions':    np.zeros((awr_buffer_size, self.action_dim), dtype=np.float32),
            'returns':    np.zeros(awr_buffer_size, dtype=np.float32),
            'advantages': np.zeros(awr_buffer_size, dtype=np.float32),
            'ptr': 0, 'size': 0, 'max_size': awr_buffer_size,
        }
        self.num_timesteps = 0
 
        self.tensorboard_log = kwargs.get('tensorboard_log', None)
        self.verbose         = kwargs.get('verbose', 1)
        self.logger          = None
        self.ep_info_buffer  = deque(maxlen=5)
 
    def set_logger(self, logger):
        self.logger = logger
 
    # ------------------------------------------------------------------
    # Vectorized ring-buffer write
    # ------------------------------------------------------------------
 
    def _store_in_awr_buffer(self, flat_obs, flat_actions, flat_returns, flat_advantages):
        """Write a batch into the ring buffer without a Python loop."""
        n   = len(flat_obs)
        buf = self.awr_buffer
        cap = buf['max_size']
 
        indices = (np.arange(n) + buf['ptr']) % cap
 
        buf['obs'][indices]        = flat_obs
        buf['actions'][indices]    = flat_actions
        buf['returns'][indices]    = flat_returns
        buf['advantages'][indices] = flat_advantages
 
        buf['ptr']  = int((buf['ptr'] + n) % cap)
        buf['size'] = min(buf['size'] + n, cap)
 
    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
 
    def learn(self, total_timesteps, callback=None, log_interval=1,
              tb_log_name="HYBRID_XGB", **kwargs):
 
        if self.logger is None:
            from stable_baselines3.common.utils import configure_logger
            self.logger = configure_logger(
                self.verbose, self.tensorboard_log, tb_log_name,
                reset_num_timesteps=True,
            )
 
        obs            = self.env.reset()
        episode_starts = np.ones(self.n_envs, dtype=bool)
        iteration      = 0
        start_time     = time.time()
 
        print("🚀 Starting Hybrid XGBoost Training Loop...")
 
        while self.num_timesteps < total_timesteps:
            self.rollout_buffer.reset()
            iteration += 1
 
            # --- 1. COLLECT ---
            for step in range(self.n_steps):
                actions, raw_actions, log_probs = self.engine.pick_action(obs)
                values = self.engine.predict_value(obs)
                next_obs, rewards, dones, infos = self.env.step(actions)
 
                for info in infos:
                    if 'episode' in info:
                        self.ep_info_buffer.append(info['episode'])
 
                self.ppo_obs_buffer[step] = obs
 
                # SB3 buffer receives zero obs (used only for return/advantage math)
                dummy_obs = np.zeros_like(obs, dtype=np.float32)
                self.rollout_buffer.add(
                    dummy_obs, raw_actions, rewards, episode_starts,
                    th.tensor(values), th.tensor(log_probs),
                )
 
                obs            = next_obs
                episode_starts = dones
                self.num_timesteps += self.n_envs
 
            # --- 2. MICRO-UPDATE (PPO) ---
            last_values = self.engine.predict_value(obs)
            self.rollout_buffer.compute_returns_and_advantage(
                last_values=th.tensor(last_values), dones=dones,
            )
 
            flat_obs        = self.ppo_obs_buffer.reshape(-1, *self.obs_dim)
            flat_actions    = self.rollout_buffer.actions.reshape(-1, self.action_dim)
            flat_old_lp     = self.rollout_buffer.log_probs.reshape(-1)
            flat_returns    = self.rollout_buffer.returns.reshape(-1)
            flat_advantages = self.rollout_buffer.advantages.reshape(-1)
 
            self._store_in_awr_buffer(flat_obs, flat_actions, flat_returns, flat_advantages)
            self.engine.train_micro_ppo(flat_obs, flat_actions, flat_old_lp, flat_advantages)
 
            # --- 3. MACRO-UPDATE (AWR) ---
            if (self.num_timesteps % self.awr_update_freq) < (self.n_envs * self.n_steps):
                valid_size = self.awr_buffer['size']
                if valid_size > 500:
                    # Read all valid samples in insertion order using ring-buffer
                    # index arithmetic. Once the buffer is full ptr wraps around,
                    # so a plain [:valid_size] slice would silently return samples
                    # in array order rather than recency order.
                    # arange(ptr - n, ptr) % cap always yields the correct slots:
                    # before wrap-around this is identical to [:valid_size];
                    # after wrap-around it correctly spans the seam.
                    cap = self.awr_buffer['max_size']
                    ptr = self.awr_buffer['ptr']
                    idx = np.arange(ptr - valid_size, ptr) % cap
                    print(f"🌲 [Timestep {self.num_timesteps}] "
                          f"Triggering AWR Macro-Update ({valid_size} samples)...")
                    self.engine.train_macro_awr(
                        self.awr_buffer['obs'][idx],
                        self.awr_buffer['actions'][idx],
                        self.awr_buffer['advantages'][idx],
                        self.awr_buffer['returns'][idx],
                    )
 
            # --- 4. LOGGING ---
            if self.logger and iteration % log_interval == 0:
                fps = int(self.num_timesteps / (time.time() - start_time))
                self.logger.record('time/iterations',     iteration)
                self.logger.record('time/fps',            fps)
                self.logger.record('time/time_elapsed',   int(time.time() - start_time))
                self.logger.record('time/total_timesteps',self.num_timesteps)
 
                if self.engine.actor_model is not None:
                    self.logger.record('trees/actor_count',
                                       self.engine.n_estimators * self.action_dim)
                    self.logger.record('trees/critic_count', self.engine.n_estimators)
                else:
                    self.logger.record('trees/actor_count',  0)
                    self.logger.record('trees/critic_count', 0)
 
                if self.ep_info_buffer:
                    self.logger.record('rollout/ep_rew_mean',
                                       np.mean([ep['r'] for ep in self.ep_info_buffer]))
                    self.logger.record('rollout/ep_len_mean',
                                       np.mean([ep['l'] for ep in self.ep_info_buffer]))
 
                self.logger.dump(step=self.num_timesteps)
 
        return self