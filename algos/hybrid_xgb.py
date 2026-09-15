"""
Tree Ensemble Reinforcement Learning (TERL) with XGBoost function approximators.

A fixed-size actor/critic ensemble is rebuilt periodically with Advantage-Weighted
Regression (macro-update) and refined in between by on-policy leaf-value updates
(micro-update, A2C or PPO), following Algorithm 1 of the paper.

Fixes relative to the first draft implementation
------------------------------------------------
 1.  AWR sample weights are actually passed to XGBoost.  Previously
     `_prepare_dmatrix` accepted `weight` and silently dropped it, which turned
     the macro-update into unweighted behaviour cloning - i.e. no off-policy
     policy improvement at all, and an update that *reset* the policy toward the
     mean action in the replay buffer.
 2.  The critic is fit unweighted.  Advantage-weighting the value regression
     biases it away from V^pi.
 3.  The critic's leaves are micro-updated too, as Algorithm 1 line 9 requires
     ("...new critic and actor tree ensembles V_{k,l+1}, pi_{k,l+1}").  The old
     code computed critic leaf statistics, never used them, and populated
     `leaf_mean` from a nonexistent `leaf_weight` dump key.
 4.  Per-observation variance is the *mean* across boosting rounds of the
     per-leaf variance of the full-ensemble residual, not the sum of per-round
     residual variances.  Summing 500 strongly-correlated residual variances
     saturated the [1e-4, 5.0] clip, pinning the exploration std at ~2.24 for
     every state and both action dimensions.
 5.  Leaf updates are normalised (per-leaf mean by default) instead of summed,
     so the effective learning rate no longer scales with n_steps * n_envs and
     is no longer proportional to leaf population.
 6.  The PPO variant uses the real clipped-surrogate gradient
     grad = ratio * A * dlog pi, masked where the clip binds.  The old version
     dropped the ratio factor, making it A2C with a masking heuristic.
 7.  The actor uses one *shared* tree structure with vector-valued leaves
     (XGBoost `multi_strategy="multi_output_tree"`, requires XGBoost >= 2.0),
     so a single ensemble of `n_estimators` trees parameterises the whole
     policy - matching the paper's "actor tree ensemble pi" rather than one
     ensemble per action dimension.  The old code assumed an unverified
     ordering of XGBoost's multi-output dump; a wrong layout would have
     silently scrambled dimensions.  The layout is now checked explicitly
     (tree count, vector leaf arity, `pred_leaf` shape) and the reconstructed
     mean is verified against `Booster.predict`.  `shared_tree_structure=False`
     falls back to one single-output booster per dimension.
 8.  The r=0 residual no longer excludes tree 0's own contribution.
 9.  Per-tree Python loops replaced by a single offset bincount.
10.  AWR trigger uses a step counter instead of modulo arithmetic that can
     double-fire or drift.
11.  Truncated episodes are bootstrapped with gamma * V(terminal_obs).
12.  Observations are stored as float32 in the SB3 buffer directly (the parallel
     object-dtype buffer and the zero-filled dummy observations are gone).
13.  Adds `predict`, save/load, and diagnostics: `train/macro_mu_delta` vs
     `train/micro_mu_delta` (how far each update actually moves the policy
     mean), `policy/std_dim*`, `policy/var_at_ceiling_dim*`,
     `train/clip_fraction`, `train/approx_kl`, `train/explained_variance`.
14.  Variance bounds and the global log-std are per action dimension.  A single
     scalar scale is wrong when T_charging spans [0, 11] and C_target [0, 2].
15.  Leaf-update weights for the actor's mean table, the critic's value
     table, and (in the "additive" var_update path) the actor's variance
     table, default to proportional to each tree's own current leaf
     magnitude - not inversely proportional. A fixed-size gradient step has
     the largest effect on the final prediction wherever a tree already
     contributes most to it (mu, V, and sigma^2 are all plain sums over K
     trees), so the step budget is concentrated there rather than on
     low-magnitude trees. Every weight set still sums to 1 (so the
     *aggregate* output moves by exactly the intended amount, independent of
     K), and each is computed from its own table's leaf values -
     mean-magnitude for mu, variance-magnitude for sigma^2, value-magnitude
     for V - not borrowed from another table. `invert_tree_weights=True`
     reverts all three to the original inverse-magnitude scheme, for
     ablating the two against each other. The default "log" var_update path
     needs no such weighting at all regardless of this setting: scaling
     every leaf by the same multiplicative factor scales the leaf-variance
     sum by exactly that factor regardless of the per-tree distribution.
"""

from __future__ import annotations

import json
import time
import pickle
import warnings
from collections import deque

import numpy as np
import xgboost as xgb


# --------------------------------------------------------------------------- #
# Tree dump helpers
# --------------------------------------------------------------------------- #

def get_leafs(tree: dict) -> dict:
    """Iterative leaf extraction from an XGBoost JSON-dump tree node."""
    leafs, stack = {}, [tree]
    while stack:
        node = stack.pop()
        children = node.get("children")
        if children:
            stack.extend(children)
        else:
            leafs[node["nodeid"]] = node
    return leafs


def _native_leaf_array(leaves_per_tree, n_trees, max_lid):
    """(n_trees, max_lid+1) array of XGBoost's own leaf weights."""
    arr = np.zeros((n_trees, max_lid + 1), dtype=np.float64)
    for i, tree in enumerate(leaves_per_tree):
        for lid, node in tree.items():
            arr[i, lid] = node.get("leaf", 0.0)
    return arr


def _native_leaf_array_multi(leaves_per_tree, n_trees, max_lid, action_dim):
    """(action_dim, n_trees, max_lid+1) array from vector-valued leaves.

    A `multi_output_tree` dump stores each leaf as a list of `action_dim`
    values, e.g. {"nodeid": 7, "leaf": [-0.33, 0.06]}.
    """
    arr = np.zeros((action_dim, n_trees, max_lid + 1), dtype=np.float64)
    for i, tree in enumerate(leaves_per_tree):
        for lid, node in tree.items():
            v = np.atleast_1d(np.asarray(node.get("leaf", 0.0), dtype=np.float64))
            if v.size != action_dim:
                raise RuntimeError(
                    f"tree {i} leaf {lid} has arity {v.size}, expected "
                    f"{action_dim}. The booster is not a multi-output tree - "
                    "check that multi_strategy='multi_output_tree' was accepted."
                )
            arr[:, i, lid] = v
    return arr


def _leaf_stats(leaf_ids, residual, n_trees, n_leaf_slots):
    """Per-(tree, leaf) count / mean / variance of `residual`.

    leaf_ids : (N, n_trees) int
    residual : (N,) shared across trees, or (N, n_trees) one target per tree.

    The second form is what the stage-residual estimator needs: tree i is fit to
    z_i = y - mu_{1..i-1}, so each tree has its own target rather than sharing
    the final residual.  A single flattened bincount replaces the per-tree loop
    either way.
    """
    n = leaf_ids.shape[0]
    offsets = np.arange(n_trees, dtype=np.int64) * n_leaf_slots
    flat = (leaf_ids.astype(np.int64) + offsets[None, :]).ravel()
    residual = np.asarray(residual, dtype=np.float64)
    w = (residual.ravel() if residual.ndim == 2
         else np.repeat(residual, n_trees))          # row-major match

    size = n_trees * n_leaf_slots
    cnt = np.bincount(flat, minlength=size).astype(np.float64)
    s1 = np.bincount(flat, weights=w, minlength=size)
    s2 = np.bincount(flat, weights=w * w, minlength=size)

    safe = cnt > 0
    denom = np.where(safe, cnt, 1.0)
    mean = np.where(safe, s1 / denom, 0.0)
    var = np.where(cnt > 1, np.maximum(s2 / denom - mean ** 2, 0.0), 0.0)

    shape = (n_trees, n_leaf_slots)
    return cnt.reshape(shape), mean.reshape(shape), var.reshape(shape)


def _scatter_leaf_update(leaf_ids, per_sample_step, tree_weights,
                         n_trees, n_leaf_slots, normalization):
    """Aggregate per-sample steps into a (n_trees, n_leaf_slots) leaf delta.

    normalization
        'leaf_mean'  - mean gradient inside each leaf  (default)
        'batch_mean' - sum divided by batch size
        'sum'        - raw sum (the original behaviour; lr then scales with N)
    """
    n = leaf_ids.shape[0]
    offsets = np.arange(n_trees, dtype=np.int64) * n_leaf_slots
    flat = (leaf_ids.astype(np.int64) + offsets[None, :]).ravel()

    w = (per_sample_step[:, None] * tree_weights[None, :]).ravel()
    size = n_trees * n_leaf_slots
    sums = np.bincount(flat, weights=w, minlength=size)

    if normalization == "leaf_mean":
        cnt = np.bincount(flat, minlength=size).astype(np.float64)
        upd = sums / np.maximum(cnt, 1.0)
    elif normalization == "batch_mean":
        upd = sums / max(n, 1)
    elif normalization == "sum":
        upd = sums
    else:
        raise ValueError(f"unknown normalization {normalization!r}")

    return upd.reshape(n_trees, n_leaf_slots)


class _LeafAdam:
    """Per-leaf Adam moments for leaf-value updates.

    Raw SGD on leaf values does not work here.  The per-leaf mean of
    A * (a - mu) / var is a near-zero-mean quantity whose magnitude depends on
    the advantage scale, the action scale and the current variance, so a fixed
    learning rate produces a step that is orders of magnitude too small (on
    Pendulum the micro-update moved the policy mean by 1e-3 per iteration
    against an action range of 4).  Neural PPO does not have this problem
    because Adam normalises the per-parameter gradient magnitude; this does the
    same for leaves, so a step is ~lr regardless of gradient scale.
    """

    def __init__(self, shape, beta1=0.9, beta2=0.999, eps=1e-8):
        self.m = np.zeros(shape)
        self.v = np.zeros(shape)
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.t = 0

    def step(self, grad):
        self.t += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * grad ** 2
        mhat = self.m / (1.0 - self.beta1 ** self.t)
        vhat = self.v / (1.0 - self.beta2 ** self.t)
        return mhat / (np.sqrt(vhat) + self.eps)


def _target_ess_beta(adv, target_ess, clip, lo=0.05, hi=50.0, iters=40):
    """AWR temperature giving an effective sample size fraction of `target_ess`.

    With normalised advantages and a fixed beta=0.5, exp(2A) clipped at 20 has
    an effective sample size of 3-7% of the buffer: the regression is then
    high-variance cloning of a handful of lucky trajectories.  ESS is monotone
    increasing in beta, so a bisection pins it to a chosen value.
    """
    def ess(b):
        w = np.exp(np.clip(adv / b, -50.0, 50.0) - np.max(adv / b))
        w = np.clip(w, 0.0, clip)
        return w.sum() ** 2 / (len(w) * (w ** 2).sum() + 1e-12)

    if ess(hi) < target_ess:
        return hi
    if ess(lo) > target_ess:
        return lo
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if ess(mid) < target_ess:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _magnitude_weights(leaf_means, counts, invert):
    """Per-tree weights derived from mean |leaf value| within populated leaves,
    summing to 1 (so the *aggregate* mu/sigma^2 moves by exactly lr * grad,
    independent of K - see the callers).

    invert=False (proportional): a tree's share of the step budget scales
    with how much it currently contributes to the ensemble output, since a
    fixed-size change there has the largest effect on the final prediction.
    Used for the actor's mean and (additive-mode) variance tables.

    invert=True: the inverse - concentrates the budget on low-magnitude
    trees instead. Used for the critic, unchanged from the original design.
    """
    mask = counts > 0
    num = np.abs(leaf_means * mask).sum(axis=1)
    den = np.maximum(mask.sum(axis=1), 1)
    mean_abs = num / den
    raw = 1.0 / (mean_abs + 1e-8) if invert else mean_abs
    total = raw.sum()
    if total <= 1e-12:
        # No signal yet (e.g. every tree still at base_score) - fall back to
        # an even split rather than dividing by ~0.
        return np.full_like(raw, 1.0 / len(raw))
    return (raw / total).astype(np.float64)


# --------------------------------------------------------------------------- #
# Core engine
# --------------------------------------------------------------------------- #

class XGBoostTreeEngine:
    """Fixed-size XGBoost actor/critic with AWR rebuilds and leaf-value updates.

    Layout
    ------
    By default the actor is a single booster trained with
    `multi_strategy="multi_output_tree"`: one shared tree structure per boosting
    round, whose leaves hold an `action_dim`-vector.  `pred_leaf` therefore
    returns (N, n_estimators) - one leaf index per round, shared by all output
    dimensions.  With `shared_tree_structure=False` the actor is instead one
    single-output booster per dimension, each also yielding (N, n_estimators).

    Either way the downstream maths is identical, because `_leaf_ids` returns a
    per-dimension list of (N, n_estimators) index arrays (the same array
    repeated in shared mode).  With

        mu[:, d]  = base_score + sum_r  actor_leaf_means[d][r, leaf(r)]
        var[:, d] = mean_r          actor_leaf_vars[d][r, leaf(r)]
        V         = base_score + sum_r  critic_leaf_means[r, leaf(r)]

    `actor_leaf_means` is initialised from XGBoost's own leaf weights after each
    AWR rebuild and then moved in place by the micro-updates.
    """

    def __init__(
        self,
        action_dim: int,
        max_depth: int,
        n_estimators: int,
        ppo_lr: float,
        awr_beta: float,
        action_low,
        action_high,
        clip_ratio: float = 0.2,
        use_ppo_clip: bool = False,
        obs_dependent_std: bool = True,
        n_ppo_epochs: int = 4,
        critic_lr: float | None = None,
        ent_coef: float = 0.0,
        var_min: float = 1e-3,
        var_max: float | None = None,
        leaf_update_norm: str = "leaf_mean",
        exploration_rho: float = 0.0,
        var_mode: str = "residual",
        exploration_factor: float | None = None,
        carry_variance: bool = False,
        var_lr: float | None = None,
        var_update: str = "log",
        actor_eta: float = 0.05,
        critic_eta: float = 0.05,
        awr_weight_clip: float = 20.0,
        awr_target_ess: float | None = 0.3,
        target_kl: float | None = None,
        grad_clip: float = 10.0,
        shared_tree_structure: bool = True,
        verify_reconstruction: bool = True,
        invert_tree_weights: bool = False,
    ):
        self.action_dim = int(action_dim)
        self.n_estimators = int(n_estimators)
        self.ppo_lr = float(ppo_lr)
        self.critic_lr = float(ppo_lr if critic_lr is None else critic_lr)
        self.awr_beta = float(awr_beta)
        self.clip_ratio = float(clip_ratio)
        self.action_low = np.asarray(action_low, dtype=np.float64)
        self.action_high = np.asarray(action_high, dtype=np.float64)
        # Leaf steps are Adam-normalised, so `ppo_lr` is a *fraction of the
        # action range* per micro-update, not a raw gradient multiplier.
        self.mu_step = self.ppo_lr * (self.action_high - self.action_low)

        self.use_ppo_clip = bool(use_ppo_clip)
        self.obs_dependent_std = bool(obs_dependent_std)
        # A2C uses a single pass: without a trust region extra epochs just let
        # the policy diverge.  PPO needs >= 2 for the clip to ever bind, since
        # on epoch 1 the policy has not moved and ratio == 1 everywhere.
        self.n_ppo_epochs = int(n_ppo_epochs) if self.use_ppo_clip else 1

        self.ent_coef = float(ent_coef)
        # Variance bounds are per action dimension: a T_charging range of [0, 11]
        # and a C_target range of [0, 2] must not share one exploration scale.
        span = self.action_high - self.action_low
        self.var_min = np.full(self.action_dim, float(var_min))
        self.var_max = (
            np.full(self.action_dim, float(var_max)) if var_max is not None
            else (span / 2.0) ** 2
        )
        self.leaf_update_norm = leaf_update_norm
        # Temporal correlation of the exploration noise.  Independent per-step
        # Gaussian noise averages to nothing over a trajectory, so on tasks that
        # need a *sustained* push (MountainCarContinuous needs the car rocked at
        # its resonant frequency for ~100 steps) it never reaches the goal, no
        # matter how large the variance.  An AR(1) noise process
        #     n_t = rho * n_{t-1} + sqrt(1 - rho^2) * N(0, 1)
        # keeps the marginal exactly N(0, 1) - so mu and var keep their meaning
        # and the Gaussian log-prob stays a valid marginal density - while
        # correlating consecutive actions.  This is the same trick as OU noise
        # in DDPG and gSDE in SB3.  rho = 0 recovers independent sampling.
        self.exploration_rho = float(exploration_rho)
        self._noise = None

        # How the per-leaf variance is defined, following Nilsson et al.,
        # "Tree Ensembles for Contextual Bandits" (Alg. 1 line 14, Eq. 9):
        #
        #   'residual'    var = mean_n s^2_n          (current default)
        #   'uncertainty' var = sum_n  s^2_n / c_n    (TEUCB/TETS)
        #
        # The division by the leaf count c_n is the substantive difference.
        # 'residual' measures how noisy the actions in a leaf were, so it decays
        # as the policy becomes deterministic - exploration stops because
        # exploration stopped.  'uncertainty' measures how well the ensemble
        # knows that leaf's mean, so it decays only as that leaf accumulates
        # data, which is the directed-exploration signal TEUCB and TETS use.
        # The paper sums over trees under an independence assumption (Eq. 9);
        # 'residual' averages, since there each round estimates the same
        # quantity rather than contributing an independent component.
        #
        # `exploration_factor` is the paper's nu, applied as nu^2 * var.
        if var_mode not in ("residual", "uncertainty"):
            raise ValueError(f"unknown var_mode {var_mode!r}")
        self.var_mode = var_mode
        # sigma^2(s) = nu^2 * sum_i v_i[l_i(s)], following Eq. 9 of Nilsson et
        # al.'s tree-ensemble bandits, where the per-tree variance terms are
        # summed under an assumption of independence.
        #
        # That assumption is naive here: boosted trees are maximally dependent
        # by construction, since tree i+1 is fit to the residual left by trees
        # 1..i.  The true aggregate is sum_i Var + 2 sum_{i<j} Cov, and the
        # covariance terms are neither small nor of known sign, so the effective
        # number of independent components lies somewhere between 1 and K and is
        # not identified.  nu absorbs it and is tuned per environment.
        #
        # nu = 1/sqrt(K) makes sigma^2 the *average* per-tree variance, and is
        # K-invariant: a value tuned at one ensemble size transfers to another,
        # which matters because K is a headline hyperparameter of the method.
        # Empirically it is also a good default - on LunarLanderContinuous it
        # beat 0.5x and 2x that scale by ~100 return, and on Pendulum the
        # objective was flat across the same range.
        # 'uncertainty' already divides each term by its leaf count, so the
        # summed standard errors are of the same order as Var(mu_hat) and nu = 1
        # is the natural scale.  'residual' sums K undivided stage variances,
        # which over-counts by roughly K, so 1/sqrt(K) is the natural scale.
        if exploration_factor is not None:
            self.exploration_factor = float(exploration_factor)
        elif self.var_mode == "uncertainty":
            self.exploration_factor = 1.0
        else:
            self.exploration_factor = 1.0 / np.sqrt(self.n_estimators)
        # In A2C/PPO the policy standard deviation is a *parameter* optimised
        # for return, not an estimate.  The micro-update treats it that way, but
        # the AWR rebuild overwrites the variance tables with fresh regression
        # residuals, discarding it.  With carry_variance the previous sigma^2 is
        # instead evaluated on the buffer states and projected onto the new leaf
        # partition, so exploration persists across rebuilds.
        self.carry_variance = bool(carry_variance)
        # How sigma^2 is stepped by the on-policy stage.
        #
        #   'additive' scales the step by (var_max - var_min), which is set by
        #   the action range.  Once sigma^2 has annealed well below var_max that
        #   step is comparable to sigma^2 itself - measured on Pendulum, a step
        #   of 0.12 against a current variance of 0.11-0.30 - so the variance can
        #   only oscillate, never converge.
        #
        #   'log' steps log(sigma^2) instead, so a step is a fixed *fraction* of
        #   the current variance and remains well-scaled as it anneals.
        if var_update not in ("additive", "log"):
            raise ValueError(f"unknown var_update {var_update!r}")
        self.var_update = var_update
        # How the leaf-update step is distributed across the K trees of mu,
        # V, and (additive-mode) sigma^2 - all plain sums over trees. Default
        # (False) weights proportionally to each tree's own current leaf
        # magnitude, since a fixed-size step has the largest effect on the
        # final prediction wherever a tree already contributes most to it -
        # see Fix 15. True reverts to the original inverse-magnitude scheme.
        self.invert_tree_weights = bool(invert_tree_weights)
        self.var_lr = float(self.ppo_lr if var_lr is None else var_lr)
        self.awr_weight_clip = float(awr_weight_clip)
        self.awr_target_ess = awr_target_ess
        self.target_kl = target_kl
        self.grad_clip = float(grad_clip)
        self.shared_tree_structure = bool(shared_tree_structure)
        self.verify_reconstruction = bool(verify_reconstruction)

        if self.shared_tree_structure:
            major = int(str(xgb.__version__).split(".")[0])
            if major < 2:
                raise RuntimeError(
                    f"shared_tree_structure requires XGBoost >= 2.0 "
                    f"(multi_strategy='multi_output_tree'); found {xgb.__version__}. "
                    "Pass shared_tree_structure=False for one booster per "
                    "action dimension."
                )

        if not self.obs_dependent_std:
            self.log_std = np.log(0.25 * span).astype(np.float64)

        self.actor_params = {
            "objective": "reg:squarederror",
            "max_depth": max_depth,
            "tree_method": "hist",
            "base_score": 0.0,
            "eta": actor_eta,
            "verbosity": 0,
        }
        self.critic_params = dict(self.actor_params, eta=critic_eta)
        if self.shared_tree_structure:
            # One tree per round, leaves carry an action_dim-vector.
            self.actor_params["multi_strategy"] = "multi_output_tree"

        self.actor_models: list[xgb.Booster] | None = None
        self.critic_model: xgb.Booster | None = None

        self.actor_leaf_means = None    # (D, n_est, L)
        self.actor_leaf_vars = None     # (D, n_est, L)
        self.actor_tree_weights = None  # (D, n_est)
        self.actor_var_tree_weights = None  # (D, n_est) - only used by var_update="additive"
        self.critic_leaf_means = None   # (n_est, L)
        self.critic_tree_weights = None # (n_est,)
        self._n_leaf_slots = 0
        # Adam moments; reset on every macro rebuild since the partition changes.
        self._adam_mu = self._adam_var = self._adam_v = None
        self._return_scale = 1.0

        self.last_diagnostics: dict = {}

    # ------------------------------------------------------------------ #
    # Data plumbing
    # ------------------------------------------------------------------ #

    @staticmethod
    def _dmatrix(X, label=None, weight=None):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return xgb.DMatrix(X, label=label, weight=weight)

    # ------------------------------------------------------------------ #
    # Macro update (AWR)
    # ------------------------------------------------------------------ #

    def train_macro_awr(self, states, actions, advantages, returns):
        """Rebuild both ensembles from the replay buffer with AWR."""
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float64).reshape(-1, self.action_dim)
        returns = np.asarray(returns, dtype=np.float64).ravel()
        adv = np.asarray(advantages, dtype=np.float64).ravel()

        # AWR (Peng et al.) uses A = R - V(s) under the *current* value
        # function.  The advantages stored in the replay buffer were computed by
        # whatever critic existed when the sample was collected - up to
        # `awr_buffer_size` steps ago - so recomputing them here is both cheaper
        # than it looks and considerably less stale.
        if self.critic_model is not None:
            adv = returns - self.predict_value(states)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        beta = self.awr_beta
        if self.awr_target_ess is not None:
            beta = _target_ess_beta(adv, self.awr_target_ess, self.awr_weight_clip)
        # Subtract the max before exponentiating for numerical stability; the
        # common factor cancels because XGBoost normalises by the weight sum.
        weights = np.exp(np.clip(adv / beta, -50.0, 50.0) - np.max(adv / beta))
        weights = np.clip(weights, 0.0, self.awr_weight_clip)
        weights = weights / max(weights.mean(), 1e-12)

        n_est = self.n_estimators
        d_eval = self._dmatrix(states)

        # Snapshot sigma^2(s) under the OLD partition before it is replaced.
        prev_var = None
        if self.carry_variance and self.actor_models is not None:
            prev_ids, _ = self._leaf_ids(states)
            prev_var = self._forward(prev_ids)[1]          # (N, D)

        # ---- critic: unweighted regression onto the returns ------------------
        self.critic_model = xgb.train(
            self.critic_params,
            self._dmatrix(states, label=returns),
            num_boost_round=n_est,
        )

        # ---- actor: AWR-weighted regression onto the actions ------------------
        if self.shared_tree_structure:
            self.actor_models = [
                xgb.train(
                    self.actor_params,
                    self._dmatrix(states, label=actions, weight=weights),
                    num_boost_round=n_est,
                )
            ]
        else:
            self.actor_models = [
                xgb.train(
                    self.actor_params,
                    self._dmatrix(states, label=actions[:, d], weight=weights),
                    num_boost_round=n_est,
                )
                for d in range(self.action_dim)
            ]

        # ---- parse dumps and size the dense arrays ---------------------------
        # multi_output_tree dumps do not support with_stats; we compute our own
        # counts and variances from the data anyway.
        actor_dumps = [
            m.get_dump(with_stats=False, dump_format="json") for m in self.actor_models
        ]
        for k, dump in enumerate(actor_dumps):
            if len(dump) != n_est:
                raise RuntimeError(
                    f"actor booster {k} produced {len(dump)} trees, expected "
                    f"{n_est}. With shared_tree_structure=True this usually "
                    "means multi_strategy='multi_output_tree' was not applied "
                    "and XGBoost fell back to one output per tree."
                )
        actor_leaves = [[get_leafs(json.loads(t)) for t in d] for d in actor_dumps]
        max_lid = max(
            (max(t) for leaves in actor_leaves for t in leaves if t), default=0
        )

        dump_c = self.critic_model.get_dump(with_stats=False, dump_format="json")
        if len(dump_c) != n_est:
            raise RuntimeError(
                f"critic produced {len(dump_c)} trees, expected {n_est}."
            )
        critic_leaves = [get_leafs(json.loads(t)) for t in dump_c]
        max_lid = max(max_lid, max((max(t) for t in critic_leaves if t), default=0))

        L = int(max_lid) + 1
        self._n_leaf_slots = L
        base = float(self.actor_params["base_score"])

        # ---- actor leaf means, variances, tree weights -----------------------
        if self.shared_tree_structure:
            native = _native_leaf_array_multi(
                actor_leaves[0], n_est, max_lid, self.action_dim
            )                                            # (D, n_est, L)
            shared_ids = self._as_leaf_ids(
                self.actor_models[0].predict(d_eval, pred_leaf=True), n_est
            )
            if shared_ids.shape[1] != n_est:
                raise RuntimeError(
                    f"pred_leaf returned {shared_ids.shape[1]} columns, expected "
                    f"{n_est}: the tree structure is not shared across outputs."
                )
            leaf_ids_per_dim = [shared_ids] * self.action_dim
        else:
            native = np.stack([
                _native_leaf_array(actor_leaves[d], n_est, max_lid)
                for d in range(self.action_dim)
            ])                                           # (D, n_est, L)
            leaf_ids_per_dim = [
                self._as_leaf_ids(m.predict(d_eval, pred_leaf=True), n_est)
                for m in self.actor_models
            ]

        self.actor_leaf_means = np.zeros((self.action_dim, n_est, L))
        self.actor_leaf_vars = (
            np.ones((self.action_dim, n_est, L)) * self.var_min[:, None, None]
        )
        self.actor_tree_weights = np.zeros((self.action_dim, n_est))
        self.actor_var_tree_weights = np.zeros((self.action_dim, n_est))

        if self.verify_reconstruction:
            ref = np.asarray(
                self.actor_models[0].predict(d_eval), dtype=np.float64
            ).reshape(len(states), -1) if self.shared_tree_structure else np.stack(
                [np.asarray(m.predict(d_eval), dtype=np.float64)
                 for m in self.actor_models], axis=1
            )
            if ref.shape[1] != self.action_dim:
                raise RuntimeError(
                    f"actor predict returned {ref.shape[1]} outputs, expected "
                    f"{self.action_dim}."
                )

        rounds = np.arange(n_est)
        for d in range(self.action_dim):
            leaf_ids = leaf_ids_per_dim[d]
            pred = base + native[d][rounds, leaf_ids].sum(axis=1)
            if self.verify_reconstruction:
                err = np.max(np.abs(pred - ref[:, d]))
                if err > 1e-3 * max(1.0, np.max(np.abs(ref[:, d]))):
                    raise RuntimeError(
                        f"actor dim {d} leaf reconstruction mismatch "
                        f"(max err {err:.3e}); the dump layout does not match "
                        "the assumed one."
                    )

            # Stage residuals, following the conditional-contribution view of
            # TEUCB/TETS: leaf l of tree i estimates the mean of
            #     z_i = a - mu_{1..i-1}(s),
            # the part of the target still unexplained when tree i is fit, not
            # the residual of the finished ensemble.  Using the final residual
            # for every tree makes the K leaf variances redundant estimates of
            # one quantity; the stage form gives a genuine per-tree
            # decomposition whose terms decrease along the boosting sequence.
            contrib = native[d][rounds, leaf_ids]                  # (N, n_est)
            stage_pred = base + np.cumsum(contrib, axis=1) - contrib
            Z = actions[:, d][:, None] - stage_pred                # (N, n_est)
            cnt, _, var = _leaf_stats(leaf_ids, Z, n_est, L)

            self.actor_leaf_means[d] = native[d]
            # Per-leaf conditional variance of the residual.  Averaged (not
            # summed) across rounds in `_forward`: each round is a different
            # partition of the same residual, so the rounds are estimates of the
            # same quantity, not independent contributions.
            if self.var_mode == "uncertainty":
                leaf_var = var / np.maximum(cnt, 1.0)      # s^2_n / c_n
            else:
                leaf_var = var
            if prev_var is not None:
                # Project the previous sigma^2 onto the new partition: each new
                # leaf takes the mean of the old sigma^2 over the samples that
                # land in it.  sigma^2 aggregates by averaging over trees, so
                # the projected per-leaf value is the target value itself.
                _, proj, _ = _leaf_stats(leaf_ids, prev_var[:, d], n_est, L)
                # prev_var is the aggregate nu^2 * sum_i v_i; invert to per-leaf.
                proj = proj / (self.exploration_factor ** 2 * n_est)
                leaf_var = np.where(cnt > 0, proj, leaf_var)
            self.actor_leaf_vars[d] = np.where(
                cnt > 1, np.maximum(leaf_var, self.var_min[d]), self.var_min[d]
            )
            self.actor_tree_weights[d] = _magnitude_weights(
                native[d], cnt, invert=self.invert_tree_weights
            )
            # Same construction as actor_tree_weights, but from this tree's own
            # variance leaves rather than the mean leaves - see Fix 15/16.  Used
            # only by the "additive" var_update path; the default "log" path
            # doesn't need any per-tree weighting (see train_micro).
            self.actor_var_tree_weights[d] = _magnitude_weights(
                self.actor_leaf_vars[d], cnt, invert=self.invert_tree_weights
            )

        # ---- critic leaf means / tree weights --------------------------------
        native_c = _native_leaf_array(critic_leaves, n_est, max_lid)
        leaf_ids_c = self._as_leaf_ids(
            self.critic_model.predict(d_eval, pred_leaf=True), n_est
        )
        pred_c = float(self.critic_params["base_score"]) + native_c[
            np.arange(n_est), leaf_ids_c
        ].sum(axis=1)
        if self.verify_reconstruction:
            ref_c = self.critic_model.predict(d_eval).astype(np.float64)
            err = np.max(np.abs(pred_c - ref_c))
            if err > 1e-3 * max(1.0, np.max(np.abs(ref_c))):
                raise RuntimeError(
                    f"critic leaf reconstruction mismatch (max err {err:.3e})."
                )
        cnt_c, _, _ = _leaf_stats(leaf_ids_c, returns - pred_c, n_est, L)
        self.critic_leaf_means = native_c
        self.critic_tree_weights = _magnitude_weights(
            native_c, cnt_c, invert=self.invert_tree_weights
        )

        # Adam moments are tied to the leaf partition, which has just changed.
        self._adam_mu = _LeafAdam(self.actor_leaf_means.shape)
        self._adam_var = _LeafAdam(self.actor_leaf_vars.shape)
        self._adam_v = _LeafAdam(self.critic_leaf_means.shape)
        self._return_scale = float(returns.std() + 1e-8)

        self.last_diagnostics["awr/mu_outside_box"] = float(
            np.mean(
                (self.predict(states[:2000]) != self._forward(
                    self._leaf_ids(states[:2000])[0])[0]).any(axis=1)
            )
        )
        self.last_diagnostics["awr/beta"] = float(beta)
        self.last_diagnostics["awr/weight_mean"] = float(weights.mean())
        self.last_diagnostics["awr/weight_max"] = float(weights.max())
        self.last_diagnostics["awr/effective_sample_frac"] = float(
            weights.sum() ** 2 / (len(weights) * (weights ** 2).sum() + 1e-12)
        )
        self.last_diagnostics["awr/n_samples"] = int(len(states))

    # ------------------------------------------------------------------ #
    # Forward pass
    # ------------------------------------------------------------------ #

    @staticmethod
    def _as_leaf_ids(raw, n_est):
        return np.atleast_2d(raw).astype(np.int64).reshape(-1, n_est)

    def _leaf_ids(self, states):
        """Per-dimension leaf assignments, (N, n_estimators) each.

        In shared mode all dimensions index the *same* tree structure, so one
        `pred_leaf` call is made and the resulting array is reused for every
        dimension.  Everything downstream is agnostic to which mode is active.
        """
        d = self._dmatrix(states)
        n_est = self.n_estimators

        if self.shared_tree_structure:
            shared = self._as_leaf_ids(
                self.actor_models[0].predict(d, pred_leaf=True), n_est
            )
            np.clip(shared, 0, self._n_leaf_slots - 1, out=shared)
            actor_ids = [shared] * self.action_dim
        else:
            actor_ids = [
                self._as_leaf_ids(m.predict(d, pred_leaf=True), n_est)
                for m in self.actor_models
            ]
            for a in actor_ids:
                np.clip(a, 0, self._n_leaf_slots - 1, out=a)

        critic_ids = self._as_leaf_ids(
            self.critic_model.predict(d, pred_leaf=True), n_est
        )
        np.clip(critic_ids, 0, self._n_leaf_slots - 1, out=critic_ids)
        return actor_ids, critic_ids

    def _forward(self, actor_ids):
        """mu (N, D) and var (N, D) from cached leaf assignments."""
        n_est = self.n_estimators
        rounds = np.arange(n_est)
        n = actor_ids[0].shape[0]

        mu = np.empty((n, self.action_dim))
        for d in range(self.action_dim):
            mu[:, d] = self.actor_leaf_means[d][rounds, actor_ids[d]].sum(axis=1)
        mu += float(self.actor_params["base_score"])

        if self.obs_dependent_std:
            var = np.empty((n, self.action_dim))
            for d in range(self.action_dim):
                var[:, d] = self.actor_leaf_vars[d][rounds, actor_ids[d]].sum(axis=1)
            var *= self.exploration_factor ** 2
            np.clip(var, self.var_min[None, :], self.var_max[None, :], out=var)
        else:
            var = np.broadcast_to(
                np.exp(2.0 * self.log_std), (n, self.action_dim)
            ).copy()
        return mu, var

    def _value(self, critic_ids):
        rounds = np.arange(self.n_estimators)
        return (
            float(self.critic_params["base_score"])
            + self.critic_leaf_means[rounds, critic_ids].sum(axis=1)
        )

    @staticmethod
    def _log_prob(actions, mu, var):
        return (
            -0.5 * (((actions - mu) ** 2) / var + np.log(var) + np.log(2.0 * np.pi))
        ).sum(axis=1)

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def pick_action(self, obs):
        """Sample an action.  Returns (clipped, raw, log_prob)."""
        obs = np.asarray(obs, dtype=np.float32)
        n = len(obs)

        if self.actor_models is None:
            raw = np.random.uniform(
                self.action_low, self.action_high, size=(n, self.action_dim)
            )
            lp = np.full(n, -np.sum(np.log(self.action_high - self.action_low)))
            return np.clip(raw, self.action_low, self.action_high), raw, lp

        actor_ids, _ = self._leaf_ids(obs)
        mu, var = self._forward(actor_ids)
        raw = mu + np.sqrt(var) * self._draw_noise(mu.shape)
        lp = self._log_prob(raw, mu, var)
        return np.clip(raw, self.action_low, self.action_high), raw, lp

    def _draw_noise(self, shape):
        if self.exploration_rho <= 0.0:
            return np.random.normal(size=shape)
        if self._noise is None or self._noise.shape != shape:
            self._noise = np.random.normal(size=shape)
        rho = self.exploration_rho
        self._noise = (rho * self._noise
                       + np.sqrt(1.0 - rho ** 2) * np.random.normal(size=shape))
        return self._noise

    def reset_noise(self, mask):
        """Redraw the noise state for environments that just terminated."""
        if self._noise is None or self.exploration_rho <= 0.0:
            return
        mask = np.asarray(mask, dtype=bool)
        if mask.any():
            self._noise[mask] = np.random.normal(size=(int(mask.sum()),
                                                       self._noise.shape[1]))

    def predict(self, obs, deterministic=True):
        """Greedy (mean) action, for evaluation."""
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs[None, :]
        if self.actor_models is None:
            return np.clip(
                np.random.uniform(self.action_low, self.action_high,
                                  size=(len(obs), self.action_dim)),
                self.action_low, self.action_high,
            )
        if not deterministic:
            return self.pick_action(obs)[0]
        actor_ids, _ = self._leaf_ids(obs)
        mu, _ = self._forward(actor_ids)
        return np.clip(mu, self.action_low, self.action_high)

    def predict_value(self, obs):
        if self.critic_model is None:
            return np.zeros(len(obs))
        _, critic_ids = self._leaf_ids(np.asarray(obs, dtype=np.float32))
        return self._value(critic_ids)

    # ------------------------------------------------------------------ #
    # Micro update (A2C / PPO leaf-value refit) + critic leaf refit
    # ------------------------------------------------------------------ #

    def train_micro(self, states, actions, old_log_probs, advantages, returns):
        if self.actor_models is None or self._adam_mu is None:
            return {}

        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float64).reshape(-1, self.action_dim)
        old_lp = np.asarray(old_log_probs, dtype=np.float64).ravel()
        adv = np.asarray(advantages, dtype=np.float64).ravel()
        returns = np.asarray(returns, dtype=np.float64).ravel()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n, n_est, L = len(states), self.n_estimators, self._n_leaf_slots
        # Keep the total step size epoch-invariant.
        lr_scale = 1.0 / self.n_ppo_epochs

        # Tree *structure* is frozen between macro-updates, so leaf assignments
        # can be computed once and reused across epochs.
        actor_ids, critic_ids = self._leaf_ids(states)
        mu_before, _ = self._forward(actor_ids)

        clip_frac, kl, epochs_run = 0.0, 0.0, 0
        for _epoch in range(self.n_ppo_epochs):
            epochs_run += 1
            mu, var = self._forward(actor_ids)

            if self.use_ppo_clip:
                new_lp = self._log_prob(actions, mu, var)
                ratio = np.exp(np.clip(new_lp - old_lp, -20.0, 20.0))
                kl = float(np.mean(old_lp - new_lp))
                # Real clipped surrogate: grad L = ratio * A * dlog pi where the
                # clip does not bind, 0 where it does.
                binding = ((ratio > 1.0 + self.clip_ratio) & (adv > 0)) | (
                    (ratio < 1.0 - self.clip_ratio) & (adv < 0)
                )
                clip_frac = float(binding.mean())
                eff = ratio * adv * (~binding)
                if self.target_kl is not None and kl > self.target_kl:
                    break
            else:
                eff = adv                                     # A2C
            eff2d = eff[:, None]

            grad_mu = np.clip(
                eff2d * (actions - mu) / var, -self.grad_clip, self.grad_clip
            )
            grad_var = 0.5 * eff2d * (
                ((actions - mu) ** 2) / var ** 2 - 1.0 / var
            )
            if self.ent_coef:
                grad_var = grad_var + self.ent_coef * 0.5 / var
            grad_var = np.clip(grad_var, -self.grad_clip, self.grad_clip)

            # Aggregate raw per-leaf gradients, Adam-normalise them, then apply
            # the tree weights (which sum to 1) so the *ensemble output* moves by
            # about `mu_step` per micro-update regardless of gradient scale.
            g_mu = np.stack([
                _scatter_leaf_update(
                    actor_ids[d], grad_mu[:, d], np.ones(n_est),
                    n_est, L, self.leaf_update_norm,
                )
                for d in range(self.action_dim)
            ])
            u_mu = self._adam_mu.step(g_mu)
            for d in range(self.action_dim):
                # mu = sum_r m[r]
                self.actor_leaf_means[d] += (
                    (lr_scale * self.mu_step[d])
                    * self.actor_tree_weights[d][:, None] * u_mu[d]
                )

            if self.obs_dependent_std:
                g_var = np.stack([
                    _scatter_leaf_update(
                        actor_ids[d], grad_var[:, d], np.ones(n_est),
                        n_est, L, self.leaf_update_norm,
                    )
                    for d in range(self.action_dim)
                ])
                u_var = self._adam_var.step(g_var)
                if self.var_update == "log":
                    # Multiplicative: every leaf moves by the same fraction, so
                    # the aggregate sigma^2 = nu^2 * sum_i v_i moves by that
                    # fraction too, with no K or nu correction needed - and no
                    # tree weighting either, since introducing one here would
                    # make different trees' leaves move by different fractions,
                    # forfeiting exactly this invariant.
                    self.actor_leaf_vars *= np.exp(
                        np.clip(lr_scale * self.var_lr * u_var, -0.5, 0.5))
                else:
                    # sigma^2 = nu^2 * sum_i v_i is a sum over K trees, so a
                    # step needs distributing across them; weight each tree's
                    # share by actor_var_tree_weights (inversely proportional
                    # to that tree's own variance-leaf magnitude, summing to
                    # 1) rather than a uniform 1/n_est - same rationale as the
                    # mean update, but computed from the variance leaves
                    # themselves rather than borrowed from the mean's decay
                    # pattern. Weights summing to 1 replace the old /n_est.
                    var_step = (lr_scale * self.var_lr
                                * (self.var_max - self.var_min)
                                / (self.exploration_factor ** 2))
                    for d in range(self.action_dim):
                        self.actor_leaf_vars[d] += (
                            var_step[d] * self.actor_var_tree_weights[d][:, None]
                            * u_var[d]
                        )
            if self.obs_dependent_std:
                np.clip(self.actor_leaf_vars,
                        self.var_min[:, None, None], self.var_max[:, None, None],
                        out=self.actor_leaf_vars)
            else:
                g = np.clip(
                    (eff2d * (((actions - mu) ** 2) / var - 1.0)).mean(axis=0),
                    -0.5, 0.5,
                )
                # log-space step, so a plain lr is the right scale here
                self.log_std = np.clip(
                    self.log_std + lr_scale * self.ppo_lr * g,
                    0.5 * np.log(self.var_min), 0.5 * np.log(self.var_max),
                )  # bounds are per-dimension arrays

        # ---- critic leaf refit (Algorithm 1 line 9) --------------------------
        v_before = self._value(critic_ids)
        td_error = returns - v_before
        g_v = _scatter_leaf_update(
            critic_ids, td_error, np.ones(n_est), n_est, L, self.leaf_update_norm,
        )
        self.critic_leaf_means += (
            (self.critic_lr * self._return_scale)
            * self.critic_tree_weights[:, None] * self._adam_v.step(g_v)
        )
        v_after = self._value(critic_ids)

        var_ret = float(np.var(returns))
        diag = {
            "train/micro_epochs": epochs_run,
            "train/clip_fraction": clip_frac,
            "train/approx_kl": kl,
            "train/micro_mu_delta": float(
                np.abs(self._forward(actor_ids)[0] - mu_before).mean()
            ),
            "train/critic_lr_step": float(np.abs(v_after - v_before).mean()),
            "train/value_loss": float(np.mean(td_error ** 2)),
            "train/explained_variance": float(
                1.0 - np.var(returns - v_after) / (var_ret + 1e-12)
            ) if var_ret > 0 else 0.0,
        }
        mu, var = self._forward(actor_ids)
        for d in range(self.action_dim):
            diag[f"policy/std_dim{d}"] = float(np.sqrt(var[:, d]).mean())
            diag[f"policy/mu_dim{d}"] = float(mu[:, d].mean())
            # Spread of the policy mean ACROSS STATES.  mu_dim* alone cannot
            # distinguish a timid policy (mu ~ 0 everywhere) from a symmetric
            # state-dependent one (mu = +/-a_max, averaging to 0).  If this is
            # small relative to the action range, the actor is not discriminating
            # between states at all.
            diag[f"policy/mu_spread_dim{d}"] = float(mu[:, d].std())
            diag[f"policy/var_at_ceiling_dim{d}"] = float(
                np.mean(var[:, d] >= self.var_max[d] - 1e-9)
            )
        self.last_diagnostics.update(diag)
        return diag

    # Backwards-compatible alias.
    def train_micro_ppo(self, states, actions, old_log_probs, advantages, returns):
        return self.train_micro(states, actions, old_log_probs, advantages, returns)


# --------------------------------------------------------------------------- #
# SB3-compatible wrapper
# --------------------------------------------------------------------------- #

class Hybrid_XGB:
    """Training loop: rollout -> micro-update -> periodic AWR macro-update."""

    def __init__(
        self,
        env,
        awr_update_freq: int = 10000,
        awr_buffer_size: int = 15000,
        awr_min_samples: int = 500,
        n_steps: int = 256,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        ppo_lr: float = 0.03,
        critic_lr: float = 0.03,
        awr_beta: float = 0.5,
        awr_target_ess: float | None = 0.3,
        max_depth: int = 6,
        n_estimators: int = 300,
        actor_eta: float = 0.05,
        critic_eta: float = 0.05,
        use_ppo_clip: bool = False,
        obs_dependent_std: bool = True,
        n_ppo_epochs: int = 4,
        ent_coef: float = 0.0,
        target_kl: float | None = None,
        leaf_update_norm: str = "leaf_mean",
        exploration_rho: float = 0.0,
        var_mode: str = "residual",
        exploration_factor: float | None = None,
        carry_variance: bool = False,
        var_lr: float | None = None,
        var_update: str = "log",
        var_min: float = 1e-3,
        shared_tree_structure: bool = True,
        verify_reconstruction: bool = True,
        invert_tree_weights: bool = False,
        **kwargs,
    ):
        self.env = env
        self.n_envs = getattr(env, "num_envs", 1)
        self.n_steps = int(n_steps)
        self.awr_update_freq = int(awr_update_freq)
        self.awr_min_samples = int(awr_min_samples)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)

        if hasattr(env.action_space, "n"):
            raise NotImplementedError(
                "Hybrid_XGB models a diagonal Gaussian policy and supports "
                "Box action spaces only."
            )
        self.action_dim = int(env.action_space.shape[0])
        self.action_low = np.asarray(env.action_space.low, dtype=np.float64)
        self.action_high = np.asarray(env.action_space.high, dtype=np.float64)
        self.obs_dim = tuple(env.observation_space.shape)

        self.engine = XGBoostTreeEngine(
            self.action_dim, max_depth, n_estimators, ppo_lr, awr_beta,
            self.action_low, self.action_high,
            use_ppo_clip=use_ppo_clip,
            obs_dependent_std=obs_dependent_std,
            n_ppo_epochs=n_ppo_epochs,
            critic_lr=critic_lr,
            ent_coef=ent_coef,
            leaf_update_norm=leaf_update_norm,
            exploration_rho=exploration_rho,
            var_mode=var_mode,
            exploration_factor=exploration_factor,
            carry_variance=carry_variance,
            var_lr=var_lr,
            var_update=var_update,
            var_min=var_min,
            awr_target_ess=awr_target_ess,
            actor_eta=actor_eta,
            critic_eta=critic_eta,
            target_kl=target_kl,
            shared_tree_structure=shared_tree_structure,
            verify_reconstruction=verify_reconstruction,
            invert_tree_weights=invert_tree_weights,
        )

        try:
            from stable_baselines3.common.buffers import RolloutBuffer

            self.rollout_buffer = RolloutBuffer(
                self.n_steps, env.observation_space, env.action_space,
                device="cpu", gamma=gamma, gae_lambda=gae_lambda, n_envs=self.n_envs,
            )
            self._sb3 = True
        except Exception:
            self.rollout_buffer = _SimpleRolloutBuffer(
                self.n_steps, self.obs_dim, self.action_dim,
                self.n_envs, gamma, gae_lambda,
            )
            self._sb3 = False

        # NOTE: the AWR target is the *executed* (clipped) action.  Regressing
        # on the raw Gaussian sample lets the policy mean drift outside the
        # action box, where every sample clips to the same executed action and
        # the gradient signal disappears.
        self.awr_buffer = {
            "obs": np.zeros((awr_buffer_size,) + self.obs_dim, dtype=np.float32),
            "actions": np.zeros((awr_buffer_size, self.action_dim), dtype=np.float32),
            "returns": np.zeros(awr_buffer_size, dtype=np.float32),
            "advantages": np.zeros(awr_buffer_size, dtype=np.float32),
            "ptr": 0, "size": 0, "max_size": int(awr_buffer_size),
        }

        # Executed (clipped) actions, kept alongside the raw Gaussian samples
        # that the rollout buffer stores for the PPO ratio.
        self._clipped_actions = np.zeros(
            (self.n_steps, self.n_envs, self.action_dim), dtype=np.float32
        )
        self.num_timesteps = 0
        self._steps_since_awr = 0
        self.tensorboard_log = kwargs.get("tensorboard_log", None)
        self.verbose = kwargs.get("verbose", 1)
        self.logger = None
        self.ep_info_buffer = deque(maxlen=100)

    def set_logger(self, logger):
        self.logger = logger

    # ------------------------------------------------------------------ #

    def _store_in_awr_buffer(self, obs, actions, returns, advantages):
        n, buf = len(obs), self.awr_buffer
        cap = buf["max_size"]
        idx = (np.arange(n) + buf["ptr"]) % cap
        buf["obs"][idx] = obs
        buf["actions"][idx] = actions
        buf["returns"][idx] = returns
        buf["advantages"][idx] = advantages
        buf["ptr"] = int((buf["ptr"] + n) % cap)
        buf["size"] = min(buf["size"] + n, cap)

    def _awr_view(self):
        """Valid samples in insertion order (handles ring-buffer wraparound)."""
        buf = self.awr_buffer
        idx = np.arange(buf["ptr"] - buf["size"], buf["ptr"]) % buf["max_size"]
        return (buf["obs"][idx], buf["actions"][idx],
                buf["advantages"][idx], buf["returns"][idx])

    # ------------------------------------------------------------------ #

    def learn(self, total_timesteps, callback=None, log_interval=1,
              tb_log_name="TERL_XGB", **kwargs):
        if self.logger is None and self._sb3:
            from stable_baselines3.common.utils import configure_logger

            self.logger = configure_logger(
                self.verbose, self.tensorboard_log, tb_log_name,
                reset_num_timesteps=True,
            )

        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        obs = np.asarray(obs, dtype=np.float32)
        episode_starts = np.ones(self.n_envs, dtype=bool)
        iteration, start_time = 0, time.time()

        # Echo the settings that actually reached the engine.  Several of these
        # were historically dropped between the CLI and the constructor, so a
        # run could silently use defaults while the shell script said otherwise.
        e = self.engine
        cfg = (f"n_estimators={e.n_estimators} max_depth={e.actor_params['max_depth']} "
               f"gamma={self.gamma} ppo_lr={e.ppo_lr} critic_lr={e.critic_lr} "
               f"exploration_rho={e.exploration_rho} nu={e.exploration_factor:.4f} "
               f"var_mode={e.var_mode} var_update={e.var_update} "
               f"invert_tree_weights={e.invert_tree_weights} "
               f"use_ppo_clip={e.use_ppo_clip} "
               f"obs_dependent_std={e.obs_dependent_std} "
               f"awr_target_ess={e.awr_target_ess} awr_update_freq={self.awr_update_freq} "
               f"awr_buffer_size={self.awr_buffer['max_size']} n_steps={self.n_steps} "
               f"n_envs={self.n_envs}")
        if self.verbose:
            print("Starting TERL / XGBoost training loop...")
            print(f"  effective config: {cfg}")
        self._effective_config = cfg

        while self.num_timesteps < total_timesteps:
            self.rollout_buffer.reset()
            iteration += 1

            # ---- 1. collect ------------------------------------------------
            for _step in range(self.n_steps):
                actions, raw_actions, log_probs = self.engine.pick_action(obs)
                self._clipped_actions[_step] = actions
                values = self.engine.predict_value(obs)
                next_obs, rewards, dones, infos = self.env.step(actions)
                next_obs = np.asarray(next_obs, dtype=np.float32)
                rewards = np.asarray(rewards, dtype=np.float32).copy()

                for i, info in enumerate(infos):
                    if "episode" in info:
                        self.ep_info_buffer.append(info["episode"])
                    # Bootstrap value for time-limit truncations, as SB3's PPO
                    # does; without this GAE treats a truncation as terminal.
                    if (
                        dones[i]
                        and info.get("TimeLimit.truncated", False)
                        and "terminal_observation" in info
                    ):
                        term_obs = np.asarray(
                            info["terminal_observation"], dtype=np.float32
                        )[None, :]
                        rewards[i] += self.gamma * float(
                            self.engine.predict_value(term_obs)[0]
                        )

                self._add_to_buffer(
                    obs, raw_actions, rewards, episode_starts, values, log_probs
                )

                obs = next_obs
                episode_starts = np.asarray(dones, dtype=bool)
                self.engine.reset_noise(episode_starts)
                self.num_timesteps += self.n_envs
                self._steps_since_awr += self.n_envs

            # ---- 2. returns / advantages -----------------------------------
            last_values = self.engine.predict_value(obs)
            self._compute_returns(last_values, episode_starts)

            flat_obs = self._flat("observations", self.obs_dim)
            flat_actions = self._flat("actions", (self.action_dim,))
            flat_old_lp = self._flat("log_probs", ())
            flat_returns = self._flat("returns", ())
            flat_adv = self._flat("advantages", ())

            # The AWR target is the executed action.  Regressing on the raw
            # Gaussian sample biases the fit outward: samples beyond the box all
            # execute as the same boundary action, so the regression is pulled
            # toward a mean that no longer corresponds to any behaviour, and the
            # policy drifts into a region where every action clips identically
            # and the gradient signal vanishes.
            flat_exec = self._clipped_actions.reshape(-1, self.action_dim)
            self._store_in_awr_buffer(
                flat_obs, flat_exec, flat_returns, flat_adv
            )

            # ---- 3. micro-update (A2C / PPO leaves + critic leaves) ---------
            self.last_diagnostics = diag = self.engine.train_micro(
                flat_obs, flat_actions, flat_old_lp, flat_adv, flat_returns
            ) or {}

            # ---- 4. macro-update (AWR rebuild) -----------------------------
            if (
                self._steps_since_awr >= self.awr_update_freq
                and self.awr_buffer["size"] > self.awr_min_samples
            ):
                self._steps_since_awr = 0
                if self.verbose:
                    print(
                        f"[t={self.num_timesteps}] AWR macro-update "
                        f"({self.awr_buffer['size']} samples)"
                    )
                b_obs, b_act, b_adv, b_ret = self._awr_view()
                # Measure the macro-update on the same scale as the micro-update:
                # mean |change in the policy mean| over a fixed probe batch.
                probe = b_obs[np.linspace(0, len(b_obs) - 1, min(512, len(b_obs)),
                                          dtype=int)]
                mu_before = (
                    self.engine.predict(probe, deterministic=True)
                    if self.engine.actor_models is not None else None
                )
                self.engine.train_macro_awr(b_obs, b_act, b_adv, b_ret)
                if mu_before is not None:
                    mu_after = self.engine.predict(probe, deterministic=True)
                    diag["train/macro_mu_delta"] = float(
                        np.abs(mu_after - mu_before).mean()
                    )
                    self.engine.last_diagnostics.update(diag)

            # ---- 5. logging -------------------------------------------------
            if self.logger and iteration % log_interval == 0:
                elapsed = max(time.time() - start_time, 1e-9)
                # Logged every dump so the effective config is recoverable from
                # the event file alone, without the stdout log.
                self.logger.record("config/exploration_rho",
                                   float(self.engine.exploration_rho))
                self.logger.record("config/n_estimators", int(self.engine.n_estimators))
                self.logger.record("config/max_depth",
                                   int(self.engine.actor_params['max_depth']))
                self.logger.record("config/gamma", float(self.gamma))
                self.logger.record("config/ppo_lr", float(self.engine.ppo_lr))
                self.logger.record("time/iterations", iteration)
                self.logger.record("time/fps", int(self.num_timesteps / elapsed))
                self.logger.record("time/total_timesteps", self.num_timesteps)
                self.logger.record(
                    "trees/actor_count",
                    (self.engine.n_estimators
                     if self.engine.shared_tree_structure
                     else self.engine.n_estimators * self.action_dim)
                    if self.engine.actor_models else 0,
                )
                self.logger.record(
                    "trees/critic_count",
                    self.engine.n_estimators if self.engine.critic_model else 0,
                )
                for k, v in {**self.engine.last_diagnostics, **diag}.items():
                    self.logger.record(k, v)
                if self.ep_info_buffer:
                    self.logger.record(
                        "rollout/ep_rew_mean",
                        float(np.mean([e["r"] for e in self.ep_info_buffer])),
                    )
                    self.logger.record(
                        "rollout/ep_len_mean",
                        float(np.mean([e["l"] for e in self.ep_info_buffer])),
                    )
                self.logger.dump(step=self.num_timesteps)

        return self

    # ------------------------------------------------------------------ #

    def _add_to_buffer(self, obs, raw_actions, rewards, episode_starts, values, log_probs):
        if self._sb3:
            import torch as th

            self.rollout_buffer.add(
                obs, raw_actions, rewards, episode_starts,
                th.as_tensor(np.asarray(values, dtype=np.float32)),
                th.as_tensor(np.asarray(log_probs, dtype=np.float32)),
            )
        else:
            self.rollout_buffer.add(
                obs, raw_actions, rewards, episode_starts, values, log_probs
            )

    def _compute_returns(self, last_values, dones):
        if self._sb3:
            import torch as th

            self.rollout_buffer.compute_returns_and_advantage(
                last_values=th.as_tensor(np.asarray(last_values, dtype=np.float32)),
                dones=np.asarray(dones, dtype=bool),
            )
        else:
            self.rollout_buffer.compute_returns_and_advantage(last_values, dones)

    def _flat(self, name, trailing):
        arr = np.asarray(getattr(self.rollout_buffer, name))
        return arr.reshape((-1,) + tuple(trailing))

    # ------------------------------------------------------------------ #

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        return self.engine.predict(obs, deterministic=deterministic), None

    def save(self, path):
        blob = {
            "actor": [m.save_raw() for m in (self.engine.actor_models or [])],
            "critic": self.engine.critic_model.save_raw()
            if self.engine.critic_model else None,
            "actor_leaf_means": self.engine.actor_leaf_means,
            "actor_leaf_vars": self.engine.actor_leaf_vars,
            "actor_tree_weights": self.engine.actor_tree_weights,
            "actor_var_tree_weights": self.engine.actor_var_tree_weights,
            "critic_leaf_means": self.engine.critic_leaf_means,
            "critic_tree_weights": self.engine.critic_tree_weights,
            "n_leaf_slots": self.engine._n_leaf_slots,
            "shared_tree_structure": self.engine.shared_tree_structure,
            "invert_tree_weights": self.engine.invert_tree_weights,
            "log_std": getattr(self.engine, "log_std", None),
            "num_timesteps": self.num_timesteps,
        }
        with open(path, "wb") as fh:
            pickle.dump(blob, fh)

    def load_weights(self, path):
        with open(path, "rb") as fh:
            blob = pickle.load(fh)
        self.engine.actor_models = []
        for raw in blob["actor"]:
            b = xgb.Booster()
            b.load_model(bytearray(raw))
            self.engine.actor_models.append(b)
        if blob["critic"] is not None:
            b = xgb.Booster()
            b.load_model(bytearray(blob["critic"]))
            self.engine.critic_model = b
        self.engine.actor_leaf_means = blob["actor_leaf_means"]
        self.engine.actor_leaf_vars = blob["actor_leaf_vars"]
        self.engine.actor_tree_weights = blob["actor_tree_weights"]
        # .get() for backward compatibility with checkpoints saved before Fix
        # 15: fall back to a uniform (summing-to-1) weighting, matching what
        # the old unconditional-/n_est additive update was equivalent to.
        if "actor_var_tree_weights" in blob:
            self.engine.actor_var_tree_weights = blob["actor_var_tree_weights"]
        else:
            n_est = self.engine.n_estimators
            self.engine.actor_var_tree_weights = np.full(
                self.engine.actor_tree_weights.shape, 1.0 / n_est
            )
        self.engine.critic_leaf_means = blob["critic_leaf_means"]
        self.engine.critic_tree_weights = blob["critic_tree_weights"]
        self.engine._n_leaf_slots = blob["n_leaf_slots"]
        self.engine.shared_tree_structure = blob["shared_tree_structure"]
        # .get() for backward compatibility with pre-Fix-15 checkpoints, which
        # were all computed with the (then-only) inverse-magnitude scheme.
        self.engine.invert_tree_weights = blob.get("invert_tree_weights", True)
        if blob["log_std"] is not None:
            self.engine.log_std = blob["log_std"]
        self.num_timesteps = blob["num_timesteps"]
        return self


# --------------------------------------------------------------------------- #
# Minimal GAE buffer, used when stable-baselines3 is unavailable
# --------------------------------------------------------------------------- #

class _SimpleRolloutBuffer:
    def __init__(self, n_steps, obs_dim, action_dim, n_envs, gamma, gae_lambda):
        self.n_steps, self.n_envs = n_steps, n_envs
        self.gamma, self.gae_lambda = gamma, gae_lambda
        self.observations = np.zeros((n_steps, n_envs) + tuple(obs_dim), np.float32)
        self.actions = np.zeros((n_steps, n_envs, action_dim), np.float32)
        self.rewards = np.zeros((n_steps, n_envs), np.float32)
        self.episode_starts = np.zeros((n_steps, n_envs), np.float32)
        self.values = np.zeros((n_steps, n_envs), np.float32)
        self.log_probs = np.zeros((n_steps, n_envs), np.float32)
        self.advantages = np.zeros((n_steps, n_envs), np.float32)
        self.returns = np.zeros((n_steps, n_envs), np.float32)
        self.pos = 0

    def reset(self):
        self.pos = 0

    def add(self, obs, action, reward, episode_start, value, log_prob):
        i = self.pos
        self.observations[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.episode_starts[i] = episode_start
        self.values[i] = np.asarray(value).ravel()
        self.log_probs[i] = np.asarray(log_prob).ravel()
        self.pos += 1

    def compute_returns_and_advantage(self, last_values, dones):
        last_values = np.asarray(last_values, dtype=np.float32).ravel()
        last_gae = np.zeros(self.n_envs, dtype=np.float32)
        for step in reversed(range(self.n_steps)):
            if step == self.n_steps - 1:
                next_non_terminal = 1.0 - np.asarray(dones, dtype=np.float32)
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = (
                self.rewards[step]
                + self.gamma * next_values * next_non_terminal
                - self.values[step]
            )
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            self.advantages[step] = last_gae
        self.returns = self.advantages + self.values


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import gymnasium as gym

    class _VecWrap:
        """Tiny single-env VecEnv shim so the file is runnable standalone."""

        def __init__(self, env):
            self.env, self.num_envs = env, 1
            self.observation_space = env.observation_space
            self.action_space = env.action_space
            self._ret, self._len = 0.0, 0

        def reset(self):
            obs, _ = self.env.reset()
            self._ret, self._len = 0.0, 0
            return obs[None, :]

        def step(self, actions):
            a = np.clip(actions[0], self.action_space.low, self.action_space.high)
            obs, r, term, trunc, info = self.env.step(a)
            self._ret += r
            self._len += 1
            done = term or trunc
            infos = [dict(info)]
            if done:
                infos[0]["episode"] = {"r": self._ret, "l": self._len}
                infos[0]["terminal_observation"] = obs
                infos[0]["TimeLimit.truncated"] = bool(trunc and not term)
                obs, _ = self.env.reset()
                self._ret, self._len = 0.0, 0
            return obs[None, :], np.array([r], np.float32), np.array([done]), infos

    from EVCorridorEnv import EVCorridorEnv

    env = _VecWrap(EVCorridorEnv(use_meteostat=False, weather_cache=None))
    variants = [
        ("A2C  per-leaf std  shared", dict(use_ppo_clip=False, obs_dependent_std=True)),
        ("A2C  global  std  shared", dict(use_ppo_clip=False, obs_dependent_std=False)),
        ("PPO  per-leaf std  shared", dict(use_ppo_clip=True, obs_dependent_std=True)),
        ("PPO  global  std  shared", dict(use_ppo_clip=True, obs_dependent_std=False)),
        ("PPO  per-leaf std  per-dim",
         dict(use_ppo_clip=True, obs_dependent_std=True,
              shared_tree_structure=False)),
    ]
    for variant, kw in variants:
        model = Hybrid_XGB(
            env, n_steps=128, n_estimators=40, max_depth=4,
            awr_update_freq=512, awr_buffer_size=4000, awr_min_samples=200,
            ppo_lr=0.05, verbose=0, **kw,
        )
        model.learn(total_timesteps=2048)
        d = model.engine.last_diagnostics
        print(
            f"{variant}: std={d.get('policy/std_dim0', float('nan')):.3f}/"
            f"{d.get('policy/std_dim1', float('nan')):.3f}  "
            f"ceil_frac={d.get('policy/var_at_ceiling_dim0', 0):.2f}  "
            f"ev={d.get('train/explained_variance', 0):+.3f}  "
            f"clip={d.get('train/clip_fraction', 0):.3f}  "
            f"micro_dmu={d.get('train/micro_mu_delta', 0):.3f}  "
            f"macro_dmu={d.get('train/macro_mu_delta', float('nan')):.3f}"
        )