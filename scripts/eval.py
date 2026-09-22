"""Year-round evaluation for EVCorridorEnv.

EVCorridorEnv's clock (`global_hour`) is never reset between episodes, so
ambient temperature - and with it energy consumption - follows the calendar.
An evaluation env built fresh starts on 1 January, and a block of consecutive
evaluation episodes only covers the first few months: mostly winter, the
expensive end of the year.

This module evaluates on a fixed, calendar-stratified set of start times
instead: the same number of episodes starting in each month, spread evenly
within the month and across the years of the temperature record, each episode
reset with its own fixed seed.  Every agent is evaluated on exactly the same
(start hour, seed) pairs, so differences between agents are not diluted by
seasonal sampling noise (common random numbers).

    from ev_year_eval import evaluate_year_round, summarise
    per_episode = evaluate_year_round(model, env_kwargs, episodes_per_month=10)
    stats = summarise(per_episode)      # annual mean + per-month + seasons
"""

from __future__ import annotations

import numpy as np

from env.EVCorridorEnv import make_nn_env

HOURS_PER_YEAR = 8760.0
HOURS_PER_MONTH = HOURS_PER_YEAR / 12.0
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
SEASONS = {"winter": (11, 0, 1), "spring": (2, 3, 4),
           "summer": (5, 6, 7), "autumn": (8, 9, 10)}

# Fixed seed block for evaluation episodes: disjoint from training
# (seed .. seed + n_envs), the per-stage evaluation env (seed + 10_000), and
# identical for every agent and every training seed.
EVAL_SEED_BASE = 900_000


def _n_years(env) -> int:
    temps = getattr(env.unwrapped, "historical_temps", None)
    if temps is None or len(temps) < HOURS_PER_YEAR:
        return 1
    return max(1, int(len(temps) // HOURS_PER_YEAR))


def schedule(episodes_per_month: int, n_years: int):
    """(month, start_hour, seed) for every evaluation episode, deterministic."""
    out = []
    for m in range(12):
        for j in range(episodes_per_month):
            year = j % n_years
            within = (j + 0.5) / episodes_per_month * HOURS_PER_MONTH
            start = year * HOURS_PER_YEAR + m * HOURS_PER_MONTH + within
            out.append((m, float(start), EVAL_SEED_BASE + m * 1000 + j))
    return out


def _profit(info) -> float:
    return (float(info.get("revenue", 0.0))
            - float(info.get("electricity_cost", 0.0))
            - float(info.get("driver_cost", 0.0))
            - float(info.get("battery_cost", 0.0)))


def evaluate_year_round(model, env_kwargs, episodes_per_month: int = 10,
                        deterministic: bool = True):
    """Run the stratified evaluation; returns one dict per episode.

    Uses a plain (non-vectorised) env so the clock can be set *before* each
    reset: reset() reads temperature and electricity price at the current
    clock, and a VecEnv resets automatically inside step(), too late to move
    the clock first.  Observations are batched to shape (1, obs_dim), the same
    shape a single-env VecEnv hands the model, so SB3 and Hybrid_XGB both
    accept them unchanged.
    """
    # A fixed construction seed.  When the chargers CSV has no price column,
    # EVCorridorEnv draws every station's base electricity price at
    # construction from default_rng(seed) - and seed defaults to None, so each
    # new env gets a different price map.  Without this, the same model scores
    # differently on every evaluation and agents are compared on different
    # prices.  Pinning it gives every agent the same corridor.
    kwargs = dict(env_kwargs)
    kwargs.setdefault("seed", EVAL_SEED_BASE)
    env = make_nn_env(**kwargs)
    base = env.unwrapped
    rows = []
    for month, start, seed in schedule(episodes_per_month, _n_years(env)):
        base.global_hour = start
        obs, _ = env.reset(seed=seed)
        ret, profit, done, info = 0.0, 0.0, False, {}
        state, first = None, True
        while not done:
            out = model.predict(obs[None], state=state,
                                episode_start=np.array([first]),
                                deterministic=deterministic)
            action, state = out if isinstance(out, tuple) else (out, None)
            action = np.asarray(action).reshape(-1)
            obs, r, term, trunc, info = env.step(action)
            ret += float(r)
            profit += _profit(info)
            done, first = term or trunc, False
        rows.append({"month": month, "start_hour": start, "seed": seed,
                     "return": ret, "profit_eur": profit,
                     "completed": bool(info.get("mission_complete", False)),
                     "failed": bool(info.get("fail", False))})
    return rows


def summarise(rows):
    """Annual, per-season and per-month means from evaluate_year_round rows."""
    def agg(sub):
        if not sub:
            return None
        return {"return": float(np.mean([r["return"] for r in sub])),
                "profit_eur": float(np.mean([r["profit_eur"] for r in sub])),
                "completion_rate": float(np.mean([r["completed"] for r in sub])),
                "n": len(sub)}

    out = {"annual": agg(rows)}
    for name, months in SEASONS.items():
        out[name] = agg([r for r in rows if r["month"] in months])
    out["monthly"] = {MONTHS[m]: agg([r for r in rows if r["month"] == m])
                      for m in range(12)}
    return out