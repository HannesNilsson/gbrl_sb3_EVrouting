"""
EVCorridorEnv - Gymnasium environment for heavy-duty electric truck routing & charging.

Implements the charging-strategy MDP of the TERL paper (Sec. 3.1) with the
following corrections relative to the first draft implementation:

  1.  Driver hours follow EU Regulation (EC) 561/2006 and are *enforced*, not
      penalised: a 45 min break after 4.5 h of driving (Art. 7, which may be
      split into >= 15 min followed by >= 30 min), and a 9 h daily driving limit
      after which an 11 h daily rest is required (Art. 6.1, 8.2).  When a limit
      is reached the truck stops where it is; mid-leg that is the roadside, with
      no charger, so the time buys no energy.  A voluntary stop at a charging
      station discharges the same obligation *and* charges through it - and
      because the break may be split, two short top-ups can substitute for one
      longer stop.  The only remaining failure mode is running the battery flat.
  2.  Charger power is enforced: the applied C-rate is capped at
      P_charger / Q_t.  By default every station is guaranteed to deliver the
      full 2C action range (`min_charger_c_rate=2.0`), so the cap never binds
      and the observation stays the 10-dimensional vector of Eq. 12.  Set
      `min_charger_c_rate=None` to use the real CSV powers, in which case
      `include_charger_power=True` is strongly recommended so the agent can
      see the limit it is subject to.
  3.  Ambient temperature advances with the simulation clock instead of being
      frozen at reset; the clock is a float and no longer rounded per step.
  4.  Degradation is converted to kWh via the nominal capacity
      (Q <- Q - Q_nom * D) and alpha/beta/gamma are recalibrated so the
      fast-charge / time-saved trade-off is actually balanced (see NOTES).
  5.  The route keeps its real per-segment geometry: road segments are
      integrated between consecutive charger positions instead of being
      replaced by the route average.
  6.  Load weight and total vehicle weight are separate: Eq. 11 uses the total,
      Eq. 13 uses the load.
  7.  All sampling uses `self.np_random`, so `reset(seed=...)` is honoured.
  8.  Local electricity price varies per node and with time of day.
  9.  Revenue is not paid for a leg that failed.
 10.  The BMS charge curve (Eq. 7) is integrated in closed form instead of a
      660-iteration Python loop.
 11.  Charging efficiency separates battery-side from grid-side energy.
 12.  Decision order is charge-then-drive, so the agent commits to a stop
      *before* observing the stochastic consumption of the leg ahead, and no
      action is wasted at the destination node.
 13.  Battery state of health optionally persists across episodes, which is
      what makes the environment non-stationary in the sense of Sec. 2.2.

NOTES on calibration
--------------------
alpha / beta / gamma are expressed as *fractions of nominal capacity*, and the
capacity update is Q <- Q - Q_nom * (D_charge + D_drive).  The defaults
(alpha=2.5e-4, beta=1.5e-3, gamma=8e-5) were chosen so that, over a 1600 km
corridor with c_b = 150 EUR/kWh:

  * modelled battery depreciation is ~70-90 EUR per mission, consistent with a
    90 kEUR pack amortised over a ~1.5 M km life;
  * the C-rate has an interior optimum.  On an unconstrained charger the
    heuristic policy returns ~441 EUR at 0.8C but only ~107 EUR at 2.0C,
    because the C^2 term in Eq. 8 outruns the driver-wage time saved;
  * state of health falls below the 70% replacement threshold after ~375
    missions (~6 k environment steps), so battery ageing is visible inside a
    normal training run - this is the non-stationarity discussed in Sec. 2.2.

With the original alpha=0.05 applied to a dimensionless D and subtracted
directly from a capacity in kWh, the loss was ~0.14 kWh per charge and the
trade-off the paper is about had no effect on the reward at all.

"""

from __future__ import annotations

import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


# Candidate column names for locating chargers along the corridor.
_POSITION_COLUMNS = (
    "distance_km", "cum_distance_km", "cumulative_km", "position_km",
    "dist_km", "km", "s_km", "offset_km",
)
_POWER_COLUMNS = ("power_kw", "max_power_kw", "rated_power_kw", "power")
_PRICE_COLUMNS = ("price_eur_kwh", "price", "cost_eur_kwh", "energy_price")


class EVCorridorEnv(gym.Env):
    """Heavy-duty electric truck routing & charging along a 1D corridor."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        chargers_csv: str = "gallivare_to_gothenburg_chargers.csv",
        roads_csv: str = "gallivare_to_gothenburg_roads.csv",
        # --- vehicle / physics -------------------------------------------------
        speed_kmh: float = 80.0,
        initial_capacity_kwh: float = 600.0,
        tare_weight_t: float = 20.0,
        load_weight_range_t: tuple[float, float] = (10.0, 45.0),
        charge_efficiency: float = 0.92,
        # --- driver regulation -------------------------------------------------
        # --- EU Regulation (EC) 561/2006 ---------------------------------------
        max_drive_before_break: float = 4.5,
        break_duration: float = 0.75,
        split_break_first: float = 0.25,
        split_break_second: float = 0.50,
        max_daily_drive_hours: float = 9.0,
        daily_rest_hours: float = 11.0,
        rest_wage_factor: float = 0.3,
        include_eu_state: bool = True,
        # --- degradation -------------------------------------------------------
        alpha: float = 2.5e-4,
        beta: float = 1.5e-3,
        gamma: float = 8.0e-5,
        # --- economics ---------------------------------------------------------
        M: float = 0.05,
        c_e: float = 0.15,
        c_w: float = 30.0,
        c_b: float = 150.0,
        P_fail: float = 10000.0,
        price_volatility: float = 0.30,
        diurnal_price_amplitude: float = 0.25,
        # --- battery lifetime --------------------------------------------------
        persist_battery: bool = True,
        eol_capacity_fraction: float = 0.70,
        # --- exploration -------------------------------------------------------
        random_start: bool = False,
        # --- observation -------------------------------------------------------
        include_charger_power: bool = False,
        min_charger_c_rate: float | None = 2.0,
        # --- weather -----------------------------------------------------------
        weather_cache: str | None = "corridor_temps.npy",
        weather_years: tuple[int, int] = (2018, 2026),
        weather_point: tuple[float, float, float] = (63.8258, 20.2630, 12.0),
        use_meteostat: bool = True,
        seed: int | None = None,
        **kwargs,
    ):
        super().__init__()

        if kwargs:
            warnings.warn(f"EVCorridorEnv: ignoring unknown kwargs {sorted(kwargs)}")

        # ---- action space: [T_charging (h), C_target (C-rate)] ----------------
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([daily_rest_hours, 2.0], dtype=np.float32),
            dtype=np.float32,
        )

        # ---- constants --------------------------------------------------------
        self.v = float(speed_kmh)
        self.Q_nom = float(initial_capacity_kwh)
        self.tare_weight_t = float(tare_weight_t)
        self.load_weight_range_t = tuple(load_weight_range_t)
        self.eta_charge = float(charge_efficiency)

        self.max_drive_before_break = float(max_drive_before_break)
        self.break_duration = float(break_duration)
        self.split_break_first = float(split_break_first)
        self.split_break_second = float(split_break_second)
        self.max_daily_drive = float(max_daily_drive_hours)
        self.daily_rest_hours = float(daily_rest_hours)
        self.include_eu_state = bool(include_eu_state)
        self.rest_wage_factor = float(rest_wage_factor)

        self.alpha, self.beta, self.gamma = float(alpha), float(beta), float(gamma)

        self.M, self.c_e, self.c_w, self.c_b = float(M), float(c_e), float(c_w), float(c_b)
        self.P = float(P_fail)
        self.price_volatility = float(price_volatility)
        self.diurnal_price_amplitude = float(diurnal_price_amplitude)

        self.persist_battery = bool(persist_battery)
        self.eol_capacity_fraction = float(eol_capacity_fraction)
        self.include_charger_power = bool(include_charger_power)
        # Reverse-curriculum aid for long corridors.  On a 17-leg route the agent
        # must pay a full daily rest around leg 8, which only pays off if the
        # remaining nine legs also succeed - so while the downstream policy is
        # bad the rest has negative value, and the agent never learns it.
        # Starting episodes at a random node exposes the tail of the mission
        # directly, so the value of resting can be learned locally and propagate
        # backwards.  Always evaluate with random_start=False.
        self.random_start = bool(random_start)
        self.min_charger_c_rate = (
            None if min_charger_c_rate is None else float(min_charger_c_rate)
        )

        # Energy consumption model coefficients (Eq. 11)
        self.k1, self.m1 = 0.174, 5.13
        self.m2, self.m3, self.m4, self.m5 = -0.0131, 1.83e-5, 0.001, 0.710

        # ---- RNG (seeded through gym.Env.reset, but needed already here) ------
        self._construct_rng = np.random.default_rng(seed)
        super().reset(seed=seed)

        # ---- route ------------------------------------------------------------
        self.chargers_csv = chargers_csv
        self.roads_csv = roads_csv
        self._build_1d_track()

        # ---- weather ----------------------------------------------------------
        self.weather_cache = weather_cache
        self.weather_years = tuple(weather_years)
        self.weather_point = tuple(weather_point)
        self.use_meteostat = bool(use_meteostat)
        self._load_temperature_series()

        # ---- observation space ------------------------------------------------
        obs_dim = 10 + (2 if self.include_eu_state else 0) + (
            1 if self.include_charger_power else 0)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # ---- persistent state across episodes --------------------------------
        self.global_hour = 0.0          # float chronological clock
        self.Q_t = self.Q_nom
        self.battery_replacements = 0

        self.current_node = 0
        self.S_t = 1.0
        self._reset_day()
        self.h_t = 0.0
        self.load_t = 0.0
        self.w_t = 0.0

    # ==================================================================== #
    # Route construction
    # ==================================================================== #

    def _build_1d_track(self):
        """Build per-leg distances / altitude deltas / charger powers / prices.

        Road segments are integrated between consecutive charger positions, so
        terrain and station spacing keep their real heterogeneity.
        """
        have_csv = os.path.exists(self.chargers_csv) and os.path.exists(self.roads_csv)

        if not have_csv:
            warnings.warn(
                "EVCorridorEnv: CSVs not found, using a synthetic 7-node corridor."
            )
            self.jump_distances = np.array(
                [80.0, 110.0, 95.0, 120.0, 105.0, 90.0], dtype=np.float64
            )
            self.jump_altitudes = np.array(
                [15.0, -10.0, 25.0, -5.0, 30.0, -20.0], dtype=np.float64
            )
            node_powers = np.array(
                [350.0, 150.0, 250.0, 350.0, 250.0, 400.0, 350.0], dtype=np.float64
            )
            node_prices = None
        else:
            chargers_df = pd.read_csv(self.chargers_csv)
            roads_df = pd.read_csv(self.roads_csv)

            seg_len = roads_df["length_km"].to_numpy(dtype=np.float64)
            seg_alt = roads_df["delta_h_m"].to_numpy(dtype=np.float64)
            cum_km = np.concatenate([[0.0], np.cumsum(seg_len)])
            cum_alt = np.concatenate([[0.0], np.cumsum(seg_alt)])
            total_km = float(cum_km[-1])

            # -- charger positions along the corridor --------------------------
            pos_col = next(
                (c for c in _POSITION_COLUMNS if c in chargers_df.columns), None
            )
            if pos_col is not None:
                node_km = chargers_df[pos_col].to_numpy(dtype=np.float64)
            else:
                warnings.warn(
                    "EVCorridorEnv: no position column in chargers CSV "
                    f"(looked for {_POSITION_COLUMNS}); spacing stations evenly. "
                    "Per-leg distances will be uniform - add a distance column "
                    "to recover real station spacing."
                )
                node_km = np.linspace(0.0, total_km, len(chargers_df))

            order = np.argsort(node_km)
            node_km = np.clip(node_km[order], 0.0, total_km)
            chargers_df = chargers_df.iloc[order].reset_index(drop=True)

            # -- charger power / price ----------------------------------------
            pw_col = next((c for c in _POWER_COLUMNS if c in chargers_df.columns), None)
            if pw_col is None:
                warnings.warn("EVCorridorEnv: no power column found; assuming 350 kW.")
                node_powers = np.full(len(node_km), 350.0)
            else:
                node_powers = chargers_df[pw_col].to_numpy(dtype=np.float64)

            pr_col = next((c for c in _PRICE_COLUMNS if c in chargers_df.columns), None)
            node_prices = (
                chargers_df[pr_col].to_numpy(dtype=np.float64) if pr_col else None
            )

            # -- append the destination if the last charger is not there -------
            if node_km[-1] < total_km - 1e-6:
                node_km = np.append(node_km, total_km)
                node_powers = np.append(node_powers, 0.0)
                if node_prices is not None:
                    node_prices = np.append(node_prices, node_prices[-1])
            # -- drop duplicate positions --------------------------------------
            keep = np.concatenate([[True], np.diff(node_km) > 1e-6])
            node_km, node_powers = node_km[keep], node_powers[keep]
            if node_prices is not None:
                node_prices = node_prices[keep]

            self.jump_distances = np.diff(node_km)
            self.jump_altitudes = np.diff(np.interp(node_km, cum_km, cum_alt))

        self.num_nodes = len(self.jump_distances) + 1
        self.charger_powers = np.asarray(node_powers, dtype=np.float64)
        if len(self.charger_powers) != self.num_nodes:
            self.charger_powers = np.resize(self.charger_powers, self.num_nodes)

        # Base (time-invariant) local electricity price per node.
        if node_prices is not None and len(node_prices) == self.num_nodes:
            self.node_base_prices = np.asarray(node_prices, dtype=np.float64)
        else:
            jitter = self._construct_rng.uniform(
                -self.price_volatility, self.price_volatility, size=self.num_nodes
            )
            self.node_base_prices = self.c_e * (1.0 + jitter)

        # Guarantee that the full C_target action range is physically reachable
        # at every station: P >= C_max * Q_nom kW, since C = P / Q and Q <= Q_nom.
        # With the default the power cap never binds, so the agent is not subject
        # to a constraint it cannot observe when include_charger_power=False.
        if self.min_charger_c_rate is not None:
            self.charger_powers = np.maximum(
                self.charger_powers, self.min_charger_c_rate * self.Q_nom
            )

        # Suffix sums for the "remaining to destination" observations.
        self._suffix_dist = np.concatenate(
            [np.cumsum(self.jump_distances[::-1])[::-1], [0.0]]
        )
        self._suffix_alt = np.concatenate(
            [np.cumsum(self.jump_altitudes[::-1])[::-1], [0.0]]
        )

    # ==================================================================== #
    # Weather
    # ==================================================================== #

    def _load_temperature_series(self):
        """Load hourly ambient temperatures, with cache and synthetic fallback."""
        self.historical_temps = None

        if self.weather_cache and os.path.exists(self.weather_cache):
            try:
                self.historical_temps = np.load(self.weather_cache)
                if self.historical_temps.size == 0:
                    self.historical_temps = None
            except Exception as exc:  # pragma: no cover
                warnings.warn(f"EVCorridorEnv: could not read weather cache ({exc}).")

        if self.historical_temps is None and self.use_meteostat:
            self.historical_temps = self._fetch_meteostat()
            if self.historical_temps is not None and self.weather_cache:
                try:
                    np.save(self.weather_cache, self.historical_temps)
                except Exception as exc:  # pragma: no cover
                    warnings.warn(f"EVCorridorEnv: could not write cache ({exc}).")

        if self.historical_temps is None:
            warnings.warn(
                "EVCorridorEnv: using a synthetic seasonal temperature model."
            )

    def _fetch_meteostat(self):
        try:
            from meteostat import Point, Hourly

            lat, lon, alt = self.weather_point
            point = Point(lat, lon, alt)
            y0, y1 = self.weather_years

            temps = []
            for year in range(y0, y1 + 1):
                start = datetime(year, 1, 1)
                end = datetime(year, 12, 31, 23, 59)
                df = Hourly(point, start, end).fetch()
                if df is not None and not df.empty and "temp" in df.columns:
                    # Interpolate rather than drop, so the hourly index stays aligned
                    # with wall-clock time.
                    temps.append(
                        df["temp"].interpolate(limit_direction="both").to_numpy()
                    )
            if temps:
                series = np.concatenate(temps)
                series = series[np.isfinite(series)]
                if series.size:
                    return series.astype(np.float64)
            return None
        except Exception as exc:
            warnings.warn(f"EVCorridorEnv: Meteostat unavailable ({exc}).")
            return None

    def _ambient_temp(self, hour: float) -> float:
        """Ambient temperature at absolute hour `hour` since the epoch start."""
        if self.historical_temps is not None and len(self.historical_temps):
            idx = int(hour) % len(self.historical_temps)
            return float(self.historical_temps[idx])
        # Synthetic northern-Sweden seasonal + diurnal model.
        day = hour / 24.0
        seasonal = 3.0 + 12.0 * np.cos(2.0 * np.pi * (day - 200.0) / 365.25)
        diurnal = 3.0 * np.cos(2.0 * np.pi * (hour % 24.0 - 14.0) / 24.0)
        return float(seasonal + diurnal + self.np_random.normal(0.0, 1.5))

    def _local_price(self, node: int, hour: float) -> float:
        """Local electricity price: node-specific base with a diurnal profile."""
        diurnal = 1.0 + self.diurnal_price_amplitude * np.cos(
            2.0 * np.pi * (hour % 24.0 - 18.0) / 24.0
        )
        return float(max(0.01, self.node_base_prices[node] * diurnal))

    # ==================================================================== #
    # Gym API
    # ==================================================================== #

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.random_start and self.num_nodes > 2:
            self.current_node = int(self.np_random.integers(0, self.num_nodes - 1))
            self.S_t = float(self.np_random.uniform(0.35, 1.0))
            self._reset_day()
            self.drive_today = float(self.np_random.uniform(0.0, self.max_daily_drive))
            self.drive_since_break = float(
                self.np_random.uniform(0.0, self.max_drive_before_break))
        else:
            self.current_node = 0
            self.S_t = 1.0
            self._reset_day()

        if not self.persist_battery:
            self.Q_t = self.Q_nom
        elif self.Q_t < self.eol_capacity_fraction * self.Q_nom:
            # End of life: pack replaced, health resets (a non-stationary jump).
            self.Q_t = self.Q_nom
            self.battery_replacements += 1

        lo, hi = self.load_weight_range_t
        self.load_t = float(self.np_random.uniform(lo, hi))
        self.w_t = self.tare_weight_t + self.load_t

        self.h_t = self._ambient_temp(self.global_hour)
        self.p_t = self._local_price(self.current_node, self.global_hour)

        return self._get_obs(), self._info()

    def _get_obs(self) -> np.ndarray:
        node = self.current_node
        rem_dist = float(self._suffix_dist[node])
        rem_alt = float(self._suffix_alt[node])
        next_dist = (
            float(self.jump_distances[node]) if node < len(self.jump_distances) else 0.0
        )
        next_alt = (
            float(self.jump_altitudes[node]) if node < len(self.jump_altitudes) else 0.0
        )

        obs = [
            self.S_t, self.Q_t, self.drive_today, self.p_t,
            rem_dist, next_dist, rem_alt, next_alt,
            self.h_t, self.w_t,
        ]
        if self.include_eu_state:
            # Without these the MDP is partially observed: the agent cannot tell
            # whether the next leg will trigger a 45 min break or an 11 h daily
            # rest, and whether half a break is already banked.  Eq. 12 of the
            # draft needs extending to match; set
            # include_eu_state=False to recover the 10-dimensional vector.
            obs.extend([
                float(self.drive_since_break),
                float(self.break_part_taken),
            ])
        if self.include_charger_power:
            obs.append(float(self.charger_powers[node]))
        return np.asarray(obs, dtype=np.float32)

    def observation_bounds(self):
        """Analytic (low, high) per observation dimension, for rescaling."""
        total_km = float(self._suffix_dist[0])
        max_jump = float(self.jump_distances.max())
        alt_cum = np.concatenate([[0.0], np.cumsum(self.jump_altitudes)])
        lo_list = [
            0.0,                                        # S_t
            self.eol_capacity_fraction * self.Q_nom,    # Q_t
            0.0,                                        # drive_today
            0.5 * self.c_e,                             # p_t
            0.0,                                        # x_d
            0.0,                                        # x_c
            float(min(0.0, alt_cum.min())),             # a_d
            float(self.jump_altitudes.min()),           # a_c
            -30.0,                                      # h_t
            self.tare_weight_t + self.load_weight_range_t[0],   # w_t
        ]
        hi_list = [
            1.0, self.Q_nom, self.max_daily_drive, 2.0 * self.c_e,
            total_km, max_jump,
            float(max(0.0, alt_cum.max())),
            float(self.jump_altitudes.max()),
            30.0,
            self.tare_weight_t + self.load_weight_range_t[1],
        ]
        if self.include_eu_state:
            lo_list += [0.0, 0.0]
            hi_list += [self.max_drive_before_break, 1.0]
        if self.include_charger_power:
            lo_list.append(0.0)
            hi_list.append(float(self.charger_powers.max()))
        return (np.asarray(lo_list, dtype=np.float64),
                np.asarray(hi_list, dtype=np.float64))

    def _info(self, **extra) -> dict:
        info = {
            "node": int(self.current_node),
            "soc": float(self.S_t),
            "capacity_kwh": float(self.Q_t),
            "soh": float(self.Q_t / self.Q_nom),
            "drive_today_h": float(self.drive_today),
            "drive_since_break_h": float(self.drive_since_break),
            "clock_h": float(self.global_hour),
            "load_t": float(self.load_t),
            "battery_replacements": int(self.battery_replacements),
        }
        info.update(extra)
        return info

    # ------------------------------------------------------------------ #
    # Physics helpers
    # ------------------------------------------------------------------ #

    def _charge_forward(self, S0: float, t_hours: float, C: float) -> float:
        """Closed-form integration of the BMS charge curve (Eq. 7).

        dS/dt = 10*C*S           for S < 0.10   -> S(t) = S0 * exp(10*C*t)
        dS/dt = C                for 0.10<=S<=0.80 -> linear
        dS/dt = 5*C*(1-S)        for S > 0.80   -> S(t) = 1-(1-S0)*exp(-5*C*t)

        Note S = 0 is an absorbing point of this model; callers must pass S0 > 0.
        """
        if t_hours <= 0.0 or C <= 0.0:
            return S0

        S = max(float(S0), 1e-6)
        rem = float(t_hours)

        if S < 0.10:
            t_to = np.log(0.10 / S) / (10.0 * C)
            if rem <= t_to:
                return float(min(1.0, S * np.exp(10.0 * C * rem)))
            S, rem = 0.10, rem - t_to

        if S < 0.80:
            t_to = (0.80 - S) / C
            if rem <= t_to:
                return float(min(1.0, S + C * rem))
            S, rem = 0.80, rem - t_to

        return float(min(1.0, 1.0 - (1.0 - S) * np.exp(-5.0 * C * rem)))

    # ------------------------------------------------------------------ #
    # EU Regulation (EC) 561/2006
    # ------------------------------------------------------------------ #
    #
    # Modelled provisions:
    #   Art. 7   45 min break after 4.5 h of accumulated driving, which may be
    #            replaced by >= 15 min followed by >= 30 min
    #   Art. 6.1 9 h daily driving limit
    #   Art. 8.2 11 h daily rest once the daily driving limit is reached
    #
    # Weekly limits (Art. 6.2, 8.6) and the daily extensions/reductions are
    # deliberately not modelled: on a single-mission corridor they never bind.
    #
    # These are enforced, not penalised: the driver physically cannot continue,
    # so when a limit is reached the truck stops where it is.  Mid-leg that is
    # the roadside, where there is no charger and the time buys no energy.  A
    # voluntary stop at a charging station satisfies the same obligations *and*
    # charges, which is the whole reason it is worth planning one.
    #

    def _reset_day(self):
        self.drive_today = 0.0
        self.drive_since_break = 0.0
        self.break_part_taken = False

    def _take_daily_rest(self) -> float:
        self._reset_day()
        return self.daily_rest_hours

    def _required_break(self) -> float:
        """Break still owed: 30 min if the first split part is banked, else 45."""
        return (self.split_break_second if self.break_part_taken
                else self.break_duration)

    def _drive_with_rules(self, drive_time: float) -> dict:
        """Drive `drive_time` hours, inserting every stop the law requires."""
        stops = {"break": 0.0, "daily": 0.0}
        remaining = float(drive_time)
        guard = 0
        while remaining > 1e-9 and guard < 200:
            guard += 1
            until_break = self.max_drive_before_break - self.drive_since_break
            until_daily = self.max_daily_drive - self.drive_today
            allowed = min(until_break, until_daily)

            if allowed <= 1e-9:
                if until_daily <= 1e-9:
                    stops["daily"] += self._take_daily_rest()
                else:
                    stops["break"] += self._required_break()
                    self.drive_since_break = 0.0
                    self.break_part_taken = False
                continue

            step = min(remaining, allowed)
            self.drive_since_break += step
            self.drive_today += step
            remaining -= step
        return stops

    def _apply_voluntary_stop(self, hours: float) -> str:
        """Credit a chosen stop at a charger against the driver's obligations.

        A stop at a charging station is off-duty time, so it discharges whatever
        it is long enough to satisfy - and the truck charges through it.  The
        45 min break may instead be taken as >= 15 min followed by >= 30 min
        (Art. 7), so two short top-ups can substitute for one longer stop; since
        short top-ups are what the charging strategy wants anyway, the split is
        a real option rather than bookkeeping.
        """
        if hours >= self.daily_rest_hours - 1e-9:
            self._reset_day()
            return "daily_rest"
        if hours >= self.break_duration - 1e-9:
            self.drive_since_break = 0.0
            self.break_part_taken = False
            return "full_break"
        if self.break_part_taken and hours >= self.split_break_second - 1e-9:
            self.drive_since_break = 0.0
            self.break_part_taken = False
            return "split_break_completed"
        if not self.break_part_taken and hours >= self.split_break_first - 1e-9:
            self.break_part_taken = True
            return "split_break_started"
        return "none"

    def _driving_degradation(self, S_high: float, S_low: float) -> float:
        """Integral of Eq. 9 over the SoC window traversed while driving."""
        def antideriv(S):
            return (self.beta / 3.0) * ((S - 0.5) ** 3) + self.gamma * S

        return float(max(0.0, antideriv(S_high) - antideriv(S_low)))

    # ------------------------------------------------------------------ #
    # Transition
    # ------------------------------------------------------------------ #

    def step(self, action):
        node = self.current_node
        max_stop = float(self.action_space.high[0])
        t_charging = float(np.clip(action[0], 0.0, max_stop))
        c_target = float(np.clip(action[1], 0.0, float(self.action_space.high[1])))

        delta_x = float(self.jump_distances[node])
        delta_a = float(self.jump_altitudes[node])
        drive_time = delta_x / self.v

        fail_flag = 0
        fail_reason = None

        # ---- 1. charging at the current node -------------------------------
        # The physical charger caps the achievable C-rate: kW / kWh = 1/h = C.
        c_max_node = self.charger_powers[node] / max(self.Q_t, 1e-6)
        c_applied = float(min(c_target, c_max_node))

        S_high = self._charge_forward(self.S_t, t_charging, c_applied)
        delta_s_charge = max(0.0, S_high - self.S_t)
        e_charge_battery = delta_s_charge * self.Q_t
        e_charge_grid = e_charge_battery / self.eta_charge

        d_charge = self.alpha * (c_applied ** 2) * delta_s_charge

        # ---- 2. driver hours (EU 561/2006) -----------------------------------
        stop_credit = self._apply_voluntary_stop(t_charging)
        is_rest = stop_credit.endswith("rest")
        forced = self._drive_with_rules(drive_time)
        forced_rest = forced["break"] + forced["daily"]

        # ---- 3. driving ------------------------------------------------------
        e_drive_mu = delta_x * (
            self.m1 * np.exp(-self.k1 * self.v)
            + self.m2 * self.h_t
            + self.m3 * (self.w_t * 1000.0)
            + self.m4 * delta_a
            + self.m5
        )
        e_drive_mu = max(0.0, float(e_drive_mu))
        e_drive = max(0.0, float(self.np_random.normal(e_drive_mu, 0.05 * e_drive_mu)))

        S_low = S_high - e_drive / max(self.Q_t, 1e-6)
        if S_low <= 0.0:
            fail_flag = 1
            fail_reason = fail_reason or "battery_depleted"
            S_low = 0.0

        d_drive = self._driving_degradation(S_high, max(S_low, 0.0))

        # ---- 4. bookkeeping --------------------------------------------------
        total_degradation = d_charge + d_drive           # fraction of nominal
        capacity_loss_kwh = total_degradation * self.Q_nom
        self.Q_t = max(1e-3, self.Q_t - capacity_loss_kwh)

        delta_tau = drive_time + t_charging + forced_rest  # wall-clock elapsed
        self.global_hour += delta_tau

        # ---- 5. reward (Eq. 13) ---------------------------------------------
        revenue = 0.0 if fail_flag else self.M * self.load_t * delta_x
        electricity_cost = self.p_t * e_charge_grid
        # Driving is paid at full rate.  Off-duty time - any stop long enough to
        # count as a break or a rest, whether chosen or imposed - is paid at
        # `rest_wage_factor`, since EU drivers are not on an hourly rate through
        # a statutory rest.
        off_duty = forced_rest + (t_charging if stop_credit != "none" else 0.0)
        on_duty = drive_time + (t_charging if stop_credit == "none" else 0.0)
        wage_hours = on_duty + off_duty * self.rest_wage_factor
        driver_cost = self.c_w * wage_hours
        battery_cost = self.c_b * capacity_loss_kwh

        reward = (
            revenue
            - (electricity_cost + driver_cost + battery_cost)
            - self.P * fail_flag
        )

        # ---- 6. advance ------------------------------------------------------
        self.S_t = float(np.clip(S_low, 0.0, 1.0))
        self.current_node = node + 1
        self.h_t = self._ambient_temp(self.global_hour)
        self.p_t = self._local_price(
            min(self.current_node, self.num_nodes - 1), self.global_hour
        )

        terminated = bool(fail_flag) or self.current_node >= (self.num_nodes - 1)
        truncated = False

        info = self._info(
            revenue=float(revenue),
            electricity_cost=float(electricity_cost),
            driver_cost=float(driver_cost),
            battery_cost=float(battery_cost),
            capacity_loss_kwh=float(capacity_loss_kwh),
            e_charge_grid_kwh=float(e_charge_grid),
            e_drive_kwh=float(e_drive),
            c_requested=float(c_target),
            c_applied=float(c_applied),
            c_power_limited=bool(c_applied < c_target - 1e-9),
            took_daily_rest=bool(is_rest),
            stop_credit=stop_credit,
            forced_rest_h=float(forced_rest),
            forced_break_h=float(forced["break"]),
            forced_daily_rest_h=float(forced["daily"]),
            drive_today_h=float(self.drive_today),
            break_part_taken=bool(self.break_part_taken),
            fail=bool(fail_flag),
            fail_reason=fail_reason,
            mission_complete=bool(
                not fail_flag and self.current_node >= (self.num_nodes - 1)
            ),
        )

        return self._get_obs(), float(reward), terminated, truncated, info


# ---------------------------------------------------------------------------
# Wrappers for neural-network agents
# ---------------------------------------------------------------------------
#
# Tree ensembles are invariant to affine rescaling of the inputs, so TERL can be
# trained on the raw environment.  An MLP cannot: measured under a random policy
# the per-dimension observation standard deviations span a factor of ~1855
# (p_t ~ 0.04 vs x_d ~ 72), rewards have std ~5100 with a range of
# [-10428, +183], and an unsquashed Gaussian policy initialised at N(0, 1) clips
# 50% of its samples to the lower bound and executes a mean stop of 0.4 h on a
# [0, 11] range - putting the 11 h daily rest 11 standard deviations away, so it
# is never sampled.  Any of the three alone is enough to stall PPO.
#
# `make_nn_env` applies the three standard corrections.  Use it for every
# neural-network baseline; compare against TERL on the raw env only if you also
# report that TERL is scale-invariant and the baseline is not.


class ScaleObservation(gym.ObservationWrapper):
    """Affine rescaling of observations to roughly zero mean and unit range.

    Uses analytic bounds from the environment configuration rather than running
    statistics, so the transform is fixed, reproducible across seeds, and
    identical between training and evaluation.
    """

    def __init__(self, env):
        super().__init__(env)
        lo, hi = env.unwrapped.observation_bounds()
        self.center = 0.5 * (hi + lo)
        self.scale = np.maximum(0.5 * (hi - lo), 1e-6)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=env.observation_space.shape,
            dtype=np.float32,
        )

    def observation(self, obs):
        return ((np.asarray(obs, dtype=np.float64) - self.center)
                / self.scale).astype(np.float32)


class PowerScaleAction(gym.ActionWrapper):
    """Map a [-1, 1] policy action onto the environment's box with a power law.

    A linear rescaling of T_charging is badly conditioned for this problem.  A
    near-optimal policy spends 27% of decisions on "drive on" (0 h), 61% on
    top-ups under 1 h, 12% on a daily rest (>= 9 h) and essentially nothing in
    between - so under a linear map 88% of the useful actions are squeezed into
    9% of the axis while the unused 1-9 h band eats 73% of it.  A Gaussian
    policy cannot resolve that, which is what pins `std` near its initial value.

    With power p, action a in [-1, 1] maps to low + (high - low) * u**p where
    u = (a + 1) / 2.  p = 2 on the duration axis widens the 0-1 h region from 9%
    to 30% of the range while keeping the rest region ~10% wide.  This is a
    reparameterisation of the policy, not a change to the MDP - the environment
    still receives T_charging in [0, 11] - but apply it to every agent you
    compare, since it changes what each of them can express.
    """

    def __init__(self, env, power=(2.0, 1.0)):
        super().__init__(env)
        self.low = np.asarray(env.action_space.low, dtype=np.float64)
        self.high = np.asarray(env.action_space.high, dtype=np.float64)
        self.power = np.broadcast_to(
            np.asarray(power, dtype=np.float64), self.low.shape
        ).astype(np.float64).copy()
        self.action_space = spaces.Box(
            -1.0, 1.0, shape=self.low.shape, dtype=np.float32
        )

    def action(self, a):
        u = (np.clip(np.asarray(a, dtype=np.float64), -1.0, 1.0) + 1.0) / 2.0
        return (self.low + (self.high - self.low) * u ** self.power).astype(np.float32)


class ScaleReward(gym.RewardWrapper):
    """Divide rewards by a fixed constant.

    The failure penalty is ~40x a typical per-step profit, so unscaled value
    targets reach 1e4 and the squared value loss reaches 1e8.  A fixed divisor
    keeps the transform reproducible; unlike a running normaliser it does not
    change the relative weight of failures as training progresses.
    """

    def __init__(self, env, scale=1000.0):
        super().__init__(env)
        self.scale = float(scale)

    def reward(self, r):
        return float(r) / self.scale


def make_nn_env(reward_scale=1000.0, action_power=(2.0, 1.0), **env_kwargs):
    """EVCorridorEnv prepared for a learning agent.

    Applies, in order: observation rescaling, reward rescaling, and a power-law
    action map onto [-1, 1].  Use this for every agent you intend to compare,
    including the tree ensembles - although trees are invariant to the
    observation and reward transforms, the action reparameterisation changes
    what any Gaussian policy can express, so applying it to only one side would
    not be a fair comparison.

    Set action_power=(1.0, 1.0) to recover a plain linear rescaling.
    """
    env = EVCorridorEnv(**env_kwargs)
    env = ScaleObservation(env)
    env = ScaleReward(env, reward_scale)
    env = PowerScaleAction(env, power=action_power)
    return env


def verify_nn_env(env, n=2000, verbose=True):
    """Check that the scaling wrappers are actually in the loop.

    Every stalled run in this project so far traced back to the wrappers being
    bypassed - usually `make_vec_env(EVCorridorEnv, ...)`, which builds from the
    class and silently skips them.  Call this once before `learn()`.
    """
    problems = []
    lo, hi = env.action_space.low, env.action_space.high
    if not (np.allclose(lo, -1.0) and np.allclose(hi, 1.0)):
        problems.append(
            f"action_space is {env.action_space}, expected Box(-1, 1). "
            "The action wrapper is missing."
        )

    obs_mag, rew_mag = [], []
    o, _ = env.reset(seed=0)
    for _ in range(n):
        obs_mag.append(np.abs(o).max())
        o, r, term, trunc, _ = env.step(env.action_space.sample())
        rew_mag.append(abs(r))
        if term or trunc:
            o, _ = env.reset()
    obs_max, rew_max = float(np.max(obs_mag)), float(np.mean(rew_mag))

    if obs_max > 10.0:
        problems.append(
            f"max |observation| is {obs_max:.1f}, expected < ~2. "
            "ScaleObservation is missing."
        )
    if rew_max > 50.0:
        problems.append(
            f"mean |reward| is {rew_max:.1f}, expected < ~10. "
            "ScaleReward is missing - value targets will be ~1e5 and the "
            "critic will never fit (explained_variance stays near 0)."
        )

    if verbose:
        print(f"action_space      : {env.action_space}")
        print(f"max |observation| : {obs_max:.2f}")
        print(f"mean |reward|     : {rew_max:.3f}")
        print("OK" if not problems else "PROBLEMS:\n  - " + "\n  - ".join(problems))
    if problems:
        raise RuntimeError("env not correctly wrapped:\n  - " + "\n  - ".join(problems))
    return True


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    env = EVCorridorEnv(use_meteostat=False, weather_cache=None)
    print(f"nodes={env.num_nodes}  route={env._suffix_dist[0]:.0f} km  "
          f"obs_dim={env.observation_space.shape[0]}  "
          f"min charger power={env.charger_powers.min():.0f} kW")

    obs, _ = env.reset(seed=0)
    total, steps = 0.0, 0
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        steps += 1
        if terminated or truncated:
            break
    print(f"random policy: {steps} steps, return {total:.1f}, "
          f"fail={info['fail']} ({info['fail_reason']})")

    # A hand-crafted policy: charge to a comfortable level at a moderate C-rate.
    obs, _ = env.reset(seed=0)
    total = 0.0
    for _ in range(env.num_nodes - 1):
        soc, tau = obs[0], obs[2]
        stop = 11.0 if tau > 8.0 else (0.6 if soc < 0.55 else 0.0)
        obs, reward, terminated, truncated, info = env.step(np.array([stop, 0.8]))
        total += reward
        if terminated or truncated:
            break
    print(f"heuristic policy: return {total:.1f}, fail={info['fail']} "
          f"({info['fail_reason']}), SoH={info['soh']:.4f}")