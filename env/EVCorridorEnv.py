import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class EVCorridorEnv(gym.Env):
    """
    Lifelong Reinforcement Learning Environment for Heavy-Duty EV Trucking.
    Alternates dynamically between Northbound and Southbound routes.
    """
    
    def __init__(self):
        super(EVCorridorEnv, self).__init__()
        
        print("🔌 Booting Simulator: Loading Bi-Directional Corridors...")
        
        # 1. LOAD BOTH ROUTES INTO RAM
        self.north_roads = pd.read_csv('gothenburg_to_gallivare_roads.csv')
        self.north_chargers = pd.read_csv('gothenburg_to_gallivare_chargers.csv')
        self.south_roads = pd.read_csv('gallivare_to_gothenburg_roads.csv')
        self.south_chargers = pd.read_csv('gallivare_to_gothenburg_chargers.csv')
        
        # ⚡ PRECOMPUTE BOTH TRACKS ONCE AT BOOT 
        print("⚡ Precomputing Northbound 1D Track...")
        self.north_powers, self.north_dists, self.north_times = self._build_1d_track(self.north_roads, self.north_chargers)
        
        print("⚡ Precomputing Southbound 1D Track...")
        self.south_powers, self.south_dists, self.south_times = self._build_1d_track(self.south_roads, self.south_chargers)
        
        # 2. TRUCK & ECONOMIC SPECS
        self.factory_max_kwh = 600.0
        self.current_max_kwh = 600.0  
        self.lifetime_trips = 0
        self.truck_max_kw = 400.0     
        self.consumption_mean = 1.2   
        self.consumption_std = 0.15   
        
        # Fleet Economics
        self.pack_cost_per_kwh = 150.0 
        self.total_pack_cost = self.pack_cost_per_kwh * self.factory_max_kwh 
        self.usable_life_fraction = 0.20 
        self.base_cycle_life = 10000.0
        total_lifetime_kwh = self.factory_max_kwh * self.base_cycle_life
        self.base_cost_per_kwh = self.total_pack_cost / (total_lifetime_kwh * self.usable_life_fraction)
        
        self.driver_wage_per_hr = 45.0
        self.energy_cost_per_kwh = 0.40
        self.freight_rate_per_km = 3.50

        # 3. SET INITIAL ROUTE (Start Northbound)
        self.is_northbound = True
        self._set_active_route()
        
        # --- RL SPACES ---
        # AI sees: [-1.0, 1.0] for both actions
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(2,),
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=-1.0, high=np.inf, shape=(16,), dtype=np.float32
        )

    def _build_1d_track(self, roads_df, chargers_df):
        """Helper function to squash a CSV into pure math arrays."""
        num_stations = len(chargers_df)
        powers = chargers_df['power_kw'].values.astype(np.float32)
        dists = np.zeros(num_stations - 1, dtype=np.float32)
        times = np.zeros(num_stations - 1, dtype=np.float32)
        
        charger_idx = 0
        target_charger_id = chargers_df.iloc[charger_idx + 1]['id']
        current_dist, current_time = 0.0, 0.0
        
        for _, road in roads_df.iterrows():
            d = road['length_km']
            t = road.get('travel_time_hr', d / 80.0)
            if t > (d / 5.0): t = d / 80.0
                
            current_dist += d
            current_time += t
            
            if road['driven_target'] == target_charger_id:
                dists[charger_idx] = current_dist
                times[charger_idx] = current_time
                current_dist, current_time = 0.0, 0.0
                
                charger_idx += 1
                if charger_idx >= num_stations - 1:
                    break
                    
                # 🐛 THE FIX: Update the target ID for the next loop!
                target_charger_id = chargers_df.iloc[charger_idx + 1]['id']
                    
        return powers, dists, times

    def _set_active_route(self):
        """Instantly swaps the active memory arrays. No math, no printing!"""
        if self.is_northbound:
            # 🐛 THE FIX: Re-added the dataframe pointers so reset() doesn't crash!
            self.chargers_df = self.north_chargers
            self.start_node = self.north_chargers.iloc[0]['id']
            self.target_node = self.north_chargers.iloc[-1]['id']
            
            self.charger_powers = self.north_powers
            self.jump_distances = self.north_dists
            self.jump_times = self.north_times
            self.num_stations = len(self.north_powers)
        else:
            self.chargers_df = self.south_chargers
            self.start_node = self.south_chargers.iloc[0]['id']
            self.target_node = self.south_chargers.iloc[-1]['id']
            
            self.charger_powers = self.south_powers
            self.jump_distances = self.south_dists
            self.jump_times = self.south_times
            self.num_stations = len(self.south_powers)

    def reset(self, seed=None, options=None):
        """Prepares for the next trip. The battery damage persists!"""
        super().reset(seed=seed)

        # 1. LIFELONG HEALTH CHECK
        state_of_health = self.current_max_kwh / self.factory_max_kwh
        battery_dead = state_of_health < 0.70 # Or 0.80 based on your fraction
        
        if battery_dead:
            print(f"🔧 BATTERY SWAP (Trip {self.lifetime_trips}): Selling degraded pack to grid storage for €72,000. Installing fresh pack...")
            self.current_max_kwh = self.factory_max_kwh
            self.lifetime_trips = 0
            self.is_northbound = True
        else:
            if self.lifetime_trips > 0:
                self.is_northbound = not self.is_northbound
                
        self._set_active_route()
            
        # 2. SOFT RESET
        self.current_node_idx = 0
        self.current_node_id = self.start_node
        self.battery = self.current_max_kwh 
        
        self.time_since_break = 0.0
        self.total_time_today = 0.0
        
        self.lifetime_trips += 1
        
        return self._get_obs(), {}

    def step(self, action):
        # 1. UN-SQUASH THE ACTIONS
        # Convert [-1.0, 1.0] to [0.0, 1.0]
        normalized_time = (action[0] + 1.0) / 2.0
        normalized_power = (action[1] + 1.0) / 2.0
        
        # Scale up to real-world physics limits!
        charge_time_hr = normalized_time * 11.0
        requested_power_kw = normalized_power * 1000.0
        
        degradation_cost_eur = 0.0
        energy_delivered_kwh = 0.0
        step_cost_eur = 0.0
        
        # --- 1. MINUTE-BY-MINUTE CHARGING SIMULATION ---
        if charge_time_hr > 0:
            station_max_kw = self.chargers_df.iloc[self.current_node_idx]['power_kw']
            hardware_limit_kw = min(requested_power_kw, station_max_kw, self.truck_max_kw)
            total_minutes = int(charge_time_hr * 60)
            chunk_size = 5 # Calculate physics in 5-minute blocks!
            
            for minute in range(0, total_minutes, chunk_size):
                # Ensure we don't over-calculate the final partial chunk
                actual_chunk = min(chunk_size, total_minutes - minute) 
                
                current_soc = self.battery / self.current_max_kwh
                if current_soc >= 1.0: break
                
                threshold = 0.80
                if current_soc <= threshold:
                    actual_power_kw = hardware_limit_kw
                else:
                    actual_power_kw = hardware_limit_kw * ((1.0 - current_soc) / (1.0 - threshold))
                
                energy_added = actual_power_kw / 60.0
                self.battery += energy_added
                energy_delivered_kwh += energy_added
                
                c_rate = actual_power_kw / self.factory_max_kwh
                power_stress = 1.0 + (0.5 * (c_rate ** 2))
                voltage_stress = np.exp(8.0 * (current_soc - 0.80)) if current_soc > 0.80 else 1.0
                total_stress = power_stress * voltage_stress
                
                minute_cost = (energy_added * self.base_cost_per_kwh) * total_stress
                degradation_cost_eur += minute_cost

                capacity_lost_kwh = (minute_cost / self.total_pack_cost) * (self.usable_life_fraction * self.factory_max_kwh)
                self.current_max_kwh -= capacity_lost_kwh
                if self.battery > self.current_max_kwh:
                    self.battery = self.current_max_kwh
            
            actual_charge_time_hr = total_minutes / 60.0
            if actual_charge_time_hr >= 11.0:
                self.total_time_today = 0.0  
                self.time_since_break = 0.0
            elif actual_charge_time_hr >= 0.75:
                self.time_since_break = 0.0  
                
            step_cost_eur += actual_charge_time_hr * self.driver_wage_per_hr
            step_cost_eur += energy_delivered_kwh * self.energy_cost_per_kwh

        # --- 2. DRIVING TO THE NEXT STATION ---
        if self.current_node_idx >= self.num_stations - 1:
            return self._get_obs(), 5000.0, True, False, {"reason": "reached_destination"}
        
        next_node_idx = self.current_node_idx + 1
        next_node_id = self.chargers_df.iloc[next_node_idx]['id']
        
        # ⚡ INSTANT O(1) LOOKUP! (No NetworkX needed)
        distance_km = self.jump_distances[self.current_node_idx]
        travel_time_hr = self.jump_times[self.current_node_idx]
        stop_overhead_hr = 0.25 if charge_time_hr > 0 else 0.0
        
        # Fail-safe check in case of bad map data
        if distance_km > 9000:
            return self._get_obs(), -5000, True, False, {"reason": "map_disconnected"}
            
        # ... [Inside Phase 2] ...
        expected_energy = distance_km * self.consumption_mean
        actual_energy = np.random.normal(expected_energy, distance_km * self.consumption_std)
        
        self.battery -= actual_energy

        step_time_hr = travel_time_hr + charge_time_hr + stop_overhead_hr
        self.total_time_today += step_time_hr
        self.time_since_break += step_time_hr

        if charge_time_hr >= 0.75:
            self.time_since_break = 0.0

        # ==========================================
        # 📉 REAL-WORLD PHYSICS: CONTINUOUS DEGRADATION
        # ==========================================
        starting_soc = (self.battery + actual_energy) / self.current_max_kwh
        ending_soc = self.battery / self.current_max_kwh
        avg_soc = (starting_soc + ending_soc) / 2.0
        
        stress_multiplier = 1.0 
        
        # 2. Low SOC Mechanical Stress
        if avg_soc < 0.30:
            stress_multiplier += np.exp(5.0 * (0.30 - avg_soc)) - 1.0
            
        # 3. High SOC Voltage Stress
        if avg_soc > 0.80:
            stress_multiplier += np.exp(3.0 * (avg_soc - 0.80)) - 1.0
            
        continuous_damage_eur = (actual_energy * self.base_cost_per_kwh) * stress_multiplier
        
        capacity_lost = (continuous_damage_eur / self.total_pack_cost) * (self.usable_life_fraction * self.factory_max_kwh)
        self.current_max_kwh -= capacity_lost
        
        if self.battery > self.current_max_kwh:
            self.battery = self.current_max_kwh
        # ==========================================
        
        step_revenue_eur = distance_km * self.freight_rate_per_km
        step_cost_eur += travel_time_hr * self.driver_wage_per_hr
        
        # Subtract the continuous physical damage from the profit
        step_profit = step_revenue_eur - (step_cost_eur + degradation_cost_eur + continuous_damage_eur)
        
        self.current_node_idx = next_node_idx
        self.current_node_id = next_node_id
        
        # --- 3. FATAL PENALTIES & REWARDS ---
        
        # A. Out of Battery (Death)
        if self.battery <= 0:
            return self._get_obs(), -100.0, True, False, {"reason": "out_of_battery"}

        # B. Destination Reached (Dynamic Bonus)
        if self.current_node_idx >= self.num_stations - 1:
            # Logic: Start with a high bonus (e.g., 100 points / €10,000)
            # Subtract 2 points for every hour the trip took.
            # Example: 15hr trip = 100 - 30 = 70 points.
            #          40hr trip = 100 - 80 = 20 points.
            arrival_bonus = max(0.0, 100.0 - (2.0 * self.total_time_today))
            
            return self._get_obs(), arrival_bonus, True, False, {"reason": "reached_destination"}

        # C. Standard Step Reward (Profit / 100)
        scaled_reward = step_profit / 100.0
        return self._get_obs(), scaled_reward, False, False, {}


        # # ==========================================
        # # 🐛 DEBUGGING: THE REPAIRED DUMMY REWARD
        # # ==========================================
        # # 1. THE CLIFF: Did the AI try to skip the charger?
        # if charge_time_hr < 0.1:
        #     dummy_reward = -100.0  # Massive punishment for skipping!
        
        # # 2. THE SLOPE: It decided to charge! Now guide it to [0.5, 200]
        # else:
        #     # We don't normalize to 11.0 anymore, we just use raw distance
        #     error_time = abs(charge_time_hr - 0.5) 
        #     error_power = abs(requested_power_kw - 200.0) / 100.0 # Scaled down so it isn't overwhelming
            
        #     # Start with 100 points, subtract points for being inaccurate
        #     dummy_reward = 100.0 - (15.0 * error_time) - (10.0 * error_power)
            
        # reward = dummy_reward
        # # ==========================================

        # return self._get_obs(), reward, False, False, {}



    def _get_expected_trip(self, start_idx, target_idx):
        """Lightning-fast array summing for the radar lookahead."""
        if start_idx >= target_idx or target_idx >= self.num_stations:
            return -1.0, -1.0
            
        # Just add up the precalculated chunks from the 1D track!
        dist = np.sum(self.jump_distances[start_idx:target_idx])
        time = np.sum(self.jump_times[start_idx:target_idx])
        
        return dist * self.consumption_mean, time

    def _get_obs(self):
        energy_to_dest, _ = self._get_expected_trip(self.current_node_idx, self.num_stations - 1)
        current_station_kw = self.charger_powers[self.current_node_idx]
        
        obs = [
            self.battery,
            self.current_max_kwh, 
            self.time_since_break,
            self.total_time_today,
            energy_to_dest,
            current_station_kw
        ]
        
        for offset in [1, 2, 3]:
            lookahead_idx = self.current_node_idx + offset
            if lookahead_idx < self.num_stations:
                energy_req, time_req = self._get_expected_trip(self.current_node_idx, lookahead_idx)
                power_avail = self.charger_powers[lookahead_idx]
                obs.extend([energy_req, time_req, power_avail])
            else:
                obs.extend([-1.0, -1.0, -1.0])
                
        obs.append(1.0 if self.current_node_idx + 3 < self.num_stations else 0.0)
        
        return np.array(obs, dtype=np.float32)