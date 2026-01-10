import itertools
import json
import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List, Any, Tuple
import concurrent.futures
import multiprocessing

from src.data_loader import load_players_from_csv
from src.league import League
from src.models import Team, Player
from src.utils import calculate_player_metrics, simplify_position, compute_error

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================ 

PLAYER_CSV = 'data/raw/player_stats_2024-25.csv'
OUTPUT_DIR = 'output'

# Ground Truth: 2024-25 Premier League Standings
GROUND_TRUTH = [
    {'Team': 'Liverpool', 'Position': 1, 'Points': 84},
    {'Team': 'Arsenal', 'Position': 2, 'Points': 74},
    {'Team': 'Manchester City', 'Position': 3, 'Points': 71},
    {'Team': 'Chelsea', 'Position': 4, 'Points': 69},
    {'Team': 'Newcastle United', 'Position': 5, 'Points': 66},
    {'Team': 'Aston Villa', 'Position': 6, 'Points': 66},
    {'Team': 'Nottingham Forest', 'Position': 7, 'Points': 65},
    {'Team': 'Brighton & Hove Albion', 'Position': 8, 'Points': 61},
    {'Team': 'AFC Bournemouth', 'Position': 9, 'Points': 56},
    {'Team': 'Brentford', 'Position': 10, 'Points': 56},
    {'Team': 'Fulham', 'Position': 11, 'Points': 54},
    {'Team': 'Crystal Palace', 'Position': 12, 'Points': 53},
    {'Team': 'Everton', 'Position': 13, 'Points': 48},
    {'Team': 'West Ham United', 'Position': 14, 'Points': 43},
    {'Team': 'Manchester United', 'Position': 15, 'Points': 42},
    {'Team': 'Wolverhampton Wanderers', 'Position': 16, 'Points': 42},
    {'Team': 'Tottenham Hotspur', 'Position': 17, 'Points': 38},
    {'Team': 'Leicester City', 'Position': 18, 'Points': 25},
    {'Team': 'Ipswich Town', 'Position': 19, 'Points': 22},
    {'Team': 'Southampton', 'Position': 20, 'Points': 12},
]

# Parameter Ranges and Importance (determines pool size)
PARAM_CONFIG = {
    # Simulation Parameters
    'sigma': {'range': (0.05, 0.35), 'importance': 15, 'type': 'float'},
    'scaling_factor': {'range': (500, 2500), 'importance': 15, 'type': 'int'},
    'home_adv': {'range': (1.0, 1.3), 'importance': 10, 'type': 'float'},

    # Position Weights (Contributions)
    'att_def': {'range': (0.0, 0.5), 'importance': 5, 'type': 'float'},
    'mid_att': {'range': (0.4, 0.9), 'importance': 10, 'type': 'float'},
    'mid_def': {'range': (0.3, 0.8), 'importance': 10, 'type': 'float'},
    'def_att': {'range': (0.0, 0.5), 'importance': 5, 'type': 'float'},

    # Player Metric Weights
    'gls_weight': {'range': (2.0, 8.0), 'importance': 10, 'type': 'float'},
    'ast_weight': {'range': (1.0, 6.0), 'importance': 5, 'type': 'float'},
    'xg_weight': {'range': (5.0, 25.0), 'importance': 10, 'type': 'float'},
    'xag_weight': {'range': (5.0, 25.0), 'importance': 5, 'type': 'float'},
    'prg_weight': {'range': (0.05, 0.5), 'importance': 5, 'type': 'float'},
}

# ============================================================================
# HELPER CLASSES
# ============================================================================

class SimplePlayer:
    """Lightweight player object for faster processing in workers."""
    def __init__(self, name, squad, pos, mins, s_def, s_att, s_gk):
        self.name = name
        self.squad_name = squad
        self.position = pos
        self.minutes_played = mins
        self.s_def = s_def
        self.s_att = s_att
        self.s_gk = s_gk

class SilentOutput:
    """Context manager to suppress stdout in worker processes."""
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

# ============================================================================
# WORKER FUNCTION
# ============================================================================

def run_single_trial(args: Tuple) -> Tuple[Dict[str, Any], float]:
    config, raw_player_data, ground_truth, num_sims = args
    
    with SilentOutput():
        # 1. Weights
        metric_weights = {
            'gls': config['gls_weight'], 'ast': config['ast_weight'],
            'xg': config['xg_weight'], 'xag': config['xag_weight'],
            'prg': config['prg_weight'],
        }
        
        sim_params = {
            'sigma': config['sigma'], 'scaling_factor': config['scaling_factor'],
            'league_avg_goals': 1.6395, 'home_adv': config['home_adv'],
            'weights': {
                'ATT': {'att': 1.0, 'def': config['att_def']},
                'MID': {'att': config['mid_att'], 'def': config['mid_def']},
                'DEF': {'att': config['def_att'], 'def': 1.0},
                'GK':  {'att': 0.0, 'def': 0.0},
                'UNK': {'att': 0.5, 'def': 0.5}
            }
        }
        
        # 2. Build League with SimplePlayer
        teams = {}
        for row in raw_player_data:
            s_def, s_att, s_gk = calculate_player_metrics(row, metric_weights)
            squad_name = row.get('Squad', 'Unknown')
            if squad_name not in teams:
                teams[squad_name] = Team(squad_name)
            
            p = SimplePlayer(
                row.get('Player', 'Unknown'), squad_name, 
                simplify_position(row.get('Pos')), row.get('Min', 0),
                s_def, s_att, s_gk
            )
            teams[squad_name].add_player(p)
            
        league = League([])
        league.teams = teams
        league._calibrate_league() 

        # 3. Simulate
        team_powers = {name: t.calculate_power(sim_params) for name, t in league.teams.items()}
        total_points = {name: 0 for name in league.teams}
        team_names = list(league.teams.keys())
        
        for _ in range(num_sims):
            season_points = {name: 0 for name in league.teams}
            for i in range(len(team_names)):
                for j in range(len(team_names)):
                    if i == j: continue
                    h, a = team_names[i], team_names[j]
                    h_att, h_def = team_powers[h]
                    a_att, a_def = team_powers[a]
                    
                    gh, ga = league.simulate_match_fast(h_att, h_def, a_att, a_def, sim_params)
                    if gh > ga: season_points[h] += 3
                    elif ga > gh: season_points[a] += 3
                    else:
                        season_points[h] += 1
                        season_points[a] += 1
            for t in total_points: total_points[t] += season_points[t]
        
        results = sorted([{'Team': t, 'Points': pts/num_sims} for t, pts in total_points.items()], 
                        key=lambda x: x['Points'], reverse=True)
        return config, compute_error(results, ground_truth)


class LeagueOptimizer:
    def __init__(self, player_data_path: str, ground_truth: List[Dict]):
        self.player_data_path = player_data_path
        self.ground_truth = ground_truth
        self.raw_player_data = self._load_data()
        self.param_pools = self._generate_param_pools()

    def _load_data(self) -> List[Dict]:
        try:
            df = pd.read_csv(self.player_data_path)
            df.fillna(0, inplace=True)
            from src.data_loader import standardize_team_name
            if 'Squad' not in df.columns and 'Team' in df.columns:
                df['Squad'] = df['Team']
            if 'Squad' in df.columns:
                df['Squad'] = df['Squad'].apply(standardize_team_name)
            return df.to_dict('records')
        except Exception as e:
            print(f"Error loading data: {e}")
            return []

    def _generate_param_pools(self) -> Dict[str, List[Any]]:
        pools = {}
        np.random.seed(42)  
        for param, config in PARAM_CONFIG.items():
            count = config['importance']
            values = np.random.uniform(*config['range'], count)
            if config['type'] == 'int':
                values = np.unique(np.round(values).astype(int))
            else:
                values = np.round(values, 3)
            pools[param] = values.tolist()
        return pools

    def get_random_config(self) -> Dict[str, Any]:
        return {param: np.random.choice(values) for param, values in self.param_pools.items()}

    def search_parallel(self, n_trials=100, sims_per_trial=50):
        # Use all available logical cores
        max_workers = os.cpu_count() or 1
            
        print(f"Starting Parallel Search: {n_trials} trials, {sims_per_trial} sims/trial")
        print(f"Utilizing all {max_workers} CPU threads.")
        
        trial_args = [(self.get_random_config(), self.raw_player_data, self.ground_truth, sims_per_trial) 
                      for _ in range(n_trials)]
            
        best_error = float('inf')
        best_config = None
        history = []
        
        # smoothing=0 makes ETA based on total average, which is more stable for parallel starts
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_single_trial, arg) for arg in trial_args]
            for future in tqdm(concurrent.futures.as_completed(futures), total=n_trials, desc="Optimizing", smoothing=0):
                config, error = future.result()
                history.append({'config': config, 'error': error})
                if error < best_error:
                    best_error, best_config = error, config
                    tqdm.write(f"  New Best: MSE={error:.2f}")

        return best_config, best_error, history


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def main():
    # Make sure to protect entry point
    optimizer = LeagueOptimizer(PLAYER_CSV, GROUND_TRUTH)
    
    print("Parameter Pools Generated:")
    for param, vals in optimizer.param_pools.items():
        print(f"  {param}: {len(vals)} values")
    print("-" * 30)

    # Increased sims/trial because it's faster now
    best_config, best_error, history = optimizer.search_parallel(n_trials=5000, sims_per_trial=10000)

    print("\n" + "="*30)
    print(f"SEARCH COMPLETE. Best MSE: {best_error:.2f}")
    print("="*30)
    print("Best Configuration:")
    print(json.dumps(best_config, indent=2, cls=NumpyEncoder))
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{OUTPUT_DIR}/tuning_results_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            'best_config': best_config,
            'best_error': best_error,
            'history_top_10': sorted(history, key=lambda x: x['error'])[:10]
        }, f, indent=2, cls=NumpyEncoder)
        
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
