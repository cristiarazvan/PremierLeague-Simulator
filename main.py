from src.data_loader import load_players_from_csv
from src.league import League

# Config
PLAYER_CSV = 'data/raw/player_stats_2024-25.csv'

def main():
    print("--- 1. Loading Data ---")
    all_players = load_players_from_csv(PLAYER_CSV)
    
    if not all_players:
        print("No players loaded. Exiting.")
        return

    print("--- 2. Building League Framework ---")
    my_league = League(all_players)

    # 3. MONTE CARLO SIMULATION
    print("\n--- Monte Carlo Simulation (Liverpool vs Man City) ---")
    
    # Parameters
    sim_params = {
        'sigma': 0.15,             # 15% variability (luck/form)
        'scaling_factor': 1500,     # Normalize power difference
        'league_avg_goals': 1.6,
        'home_adv': 1.15
    }
    
    h_team, a_team = "Liverpool", "Manchester City"
    num_sims = 2000
    
    wins_h = 0
    wins_a = 0
    draws = 0
    
    print(f"Running {num_sims} simulations...")
    
    for _ in range(num_sims):
        gh, ga = my_league.simulate_match(h_team, a_team, sim_params)
        
        if gh > ga:
            wins_h += 1
        elif ga > gh:
            wins_a += 1
        else:
            draws += 1
            
    # Results
    print(f"\nResults for {h_team} (Home) vs {a_team} (Away):")
    print(f"Liverpool Win: {wins_h / num_sims * 100:.2f}%")
    print(f"Draw:          {draws / num_sims * 100:.2f}%")
    print(f"Man City Win:  {wins_a / num_sims * 100:.2f}%")
    
if __name__ == "__main__":
    main()
