import matplotlib.pyplot as plt
import numpy as np
from src.data_loader import load_players_from_csv, load_teams_from_csv
from src.league import League

# Config
PLAYER_CSV = 'data/raw/player_stats_2024-25.csv'
TEAM_CSV = 'data/raw/team_stats_2024-25.csv'
SIM_PARAMS = {
    'sigma': 0.15, 'scaling_factor': 1500, 'league_avg_goals': 1.6, 'home_adv': 1.0,
    'weights': {'ATT': {'att': 1.0, 'def': 0.15}, 'MID': {'att': 0.7, 'def': 0.6},
                'DEF': {'att': 0.1, 'def': 1.0}, 'GK':  {'att': 0.0, 'def': 0.0},
                'UNK': {'att': 0.5, 'def': 0.5}}
}

def demonstrate_convergence():
    print("--- 1. Setting up Environment ---")
    all_players = load_players_from_csv(PLAYER_CSV)
    league = League(all_players)
    
    # Let's simulate a single match repeatedly: Liverpool vs Man City
    h_team = "Liverpool"
    a_team = "Manchester City"
    
    print(f"--- 2. Running Convergence Test ({h_team} vs {a_team}) ---")
    
    h_att, h_def = league.teams[h_team].calculate_power(SIM_PARAMS)
    a_att, a_def = league.teams[a_team].calculate_power(SIM_PARAMS)
    
    # We will track the cumulative win % for Liverpool
    # N = 10,000
    N = 10000
    results = [] # 1 if Liverpool wins, 0 otherwise
    
    print(f"Simulating {N} matches...")
    for i in range(N):
        gh, ga = league.simulate_match_fast(h_att, h_def, a_att, a_def, SIM_PARAMS)
        if gh > ga:
            results.append(1)
        else:
            results.append(0)
            
    # Calculate Running Average
    results = np.array(results)
    running_means = np.cumsum(results) / np.arange(1, N + 1)
    
    # Calculate Standard Error bounds (95% CI)
    # SE = sqrt(p(1-p)/n)
    # We use the running mean as proxy for p
    running_se = np.sqrt(running_means * (1 - running_means) / np.arange(1, N + 1))
    upper_bound = running_means + 1.96 * running_se
    lower_bound = running_means - 1.96 * running_se

    final_prob = running_means[-1]
    print(f"Final Estimated Probability: {final_prob:.4f}")

    print("--- 3. Generating Visualization ---")
    plt.figure(figsize=(14, 7))
    
    # Plot only the first 2000 for visibility of volatility, or all 10000 for smoothness
    # Let's plot all
    x_axis = np.arange(1, N + 1)
    
    plt.plot(x_axis, running_means, color='#1f77b4', label='Estimated Probability (Liverpool Win)')
    plt.fill_between(x_axis, lower_bound, upper_bound, color='#1f77b4', alpha=0.2, label='95% Confidence Interval')
    
    plt.axhline(y=final_prob, color='red', linestyle='--', linewidth=1, label=f'Final Converged Value ({final_prob:.3f})')
    
    plt.title(f"Monte Carlo Convergence: {h_team} Win Probability vs {a_team} Number of Simulations ({N})", fontsize=16)
    plt.xlabel(f"Number of Simulations ({N})", fontsize=12)
    plt.ylabel("Probability", fontsize=12)
    plt.ylim(final_prob - 0.05, final_prob + 0.05) # Zoom in around the mean
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_file = "convergence_plot.png"
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    plt.show()

if __name__ == "__main__":
    demonstrate_convergence()
