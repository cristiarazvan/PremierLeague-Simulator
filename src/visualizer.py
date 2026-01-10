import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

def ensure_plots_dir():
    if not os.path.exists('plots'):
        os.makedirs('plots')

def plot_league_heatmap(rankings_data, team_names, filename='league_heatmap.png'):
    ensure_plots_dir()
    data_matrix = []
    sorted_teams = sorted(team_names) 
    
    for team in sorted_teams:
        row = []
        total_sims = sum(rankings_data[team].values())
        if total_sims == 0:
            row = [0] * 20
        else:
            for pos in range(1, 21):
                count = rankings_data[team].get(pos, 0)
                percentage = (count / total_sims) * 100
                row.append(percentage)
        data_matrix.append(row)
        
    df = pd.DataFrame(data_matrix, index=sorted_teams, columns=range(1, 21))
    df = df.sort_values(by=1, ascending=False)

    plt.figure(figsize=(16, 10))
    sns.heatmap(df, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label': 'Probability (%)'})
    plt.title("League Position Probabilities (Monte Carlo N=10,000)", fontsize=16)
    plt.xlabel("League Position", fontsize=12)
    plt.ylabel("Team", fontsize=12)
    plt.tight_layout()
    
    save_path = os.path.join('plots', filename)
    plt.savefig(save_path)
    print(f"Heatmap saved to {save_path}")
    plt.show()

def plot_points_distribution(points_history, team_name, filename=None):
    ensure_plots_dir()
    if filename is None:
        filename = f'{team_name}_points_dist.png'
        
    plt.figure(figsize=(10, 6))
    sns.histplot(points_history, kde=True, bins=20, color='skyblue', edgecolor='black')
    
    mean_pts = np.mean(points_history)
    std_pts = np.std(points_history)
    
    plt.axvline(mean_pts, color='red', linestyle='--', label=f'Mean: {mean_pts:.1f}')
    plt.axvline(mean_pts + 1.96*std_pts, color='green', linestyle=':', label='95% CI')
    plt.axvline(mean_pts - 1.96*std_pts, color='green', linestyle=':')
    
    plt.title(f"Projected Points Distribution for {team_name}", fontsize=14)
    plt.xlabel("Points", fontsize=12)
    plt.ylabel("Frequency (Simulations)", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join('plots', filename)
    plt.savefig(save_path)
    print(f"Distribution plot saved to {save_path}")
    plt.show()

def plot_convergence(h_history, d_history, a_history, h_name, a_name, filename=None):
    ensure_plots_dir()
    if filename is None:
        filename = f'{h_name}_vs_{a_name}_convergence.png'

    N = len(h_history)
    x_axis = np.arange(1, N + 1)
    
    # Prepare data for all 3 outcomes
    outcomes = [
        (h_history, f"{h_name} Win", '#1f77b4'), # Blue
        (d_history, "Draw", '#7f7f7f'),           # Gray
        (a_history, f"{a_name} Win", '#d62728')   # Red
    ]

    fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    fig.suptitle(f"Monte Carlo Convergence: {h_name} vs {a_name} (N={N})", fontsize=16)

    for i, (history, label, color) in enumerate(outcomes):
        results = np.array(history)
        running_means = np.cumsum(results) / x_axis
        
        # Standard Error & 95% CI
        running_se = np.sqrt(running_means * (1 - running_means) / x_axis)
        running_se = np.nan_to_num(running_se)
        
        upper_bound = running_means + 1.96 * running_se
        lower_bound = running_means - 1.96 * running_se
        
        final_prob = running_means[-1]

        axs[i].plot(x_axis, running_means, color=color, label=f'Est. Probability')
        axs[i].fill_between(x_axis, lower_bound, upper_bound, color=color, alpha=0.2, label='95% CI')
        axs[i].axhline(y=final_prob, color='black', linestyle='--', linewidth=1, label=f'Final: {final_prob:.3f}')
        
        axs[i].set_ylabel("Probability", fontsize=10)
        axs[i].set_title(label, fontsize=12, fontweight='bold')
        axs[i].legend(loc='upper right')
        axs[i].grid(True, alpha=0.3)
        
        # Zoom Y-axis
        if 0 < final_prob < 1:
            axs[i].set_ylim(max(0, final_prob - 0.1), min(1, final_prob + 0.1))

    plt.xlabel("Number of Simulations (N)", fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Make room for suptitle
    
    save_path = os.path.join('plots', filename)
    save_path = "".join(save_path.split())
    plt.savefig(save_path)
    print(f"Convergence plot saved to {save_path}")
    plt.show()
