import sys
import numpy as np
import pandas as pd
from src.data_loader import load_players_from_csv, load_teams_from_csv
from src.league import League
from src.models import Team
from src.visualizer import plot_league_heatmap, plot_points_distribution, plot_convergence

# Config
PLAYER_CSV = 'data/raw/player_stats_2024-25.csv'
TEAM_CSV = 'data/raw/team_stats_2024-25.csv'

SIM_PARAMS = {
    'sigma': 0.15,
    'scaling_factor': 1500,
    'league_avg_goals': 1.6,
    'home_adv': 1.0,
    'weights': {
        'ATT': {'att': 1.0, 'def': 0.15},
        'MID': {'att': 0.7, 'def': 0.6},
        'DEF': {'att': 0.1, 'def': 1.0},
        'GK':  {'att': 0.0, 'def': 0.0},
        'UNK': {'att': 0.5, 'def': 0.5}
    }
}

class PremierLeagueCLI:
    def __init__(self):
        print("Loading data...")
        self.all_players = load_players_from_csv(PLAYER_CSV)
        self.team_names = load_teams_from_csv(TEAM_CSV)
        self.league = League(self.all_players)
        self.custom_lineups = {} # Format: {'TeamName': [PlayerObj1, PlayerObj2...]}
        print("System Ready.\n")

    def run(self):
        while True:
            print("\n=== PREMIER LEAGUE MANAGER ===")
            print("1. Simulate Single Match")
            print("2. Simulate Full League Season")
            print("3. Manage Team / Edit Lineup")
            print("4. Create Custom Team")
            print("5. Exit")
            
            choice = input("Select option: ")
            
            if choice == '1':
                self.menu_match_sim()
            elif choice == '2':
                self.menu_league_sim()
            elif choice == '3':
                self.menu_manage_team()
            elif choice == '4':
                self.menu_create_team()
            elif choice == '5':
                print("Exiting...")
                sys.exit()
            else:
                print("Invalid option.")

    def _get_lineup_for_team(self, team_name):
        if team_name in self.custom_lineups:
            return self.custom_lineups[team_name]
        return None # Triggers default 11 in calculate_power

    def menu_match_sim(self):
        print("\n--- Match Simulation ---")
        h_team = input("Enter Home Team Name: ")
        a_team = input("Enter Away Team Name: ")

        if h_team not in self.league.teams or a_team not in self.league.teams:
            print("Error: One or both teams not found.")
            return

        print(f"Simulating {h_team} vs {a_team}...")
        
        # Pre-calc power using potential custom lineups
        h_lineup = self._get_lineup_for_team(h_team)
        a_lineup = self._get_lineup_for_team(a_team)

        # Note: calculate_power expects list of NAMES if specific lineup provided, or None
        # We need to extract names if we have objects
        h_names = [p.name for p in h_lineup] if h_lineup else None
        a_names = [p.name for p in a_lineup] if a_lineup else None

        h_att, h_def = self.league.teams[h_team].calculate_power(SIM_PARAMS, h_names)
        a_att, a_def = self.league.teams[a_team].calculate_power(SIM_PARAMS, a_names)

        wins_h, wins_a, draws = 0, 0, 0
        num_sims = 10000
        
        # Track history for convergence plotting (3 separate lists)
        h_hist = []
        d_hist = []
        a_hist = []

        for _ in range(num_sims):
            gh, ga = self.league.simulate_match_fast(h_att, h_def, a_att, a_def, SIM_PARAMS)
            if gh > ga: 
                wins_h += 1
                h_hist.append(1); d_hist.append(0); a_hist.append(0)
            elif ga > gh: 
                wins_a += 1
                h_hist.append(0); d_hist.append(0); a_hist.append(1)
            else: 
                draws += 1
                h_hist.append(0); d_hist.append(1); a_hist.append(0)

        print(f"\nResults ({num_sims} runs):")
        print(f"{h_team}: {wins_h/num_sims*100:.1f}%")
        print(f"Draw:      {draws/num_sims*100:.1f}%")
        print(f"{a_team}: {wins_a/num_sims*100:.1f}%")
        
        # Option to visualize convergence
        print("\n[Options]")
        print("V. Visualize Convergence (Generate Graph)")
        print("B. Back to Menu")
        choice = input("Select: ").upper()
        
        if choice == 'V':
            print("Generating Convergence Plot...")
            plot_convergence(h_hist, d_hist, a_hist, h_team, a_team)
            # No need to break or exit, plot_convergence shows plot then returns

    def menu_league_sim(self):
        print("\n--- League Simulation ---")
        print("Calculating team powers...")
        
        team_powers = {}
        for name, team in self.league.teams.items():
            lineup_objs = self._get_lineup_for_team(name)
            lineup_names = [p.name for p in lineup_objs] if lineup_objs else None
            team_powers[name] = team.calculate_power(SIM_PARAMS, lineup_names)

        # Storage for results
        rankings_data = {name: {} for name in self.league.teams} # Team -> {Pos: Count}
        points_history = {name: [] for name in self.league.teams} # Team -> [Points list]
        
        table_totals = {name: {'Points': 0, 'GF': 0} for name in self.league.teams}
        
        num_sims = 10000
        print(f"Running {num_sims} simulations of the entire season...")

        for _ in range(num_sims):
            # Run one full season round-robin
            season_points = {name: 0 for name in self.league.teams}
            season_gf = {name: 0 for name in self.league.teams}
            
            teams_list = list(self.league.teams.keys())
            for i in range(len(teams_list)):
                for j in range(i + 1, len(teams_list)):
                    t1, t2 = teams_list[i], teams_list[j]
                    
                    # Home Game for T1
                    t1_att, t1_def = team_powers[t1]
                    t2_att, t2_def = team_powers[t2]
                    
                    gh, ga = self.league.simulate_match_fast(t1_att, t1_def, t2_att, t2_def, SIM_PARAMS)
                    
                    season_gf[t1] += gh
                    season_gf[t2] += ga
                    if gh > ga: season_points[t1] += 3
                    elif ga > gh: season_points[t2] += 3
                    else:
                        season_points[t1] += 1
                        season_points[t2] += 1

                    # Away Game for T1 (Home for T2)
                    gh, ga = self.league.simulate_match_fast(t2_att, t2_def, t1_att, t1_def, SIM_PARAMS)
                    
                    season_gf[t2] += gh
                    season_gf[t1] += ga
                    if gh > ga: season_points[t2] += 3
                    elif ga > gh: season_points[t1] += 3
                    else:
                        season_points[t2] += 1
                        season_points[t1] += 1
            
            # Record season results
            sorted_table = sorted(season_points.items(), key=lambda x: (x[1], season_gf[x[0]]), reverse=True)
            
            for rank, (team_name, pts) in enumerate(sorted_table):
                pos = rank + 1
                # Update Rankings Hist
                rankings_data[team_name][pos] = rankings_data[team_name].get(pos, 0) + 1
                # Update Points Hist
                points_history[team_name].append(pts)
                
                # Totals for Average Table
                table_totals[team_name]['Points'] += pts
                table_totals[team_name]['GF'] += season_gf[team_name]

        # Average out
        results = []
        for t in table_totals:
            avg_pts = table_totals[t]['Points'] / num_sims
            avg_gf = table_totals[t]['GF'] / num_sims
            results.append({'Team': t, 'Avg Pts': avg_pts, 'Avg GF': avg_gf})

        # Sort by Points then GF
        results.sort(key=lambda x: (x['Avg Pts'], x['Avg GF']), reverse=True)

        print(f"\n{'Pos':<4} {'Team':<25} {'Pts':<6} {'GF':<6}")
        print("-" * 45)
        for i, res in enumerate(results):
            print(f"{i+1:<4} {res['Team']:<25} {res['Avg Pts']:.1f}   {res['Avg GF']:.1f}")
            
        # Visualisation Menu
        while True:
            print("\n[Visualisation Options]")
            print("1. Show Position Heatmap (All Teams)")
            print("2. Show Points Distribution (Specific Team)")
            print("3. Return to Main Menu")
            v_choice = input("Select: ")
            
            if v_choice == '1':
                print("Generating Heatmap...")
                plot_league_heatmap(rankings_data, self.league.teams.keys())
            elif v_choice == '2':
                t_input = input("Enter Team Name: ")
                if t_input in self.league.teams:
                    plot_points_distribution(points_history[t_input], t_input)
                else:
                    print("Team not found.")
            elif v_choice == '3':
                break

    def menu_manage_team(self):
        print("\n--- Team Manager ---")
        t_name = input("Enter Team Name to manage: ")
        if t_name not in self.league.teams:
            print("Team not found.")
            return

        team = self.league.teams[t_name]
        
        # Get current lineup (Custom or Default)
        if t_name in self.custom_lineups:
            current_lineup = self.custom_lineups[t_name]
            print(f"\n[Using CUSTOM Lineup for {t_name}]")
        else:
            current_lineup = team.get_default_11()
            print(f"\n[Using DEFAULT Lineup for {t_name}]")

        print("\n--- Starting XI ---")
        for i, p in enumerate(current_lineup):
            print(f"{i+1}. {p.name} ({p.position}) - Min: {p.minutes_played}")

        print("\n--- Options ---")
        print("S. Swap Player")
        print("R. Reset to Default")
        print("B. Back")

        opt = input("Choice: ").upper()

        if opt == 'R':
            if t_name in self.custom_lineups:
                del self.custom_lineups[t_name]
            print("Reset to default.")
        
        elif opt == 'S':
            idx = int(input("Enter number of player to remove (1-11): ")) - 1
            if idx < 0 or idx > 10: return

            player_out = current_lineup[idx]
            print(f"Removing {player_out.name}...")

            # Show available bench (everyone in squad pool not in current_lineup)
            # Since custom lineup might be a mixed bag, we look at team.squad_pool
            current_ids = {p.name for p in current_lineup}
            bench = [p for p in team.squad_pool.values() if p.name not in current_ids]
            bench.sort(key=lambda x: x.minutes_played, reverse=True)

            print(f"\n--- Bench (Top 10 options) ---")
            for i, p in enumerate(bench[:10]):
                print(f"{i+1}. {p.name} ({p.position})")
            
            b_idx = int(input("Enter number of player to bring in: ")) - 1
            player_in = bench[b_idx]

            # Update Lineup
            new_lineup = current_lineup.copy()
            new_lineup[idx] = player_in
            self.custom_lineups[t_name] = new_lineup
            print(f"Substituted {player_in.name} for {player_out.name}.")

    def menu_create_team(self):
        print("\n--- Create Custom Team ---")
        name = input("Enter new team name: ")
        if name in self.league.teams:
            print("Team already exists!")
            return

        new_team = Team(name)
        
        print("Search for players to add. Type 'DONE' when finished.")
        while True:
            query = input("Search Player: ")
            if query == 'DONE':
                break
            
            # Search in all_players
            matches = [p for p in self.all_players if query.lower() in p.name.lower()]
            
            if not matches:
                print("No matches found.")
                continue
            
            print(f"Found {len(matches)} players:")
            for i, p in enumerate(matches[:5]):
                print(f"{i+1}. {p.name} ({p.squad_name} - {p.position})")
            
            sel = input("Select # to add (or 0 to skip): ")
            if sel.isdigit() and int(sel) > 0 and int(sel) <= len(matches):
                p_obj = matches[int(sel)-1]
                new_team.add_player(p_obj)
                print(f"Added {p_obj.name}.")
            
        if len(new_team.squad_pool) < 11:
            print("Warning: Team has fewer than 11 players. Simulation might crash or be weird.")
        
        self.league.teams[name] = new_team
        print(f"Team {name} added to the league!")

if __name__ == "__main__":
    app = PremierLeagueCLI()
    app.run()
