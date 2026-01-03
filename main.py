from src import league
from src.data_loader import load_players_from_csv
from src.league import League
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Config
PLAYER_CSV = 'data/raw/player_stats_2024-25.csv'

# Simulated Parameters - HARDCODED fot now
sim_params = {
            'sigma': 0.15,  # 15% variability (luck/form)
            'scaling_factor': 1500,  # Normalize power difference
            'league_avg_goals': 1.6,
            'home_adv': 1.0,
            'weights' : {
                        'ATT': {'att': 1.0, 'def': 0.15},
                        'MID': {'att': 0.7, 'def': 0.6},
                        'DEF': {'att': 0.1, 'def': 1.0},
                        'GK':  {'att': 0.0, 'def': 0.0},
                        'UNK': {'att': 0.5, 'def': 0.5}
                    }
        }

def Monte_Carlo_Match(h_team, a_team, league, sim_params):
    print("\n--- Monte Carlo Simulation (" + h_team + " vs " + a_team + ") ---")

    num_sims = 2000

    wins_h = 0
    wins_a = 0
    draws = 0

    print(f"Running {num_sims} simulations...")

    for _ in range(num_sims):
        gh, ga = league.simulate_match(h_team, a_team, sim_params)

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

def Monte_Carlo_League(league, sim_params, lineups = None):
    if lineups is None:
        lineups = {}
        for team in league.teams:
            lineups[team] = None
    else:
        for team in league.teams:
            if team not in lineups:
                lineups[team] = None

    Rankings = {}
    for team in league.teams:
        Rankings[team] = {}
        for place in range(1, len(league.teams) + 1):
            Rankings[team][place] = 0
        Rankings[team]["goals_per_match"] = 0
        Rankings[team]["points"] = 0

    num_sims = 2000
    for _ in range(num_sims):
        Goals = {}
        Points = {}
        for team in league.teams:
            Goals[team] = 0
            Points[team] = 0

        # League
        for team1 in league.teams:
            for team2 in league.teams:
                if team1 == team2:
                    continue

                gh, ga = league.simulate_match(team1, team2, sim_params, lineups[team1], lineups[team2])
                Goals[team1] += gh
                Goals[team2] += ga

                if gh > ga:
                    Points[team1] += 3
                elif ga > gh:
                    Points[team2] += 3
                else:
                    Points[team1] += 1
                    Points[team2] += 1

        ranking = sorted(Points.items(), key=lambda x: (x[1], Goals[x[0]]), reverse=True)

        for i in range(len(ranking)):
            team = ranking[i][0]
            Rankings[team][i + 1] += 1
            Rankings[team]["goals_per_match"] += (Goals[team] / (len(league.teams) - 1) / 2.0)
            Rankings[team]["points"] += Points[team]
            # print(str(i + 1) + ": " + ranking[i][0])

    for team in league.teams:
        for place in range(1, len(league.teams) + 1):
            Rankings[team][place] = Rankings[team][place] / num_sims * 100
        Rankings[team]["goals_per_match"] /= num_sims
        Rankings[team]["points"] /= num_sims


    # df = pd.DataFrame(Rankings)
    # print(df)
    return Rankings

def main():
    print("--- 1. Loading Data ---")
    all_players = load_players_from_csv(PLAYER_CSV)
    
    if not all_players:
        print("No players loaded. Exiting.")
        return

    print("--- 2. Building League Framework ---")
    my_league = League(all_players)

    # 3. MONTE CARLO SIMULATION - MATCH
    # Monte_Carlo_Match("Liverpool", "Manchester City", my_league, sim_params)

    # 4. MONTE CARLO SIMULATION - LEAGUE
    Monte_Carlo_League(my_league, sim_params)
    
if __name__ == "__main__":
    main()
