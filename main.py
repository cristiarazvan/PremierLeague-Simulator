import pandas as pd
import numpy as np

# --- Configuration & Constants ---
DATA_PATH = 'data/raw/'
PLAYER_FILE = 'player_stats_2024-25.csv'
TEAM_FILE = 'team_stats_2024-25.csv'

def load_data(path=DATA_PATH):
    try:
        player_data = pd.read_csv(path + PLAYER_FILE)
        team_data = pd.read_csv(path + TEAM_FILE)
        return player_data, team_data
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None, None

def process_teams_data(team_data):
    teams_dict = {}
    if team_data is None:
        return teams_dict

    for index, row in team_data.iterrows():
        team_name = row['Squad']
        teams_dict[team_name] = {
            'MP': row['MP'],
            'W': row['W'],
            'L': row['L'],
            'Draws': row['D'],
            'GF': row['GF'],
            'GA': row['GA'],
            'GD': row['GD'],
            'Pts': row['Pts'],
            'xG': row['xG'],
            'xGA': row['xGA'],
            'xGD': row['xGD'],
            'xGd/90': row['xGD/90'],
            'Top Scorer': row['Top Team Scorer'],
            'Goalkeeper': row['Goalkeeper'],
        }
    return teams_dict

def simplify_position(pos_str):
    if pd.isna(pos_str):
        return "UNK"
    
    # first pos for now 
    primary_pos = pos_str.split(',')[0]
    
    if primary_pos == 'GK':
        return 'GK'
    elif primary_pos == 'DF':
        return 'DEF'
    elif primary_pos == 'MF':
        return 'MID'
    elif primary_pos == 'FW':
        return 'ATT'
    else:
        return 'UNK'

def calculate_skills(row):
    """
    Calculates skills based on stats 
    """
    # Fill NaN with 0 for calculation
    gls = row.get('Gls', 0)
    ast = row.get('Ast', 0)
    xg = row.get('xG', 0)
    xag = row.get('xAG', 0)
    starts = row.get('Starts', 0)
    mins = row.get('Min', 0)
    
    # --- Skill Formulas ---
    # Attacking Skill
    raw_att = (gls * 4) + (ast * 3) + (xg * 20) + (xag * 20) 
    skill_att = round(raw_att, 1)

    # Defensive Skill 
    skill_def = round(starts * 2 + (mins / 50), 1)

    # Goalkeeping Skill
    # Random skill between 0 and 0.1 for non-GK players
    skill_gk = round(starts * 3, 1) if simplify_position(row['Pos']) == 'GK' else np.random.uniform(0, 0.1)


    return skill_def, skill_att, skill_gk

def process_players_data(player_data):
    players_processed = {}
    if player_data is None:
        return players_processed

    for index, row in player_data.iterrows():
        name = row['Player']
        pos_simplified = simplify_position(row['Pos'])
        s_def, s_att, s_gk = calculate_skills(row)
        
        players_processed[name] = {
            'name': name,
            'position': pos_simplified,
            'skill_def': s_def,
            'skill_att': s_att,
            'skill_gk': s_gk,
            'squad': row['Squad']
        }
    return players_processed

def get_team_players(team_name, player_data):
    """
    Returns two lists:
    1. first_11: List of names for the best XI based on starts.
    2. all_players: List of all player names in the squad.
    """
    team_players = player_data[player_data['Squad'] == team_name].copy()
    all_players = team_players['Player'].tolist()
    
    # Select Goalkeeper
    gks = team_players[team_players['Pos'].str.contains("GK", na=False)]
    if not gks.empty:
        best_gk_name = gks.sort_values(by=['Starts', 'Min'], ascending=False).head(1)['Player'].tolist()
    else:
        best_gk_name = []

    # Select Top 10 Outfield Players
    outfield = team_players[~team_players['Pos'].str.contains("GK", na=False)]
    best_outfield_names = outfield.sort_values(by=['Starts', 'Min'], ascending=False).head(10)['Player'].tolist()
    
    first_11 = best_gk_name + best_outfield_names
    return first_11, all_players

def calculate_team_scores(first_11_names, players_processed):
    """
    Calculates the total attacking and defensive scores for a starting 11.
    """
    total_att = 0.0
    total_def = 0.0
    
    for name in first_11_names:
        if name in players_processed:
            player = players_processed[name]
            total_att += player['skill_att']
            # Defensive score includes goalkeeper skill for the team total
            total_def += player['skill_def'] + player['skill_gk']
            
    return round(total_att, 1), round(total_def, 1)

def main():
    print("Loading data...")
    player_data, team_data = load_data()
    
    if player_data is None or team_data is None:
        return

    print("Processing team data...")
    teams_dict = process_teams_data(team_data)
    
    print("Processing player data (calculating skills)...")
    players_processed = process_players_data(player_data)
    
    print("\n--- Example Processed Players ---")
    for name in list(players_processed.keys())[:3]:
        print(players_processed[name])

    print("\n=== Premier League Teams (2024-25) ===")
    for team_name in teams_dict.keys():
        first_11, all_players = get_team_players(team_name, player_data)
        team_att_score, team_def_score = calculate_team_scores(first_11, players_processed)
        
        print(f"\nTeam: {team_name}")
        print(f"First 11: {first_11}")
        print(f"Team Attack Score: {team_att_score}")
        print(f"Team Defense Score: {team_def_score}")
        print("-" * 30)

if __name__ == "__main__":
    main()
