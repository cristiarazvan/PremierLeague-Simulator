import pandas as pd
from src.models import Player

def load_players_from_csv(filepath):
    try:
        df = pd.read_csv(filepath)
        df.fillna(0, inplace=True)
        
        players_objects = []
        for _, row in df.iterrows():
            players_objects.append(Player(row))
            
        return players_objects
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return []

def load_teams_from_csv(filepath):
    try:
        df = pd.read_csv(filepath)
        df.fillna(0, inplace=True)
        
        teams = df['Squad'].unique().tolist()
        return teams
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return []
