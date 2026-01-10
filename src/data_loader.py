import pandas as pd
from src.models import Player


TEAM_NAME_MAPPING = {
    'Nott\'ham Forest': 'Nottingham Forest',
    'Newcastle Utd': 'Newcastle United',
    'Brighton': 'Brighton & Hove Albion',
    'Bournemouth': 'AFC Bournemouth',
    'West Ham': 'West Ham United',
    'Wolves': 'Wolverhampton Wanderers',
    'Manchester Utd': 'Manchester United',
    'Spurs': 'Tottenham Hotspur',
    'Tottenham': 'Tottenham Hotspur',
}


def standardize_team_name(name: str) -> str:
    """Convert team name to standard form"""
    return TEAM_NAME_MAPPING.get(name, name)


def load_players_from_csv(filepath):
    try:
        df = pd.read_csv(filepath)
        df.fillna(0, inplace=True)

        if 'Squad' not in df.columns and 'Team' in df.columns:
            df['Squad'] = df['Team']

        # Standardize team names
        if 'Squad' in df.columns:
            df['Squad'] = df['Squad'].apply(standardize_team_name)

        players_objects = []
        for _, row in df.iterrows():
            players_objects.append(Player(row))

        return players_objects
    except Exception as e:
        print(f"Error loading players CSV: {e}")
        return []


def load_teams_from_csv(filepath):
    """Load unique team names from CSV"""
    try:
        df = pd.read_csv(filepath)
        df.fillna(0, inplace=True)

        team_col = 'Squad' if 'Squad' in df.columns else 'name'
        teams = df[team_col].unique().tolist()
        return [standardize_team_name(t) for t in teams]
    except Exception as e:
        print(f"Error loading teams CSV: {e}")
        return []


def load_standings_from_csv(filepath) -> pd.DataFrame:
    try:
        df = pd.read_csv(filepath)
        df.fillna(0, inplace=True)

        if 'name' in df.columns:
            if 'scoresStr' in df.columns:
                df[['GF', 'GA']] = df['scoresStr'].str.split('-', expand=True).astype(int)

            standardized = pd.DataFrame({
                'Squad': df['name'].apply(standardize_team_name),
                'Pts': df['pts'],
                'W': df.get('wins', 0),
                'D': df.get('draws', 0),
                'L': df.get('losses', 0),
                'GF': df.get('GF', 0),
                'GA': df.get('GA', 0),
                'MP': df.get('played', 0)
            })
        else:
            standardized = pd.DataFrame({
                'Squad': df['Squad'].apply(standardize_team_name),
                'Pts': df['Pts'],
                'W': df.get('W', 0),
                'D': df.get('D', 0),
                'L': df.get('L', 0),
                'GF': df.get('GF', 0),
                'GA': df.get('GA', 0),
                'MP': df.get('MP', 0)
            })

        return standardized

    except Exception as e:
        print(f"Error loading standings CSV: {e}")
        return pd.DataFrame()


def compute_league_stats(standings: pd.DataFrame) -> dict:
    if standings.empty:
        return {}

    total_goals = standings['GF'].sum()
    total_matches = standings['MP'].sum() / 2

    return {
        'total_goals': int(total_goals),
        'total_matches': int(total_matches),
        'avg_goals_per_match': total_goals / total_matches if total_matches > 0 else 0,
        'avg_goals_per_team_per_match': total_goals / standings['MP'].sum() if standings['MP'].sum() > 0 else 0,
    }
