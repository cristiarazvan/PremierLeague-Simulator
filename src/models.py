from src.utils import simplify_position, calculate_player_metrics

class Player:
    def __init__(self, row):
        self.name = row['Player']
        self.squad_name = row['Squad']
        self.raw_pos = row.get('Pos', 'UNK')
        self.position = simplify_position(self.raw_pos)
        self.minutes_played = row.get('Min', 0)
        
        self.s_def, self.s_att, self.s_gk = calculate_player_metrics(row)

    def __repr__(self):
        return f"{self.name} ({self.position})"

class Team:
    def __init__(self, name):
        self.name = name
        self.squad_pool = {} # Dict with players by name 
        
    def add_player(self, player):
        self.squad_pool[player.name] = player

    def get_default_11(self):
        """
        Best 11 based on minutes played.
        """
        all_players = list(self.squad_pool.values())
        
        gks = [p for p in all_players if p.position == 'GK']
        top_gk = sorted(gks, key=lambda p: p.minutes_played, reverse=True)[:1]
        
        outfield = [p for p in all_players if p.position != 'GK']
        top_10 = sorted(outfield, key=lambda p: p.minutes_played, reverse=True)[:10]
        
        return top_gk + top_10

    def calculate_power(self, specific_lineup_names=None):
        """
        specific_lineup_names: List of player names to use as lineup.

        """
        active_players = []
        
        if specific_lineup_names:
            for name in specific_lineup_names:
                if name in self.squad_pool:
                    active_players.append(self.squad_pool[name])
                else:
                    pass 
            
        # Default 11 if wrong lineup
        else:
            active_players = self.get_default_11()

        total_att = 0.0
        total_def = 0.0
        has_gk = False

        weights = {
            'ATT': {'att': 1.0, 'def': 0.15},
            'MID': {'att': 0.7, 'def': 0.6},
            'DEF': {'att': 0.1, 'def': 1.0},
            'GK':  {'att': 0.0, 'def': 0.0},
            'UNK': {'att': 0.5, 'def': 0.5}
        }

        for p in active_players:
            w = weights.get(p.position, weights['UNK'])
            
            total_att += p.s_att * w['att']
            
            if p.position == 'GK':
                total_def += p.s_gk * 3.5
                has_gk = True
            else:
                total_def += p.s_def * w['def']

        # special case for no gk
        if not has_gk:
            total_def *= 0.4

        return total_att, total_def
