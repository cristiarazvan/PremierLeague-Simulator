import numpy as np
from src.models import Team, Player

class League:
    def __init__(self, players_list):
        self.teams = {}
        self.params = {
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
        self._build_teams(players_list)
        
        self.avg_att_power = 0
        self.avg_def_power = 0
        self._calibrate_league()

    def _build_teams(self, players_list):
        for p in players_list:
            if p.squad_name not in self.teams:
                self.teams[p.squad_name] = Team(p.squad_name)
            self.teams[p.squad_name].add_player(p)

    def _calibrate_league(self):
        """
        Get the average attack and defense power across all teams
        """
        all_att = []
        all_def = []
        
        for t in self.teams.values():
            a, d = t.calculate_power(self.params, specific_lineup_names=None) # Default 11
            all_att.append(a)
            all_def.append(d)
            
        self.avg_att_power = np.mean(all_att) if all_att else 1
        self.avg_def_power = np.mean(all_def) if all_def else 1
        
        print(f"League Calibrated. Avg Att: {self.avg_att_power:.1f}, Avg Def: {self.avg_def_power:.1f}")

    def predict_match(self, home_name, away_name, params, home_lineup=None, away_lineup=None):
        """
        Return the expected goals (lambda) for home and away teams.
        """
        if home_name not in self.teams or away_name not in self.teams:
            print(f"Error: Teams not found {home_name} vs {away_name}")
            return 0, 0

        home_team = self.teams[home_name]
        away_team = self.teams[away_name]

        h_att, h_def = home_team.calculate_power(params, home_lineup)
        a_att, a_def = away_team.calculate_power(params, away_lineup)

        # Comparing to league average 
        h_att_ratio = h_att / self.avg_att_power 
        a_def_ratio = a_def / self.avg_def_power
        
        a_att_ratio = a_att / self.avg_att_power
        h_def_ratio = h_def / self.avg_def_power

        LEAGUE_AVG_GOALS = 1.6 # Hardcoded for now 
        HOME_ADVANTAGE = 1.15 # Hardcoded for now

        # (Atacul Meu / Apararea Ta) * MediaLigii
        lambda_home = (h_att_ratio / a_def_ratio) * LEAGUE_AVG_GOALS * HOME_ADVANTAGE
        lambda_away = (a_att_ratio / h_def_ratio) * LEAGUE_AVG_GOALS * (1/HOME_ADVANTAGE)

        return lambda_home, lambda_away

    def simulate_match_fast(self, h_att, h_def, a_att, a_def, params):
        sigma = params.get('sigma', 0.1)
        noise_home = np.random.normal(0, sigma)
        noise_away = np.random.normal(0, sigma)

        moment_att_home = h_att * (1 + noise_home)
        moment_def_home = h_def * (1 + noise_home)

        moment_att_away = a_att * (1 + noise_away)
        moment_def_away = a_def * (1 + noise_away)

        scaling_factor = params.get('scaling_factor', 250)
        avg_goals = params.get('league_avg_goals', 1.6)
        home_adv = params.get('home_adv', 1.15)

        lambda_home = avg_goals * np.exp((moment_att_home - moment_def_away) / scaling_factor) * home_adv
        lambda_away = avg_goals * np.exp((moment_att_away - moment_def_home) / scaling_factor) * (1/home_adv)

        score_home = np.random.poisson(lambda_home)
        score_away = np.random.poisson(lambda_away)

        return score_home, score_away

    def simulate_match(self, home_name, away_name, params, home_lineup=None, away_lineup=None):
        if home_name not in self.teams or away_name not in self.teams:
            return 0, 0

        home_team = self.teams[home_name]
        away_team = self.teams[away_name]

        attack_home, defense_home = home_team.calculate_power(params, home_lineup)
        attack_away, defense_away = away_team.calculate_power(params, away_lineup)

        return self.simulate_match_fast(attack_home, defense_home, attack_away, defense_away, params)
