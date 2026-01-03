import numpy as np
import pandas as pd

def simplify_position(pos_str):
    if pd.isna(pos_str): 
        return "UNK"
    # FIrst position for now....
    primary_pos = pos_str.split(',')[0]
    
    mapping = {
        'GK': 'GK',
        'DF': 'DEF',
        'MF': 'MID',
        'FW': 'ATT'
    }
    return mapping.get(primary_pos, 'UNK')

def calculate_player_metrics(row):
    gls = row.get('Gls', 0)
    ast = row.get('Ast', 0)
    xg = row.get('xG', 0)
    xag = row.get('xAG', 0)
    starts = row.get('Starts', 0)
    mins = row.get('Min', 0)
    pos = simplify_position(row.get('Pos'))

    prg_c = row.get('PrgC', 0)
    prg_p = row.get('PrgP', 0)
    prg_r = row.get('PrgR', 0)

    # Weighted sum
    raw_att = (gls * 4) + (ast * 3) + (xg * 15) + (xag * 15)
    
    # Adding progression contributions
    raw_att += (prg_c * 0.2) + (prg_p * 0.2) + (prg_r * 0.2)

    skill_att = round(raw_att, 1)

    # TODO add better defensive metrics
    skill_def = round(starts * 2 + (mins / 45), 1)

    # Portar
    skill_gk = 0.0
    if pos == 'GK':
        skill_gk = round(starts * 3 + (mins / 30), 1)
    else:

        # random for non gk
        skill_gk = np.random.uniform(0, 0.1)

    return skill_def, skill_att, skill_gk

def calculate_scaling_factor_and_avg_goals(league, params):
    # print("--- CALIBRARE AUTOMATA SCALING FACTOR ---")
    max_attack_power = 0
    min_defense_power = 9999999

    # Iteram prin toate echipele incarcate
    for team in league.teams:
        att, def_ = league.teams[team].calculate_power(params)

        if att > max_attack_power:
            max_attack_power = att
        if def_ < min_defense_power:
            min_defense_power = def_

    delta_max_skill = max_attack_power - min_defense_power
    # print(f"1. Delta Skill Maxim (FIFA): {delta_max_skill:.2f} puncte")

    # PASUL 2: Analiza Istoricului (Realitate)
    # Aici ar trebui sa citesti din CSV.
    # Daca nu ai CSV-ul incarcat acum, punem valorile reale din Premier League 23/24.


    avg_league_goals = 0
    best_team = ""
    best_team_score = 0
    avg_best_team_goals = 0
    for team in league.teams:
        avg_league_goals += league.teams[team].goals_per_match
        if league.teams[team].points > best_team_score:
            best_team = team
            best_team_score = league.teams[team].points
    avg_league_goals /= len(league.teams)
    avg_best_team_goals = league.teams[best_team].goals_per_match

    ratio = avg_best_team_goals / avg_league_goals
    log_ratio = np.log(ratio)
    calculated_factor = delta_max_skill / log_ratio

    return calculated_factor, avg_league_goals

def compute_error(predicted_scores, actual_scores):
    mse = 0.0
    n = len(predicted_scores)
    for i in range(n):
        pred_points = predicted_scores[i]['Points']
        actual_points = actual_scores[i]['Points']
        mse += (pred_points - actual_points) ** 2

    return mse
