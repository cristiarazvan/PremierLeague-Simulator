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
