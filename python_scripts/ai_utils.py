import json
import torch

def extract_features(data):
    """
    Extrait et normalise les features d'un log Zeek JSON.
    Retourne un vecteur [dst_port, protocol, duration, orig_bytes, resp_bytes, state_code]
    """
    # 1. Features brutes
    dst_port = float(data.get('id.resp_p', 0))
    protocol = 1.0 if data.get('proto') == 'tcp' else 0.0
    duration = float(data.get('duration', 0))
    orig_bytes = float(data.get('orig_bytes', 0))
    resp_bytes = float(data.get('resp_bytes', 0))
    state_str = data.get('conn_state', 'S0')
    
    # 2. Encodage simple de l'état (State Mapping)
    # Mapping: S0=0, S1=1, S2=2, S3=3, REJ=4, RSTO=5, RSTR=6, RSTOS0=7, RSTH=8, SF=9, SH=10
    state_map = {'S0':0, 'S1':1, 'S2':2, 'S3':3, 'REJ':4, 'RSTO':5, 'RSTR':6, 'RSTOS0':7, 'RSTH':8, 'SF':9, 'SH':10}
    state_code = float(state_map.get(state_str, 0))
    
    # 3. Normalisation Logique (avant MinMax Scaling)
    # On ramène les valeurs vers 0-1 approximatif pour aider le scaler
    norm_port = dst_port / 65535.0
    norm_duration = duration / 60.0  # En minutes
    norm_orig_bytes = orig_bytes / 10000.0 # En Ko
    norm_resp_bytes = resp_bytes / 10000.0 # En Ko

    return [norm_port, protocol, norm_duration, norm_orig_bytes, norm_resp_bytes, state_code]
