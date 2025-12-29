import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
from kafka import KafkaConsumer, KafkaProducer
import ai_utils

# --- CONFIGURATION ---
KAFKA_SERVER = os.environ.get('KAFKA_SERVER', 'kafka:29092')
INPUT_TOPIC = os.environ.get('NETWORK_TOPIC', 'network_logs')
ALERT_TOPIC = 'ai_anomaly_alerts'
MODEL_PATH = '/models/vae_model.pth'
PARAMS_PATH = '/models/scaler_params.json'

# --- CHARGEMENT DU MODÈLE ---
class VAE(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=10, latent_dim=3):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

if __name__ == "__main__":
    # 1. Charger Paramètres (Min/Max et Seuil)
    if not os.path.exists(PARAMS_PATH):
        print("ERREUR: Fichier scaler_params.json introuvable. Veuillez entraîner le modèle d'abord.")
        exit(1)

    with open(PARAMS_PATH, 'r') as f:
        params = json.load(f)
    
    min_vals = params['min']
    max_vals = params['max']
    threshold = params['threshold']
    print(f"Modèle chargé. Seuil Anomalie: {threshold}")

    # 2. Charger le modèle
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(input_dim=6).to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # 3. Setup Kafka
    consumer = KafkaConsumer(INPUT_TOPIC, bootstrap_servers=[KAFKA_SERVER], value_deserializer=lambda x: json.loads(x.decode('utf-8')))
    producer = KafkaProducer(bootstrap_servers=KAFKA_SERVER, value_serializer=lambda v: json.dumps(v).encode('utf-8'))

    print("Inférence IA démarrée...")

    for message in consumer:
        data = message.value
        if data.get('_path') == 'conn':
            try:
                # Feature Engineering
                features = ai_utils.extract_features(data)
                
                # Normalisation (Même logique que training)
                features_np = []
                for i, key in enumerate(['port', 'proto', 'duration', 'orig_bytes', 'resp_bytes', 'state']):
                    min_v = min_vals[key]
                    max_v = max_vals[key]
                    val = features[i]
                    norm = 0.0
                    if max_v != min_v:
                        norm = (val - min_v) / (max_v - min_v)
                    features_np.append(norm)
                
                tensor_input = torch.tensor(features_np, dtype=torch.float32).unsqueeze(0).to(device)

                # Inference
                with torch.no_grad():
                    recon, _, _ = model(tensor_input)
                    mse = nn.functional.mse_loss(recon, tensor_input)
                    anomaly_score = mse.item()

                if anomaly_score > threshold:
                    # ANOMALIE DÉTECTÉE
                    print(f"[!] ANOMALIE (Score:{anomaly_score:.4f} > {threshold:.4f}) | {data.get('id.orig_h')} -> {data.get('id.resp_h')}:{data.get('id.resp_p')}")
                    
                    alert = {
                        "type": "AI_VAE_ANOMALY",
                        "score": anomaly_score,
                        "src_ip": data.get('id.orig_h'),
                        "dst_ip": data.get('id.resp_h'),
                        "dst_port": data.get('id.resp_p'),
                        "description": "Comportement anormal détecté (Zero-Day potentiel)"
                    }
                    producer.send(ALERT_TOPIC, value=alert)
            
            except Exception as e:
                print(f"Erreur inference: {e}")
