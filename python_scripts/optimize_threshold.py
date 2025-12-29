import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import json
import os
from torch.utils.data import DataLoader, TensorDataset
from train_vae import VAE
# Configuration
CSV_FILE = '/models/training_data.csv'
PARAMS_PATH = '/models/scaler_params.json'
MODEL_PATH = '/models/vae_model.pth'

# On redéfinit la classe VAE ici pour faire ce script autonome (robustesse)
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

def optimize_threshold():
    print("--- OPTIMISATION DU SEUIL D'ANOMALIE ---")
    
    # 1. Charger données et params
    if not os.path.exists(PARAMS_PATH):
        print("ERREUR: Fichier scaler_params.json introuvable. Entraînez d'abord le modèle.")
        return

    df = pd.read_csv(CSV_FILE)
    min_vals = df.min().to_dict()
    max_vals = df.max().to_dict()
    
    # Normalisation
    data = (df - df.min()) / (df.max() - df.min())
    data = data.fillna(0)
    
    tensor_x = torch.tensor(data.values, dtype=torch.float32)
    dataset = TensorDataset(tensor_x)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # 2. Charger Modèle
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(input_dim=6).to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # 3. Calculer l'erreur de reconstruction pour TOUTES les données d'entraînement
    all_errors = []
    print("Calcul des erreurs sur le dataset d'entraînement...")
    with torch.no_grad():
        for (data_batch,) in loader:
            data_batch = data_batch.to(device)
            recon_batch, _, _ = model(data_batch)
            # MSE per batch
            mse = nn.functional.mse_loss(recon_batch, data_batch, reduction='sum')
            all_errors.append(mse.item())
    
    all_errors = np.array(all_errors)
    
    # 4. Stats & Suggestions de Seuils
    print(f"Erreur Moyenne : {np.mean(all_errors):.6f}")
    print(f"Max Erreur     : {np.max(all_errors):.6f}")
    
    p95 = np.percentile(all_errors, 95)
    p98 = np.percentile(all_errors, 98)
    p99 = np.percentile(all_errors, 99)
    
    print(f"\n  95ème percentile : {p95:.6f} (Sensible)")
    print(f"  98ème percentile : {p98:.6f} (Standard)")
    print(f"  99ème percentile : {p99:.6f} (Strict)")

    # On choisit le 99ème pour réduire les Faux Positifs
    new_threshold = p99
    print(f"\n-> Nouveau seuil sélectionné : {new_threshold:.6f}")

    # 5. Mise à jour du fichier JSON
    with open(PARAMS_PATH, 'r') as f:
        params = json.load(f)
    
    params['threshold'] = new_threshold
    
    with open(PARAMS_PATH, 'w') as f:
        json.dump(params, f)
    
    print(f"-> Seuil mis à jour dans {PARAMS_PATH}")
    print("Relancez 'ai_inference' pour appliquer le nouveau seuil (si nécessaire).")

if __name__ == "__main__":
    optimize_threshold()
