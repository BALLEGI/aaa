import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
import os
from train_vae import VAE
CSV_FILE = '/models/training_data.csv'
MODEL_PATH = '/models/vae_model.pth'
PARAMS_PATH = '/models/scaler_params.json' # Nouveau: Sauvegarde Min/Max
BATCH_SIZE = 32
EPOCHS = 30
LATENT_DIM = 3

# --- DÉFINITION DU MODÈLE VAE (6 entrées) ---
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

def train():
    print("Chargement CSV...")
    df = pd.read_csv(CSV_FILE)
    
    # --- SAUVEGARDE DES STATS SCALER (Min/Max) ---
    # C'est critique pour que l'inférence utilise la même échelle
    min_vals = df.min().to_dict()
    max_vals = df.max().to_dict()
    with open(PARAMS_PATH, 'w') as f:
        json.dump({'min': min_vals, 'max': max_vals}, f)

    # --- NORMALISATION (MinMax) ---
    data = (df - df.min()) / (df.max() - df.min())
    # Gérer les division par zero si une colonne est constante
    data = data.fillna(0)
    
    tensor_x = torch.tensor(data.values, dtype=torch.float32)
    dataset = TensorDataset(tensor_x)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(input_dim=6, latent_dim=LATENT_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"Entraînement {EPOCHS} époques...")
    model.train()
    loss_history = []

    for epoch in range(EPOCHS):
        train_loss = 0
        for batch_idx, (data_batch,) in enumerate(loader):
            data_batch = data_batch.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data_batch)
            
            # Perte BCE + KLD
            BCE = nn.functional.binary_cross_entropy(recon_batch, data_batch, reduction='sum')
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = BCE + KLD
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_loss = train_loss / len(dataset)
        loss_history.append(avg_loss)
        print(f'Epoch {epoch+1}: Loss = {avg_loss:.6f}')

    # --- CALCUL DU SEUIL (95ème percentile des erreurs) ---
    # Au lieu de 0.5, on calcule ce qui est "normal" pendant le training
    model.eval()
    all_errors = []
    with torch.no_grad():
        for (data_batch,) in loader:
            data_batch = data_batch.to(device)
            recon_batch, _, _ = model(data_batch)
            mse = nn.functional.mse_loss(recon_batch, data_batch, reduction='sum')
            all_errors.append(mse.item())
    
    threshold = np.percentile(all_errors, 95) # Seuil à 95%
    print(f"Seuil Anomalie calculé: {threshold:.6f}")
    
    # Sauvegarder le seuil dans le fichier params
    with open(PARAMS_PATH, 'r') as f:
        params = json.load(f)
    params['threshold'] = threshold
    with open(PARAMS_PATH, 'w') as f:
        json.dump(params, f)

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Modèle & Paramètres sauvegardés.")

if __name__ == "__main__":
    train()
