import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=16, hidden_dims=(256, 128)):
        super().__init__()
        h1, h2 = hidden_dims
        
        ## Encoder
        self.enc = nn.Sequential(
            nn.Linear(input_dim,h1),
            nn.ReLU(),
            nn.Linear(h1,h2),
            nn.ReLU(),
        )
        self.mu = nn.Linear(h2,latent_dim)
        self.logvar = nn.Linear(h2,latent_dim)
        
        # Decoder
        self.dec = nn.Sequential(
            nn.Linear(latent_dim,h2),
            nn.ReLU(),
            nn.Linear(h2,h1),
            nn.ReLU(),
            nn.Linear(h1,input_dim)
            )
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
        
    def forward(self,x):
        h = self.enc(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        z = self.reparameterize(mu, logvar)
        x_hat = self.dec(z)
        return x_hat, mu, logvar, z

