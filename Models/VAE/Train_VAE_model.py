import pandas as pd
import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from VAE_Model import VAE

def vae_loss(x,x_hat,mu,logvar, beta=1.0):
    # Reconstruction (MSE for continuous features)
    recon = F.mse_loss(x_hat, x, reduction="mean")
    # KL divrgence
    kl = -0.5 *torch.mean(1+logvar-mu.pow(2)-logvar.exp())
    return recon + beta*kl, recon, kl

def train_vae(X_np, latent_dim=16, beta=1.0, lr=1e-3, batch_size=128, epochs=200, device=None):
    device =device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    X = torch.tensor(X_np, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=True)
    
    model = VAE(input_dim=X.shape[1], latent_dim=latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range (1, epochs+1):
        total,rtot,ktot = 0.0,0.0,0.0
        for (xb,) in loader:
            xb = xb.to(device)
            x_hat, mu, logvar, _ = model(xb)
            loss, recon,kl = vae_loss(xb,x_hat,mu,logvar,beta=beta)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            total += loss.item()* xb.size(0)
            rtot += recon.item() * xb.size(0)
            ktot += kl.item() * xb.size(0)
            
        if epoch % 20 == 0 or epoch == 1:
            n = len(loader.dataset)
            print(f"Epoch {epoch:4d} | loss {total/n:.6f} | recon {rtot/n:.6f} | KL {ktot/n:.6f}")
        
    return model

df_new = pd.read_pickle("Input_Data_VAE.pkl")
df_X = df_new
X_np = df_X.values.astype("float32")

model = train_vae(X_np)
print(model)
torch.save(model.state_dict(), "vae_model_AFLOW.pth")