import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphNorm, BatchNorm
from torch_geometric.nn.conv import TransformerConv

def get_norm(norm_type, hidden_dim):
    if norm_type == "layer":
        return nn.LayerNorm(hidden_dim)
    elif norm_type == "graph":
        return GraphNorm(hidden_dim)
    elif norm_type == "none":
        return nn.Identity()
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")


class NodeCoordEncoder(nn.Module):
    def __init__(self, coord_dim, hidden_channels, latent_dim, norm_type="layer"):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.latent_embedding_size = latent_dim

        self.conv1 = TransformerConv(coord_dim, 
                                    self.hidden_channels, 
                                    heads=4, 
                                    concat=False,
                                    beta=True)
        
        self.bn1 = BatchNorm(self.hidden_channels)
        self.conv2 = TransformerConv(self.hidden_channels, 
                                    self.hidden_channels, 
                                    heads=4, 
                                    concat=False,
                                    beta=True)
        self.bn2 = BatchNorm(self.hidden_channels)

        self.conv3 = TransformerConv(self.hidden_channels, 
                                    self.hidden_channels, 
                                    heads=4, 
                                    concat=False,
                                    beta=True)
        self.bn3 = BatchNorm(self.hidden_channels)

        self.conv4 = TransformerConv(self.hidden_channels, 
                                    self.hidden_channels, 
                                    heads=4, 
                                    concat=False,
                                    beta=True)

        self.gcn_mu = GCNConv(hidden_channels, latent_dim)
        self.gcn_logvar = GCNConv(hidden_channels, latent_dim)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.bn1(x)
        x = self.conv2(x, edge_index).relu()
        x = self.bn2(x)
        x = self.conv3(x, edge_index).relu()
        x = self.bn3(x)
        x = self.conv4(x, edge_index).relu()
        mu = self.gcn_mu(x, edge_index)
        logvar = self.gcn_logvar(x, edge_index)
        return mu, logvar


class NodeCoordDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_channels, coord_dim, norm_type="layer"):
        super().__init__()
        self.gcn1 = GCNConv(latent_dim, hidden_channels)
        self.norm1 = get_norm(norm_type, hidden_channels)
        self.gcn2 = GCNConv(hidden_channels, hidden_channels)
        self.norm2 = get_norm(norm_type, hidden_channels)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, coord_dim)
        )
        self.dropout = nn.Dropout(p=0)

    def forward(self, z, edge_index):
        z = self.norm1(F.relu(self.gcn1(z, edge_index)))
        z = self.dropout(z)
        z = self.norm2(F.relu(self.gcn2(z, edge_index)))
        z = self.dropout(z)
        return self.mlp(z)


class NodeCoordVAE(nn.Module):
    def __init__(self, coord_dim, hidden_channels, latent_dim, norm_type="layer"):
        super().__init__()
        self.coord_dim = coord_dim
        self.hidden_channels = hidden_channels
        self.latent_dim = latent_dim

        self.encoder = NodeCoordEncoder(coord_dim, hidden_channels, latent_dim, norm_type)
        self.decoder = NodeCoordDecoder(latent_dim, hidden_channels, coord_dim, norm_type)

    def encode(self, x, edge_index):
        mu, logvar = self.encoder(x, edge_index)
        return mu, logvar

    def reparameterize(self, mu, logvar, random=True):
        if not random:
            return mu
        else:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

    def decode(self, z, edge_index):
        return self.decoder(z, edge_index)

    def forward(self, x, edge_index, random=True):
        
        mu, logvar = self.encode(x, edge_index)
        
        z = self.reparameterize(mu, logvar, random=random)
       
        x_recon = self.decode(z, edge_index)
     
        return x_recon, mu, logvar

    def get_loss(self, x, x_recon, mu, logvar, beta=1.0):
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = recon_loss + beta * kl_loss
        return total_loss, recon_loss, kl_loss
