import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime

SAVE_DIR = Path("checkpoints")
DICT_SAVE_DIR = Path("dictionaries")

from lexico.omp import omp_v0

class Autoencoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.L = cfg["num_hidden_layers"] * 2
        self.m = cfg["head_dim"]
        self.n = cfg["dictionary_size"]
        self.s = cfg["sparsity"]

        self.D = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(self.L, self.m, self.n)))
        self.D.data[:] = self.D / self.D.norm(dim=-2, keepdim=True)
        self.to(cfg["device"])
    
    @torch.no_grad()
    def encode(self, k):
        DTD = torch.bmm(self.D.permute(0, 2, 1), self.D)
        indices, values, _, _, _, _ = omp_v0(self.D, k.transpose(0, 1), DTD, self.s)
        y = torch.zeros((self.L, k.size(0), self.n), device=self.cfg["device"])
        y.scatter_(-1, indices.to(torch.int64), values.squeeze(-1))
        return y.transpose(0, 1)
    
    def decode(self, y):
        return torch.einsum('lmn,bln->blm', self.D, y)

    def forward(self, k):
        y = self.encode(k)
        k_hat = self.decode(y)
        loss = torch.mean((k_hat - k) ** 2)
        return loss, k_hat, y

    @torch.no_grad()
    def closed_form_update(self, k, y):
        """
        Solve for self.D in closed-form, given:
        - k of shape (B, L, m)
        - y of shape (B, L, n)
        Then apply column-normalization.
        """
        B, L, m = k.shape
        _, _, n = y.shape
        # Sanity check that L == self.L, etc.
        assert L == self.L, "Mismatch in number of layers"

        # For each layer, do the closed-form least squares:
        for l in range(L):
            # Grab the slice for layer l
            # k_l: (B, m),   y_l: (B, n)
            k_l = k[:, l, :]  # shape (B, m)
            y_l = y[:, l, :]  # shape (B, n)

            # D[l] is (m, n)
            # D[l] = (k_l^T * y_l) * (y_l^T * y_l)^(-1)
            # or use torch.linalg.lstsq if you prefer a pseudo-inverse approach.

            # shape checks:
            # k_l.T: (m, B)
            # y_l:   (B, n)
            # => (m, n)
            k_l_t = k_l.transpose(0, 1)  # shape (m, B)
            y_l_t = y_l.transpose(0, 1)  # shape (n, B)

            # Normal equations approach
            # We can do pinverse or direct inverse:
            # (y_l^T y_l) is (n, n)
            # (k_l^T y_l) is (m, n)
            yty = y_l_t @ y_l  # (n, n)
            kty = k_l_t @ y_l  # (m, n)

            # Regular invert or safe pseudo-inverse if rank-deficient
            # We do a small ridge if needed, or just inverse if guaranteed full-rank
            # E.g.: yty_inv = torch.inverse(yty + 1e-5 * torch.eye(n, device=yty.device))
            yty_inv = torch.linalg.pinv(yty)  # safer than .inverse()

            D_l = kty @ yty_inv  # shape (m, n)

            self.D[l].data = D_l  # Assign back

        # Now normalize columns of each D[l] to unit norm (same as normalise_decoder_weights)
        # self.D is shape (L, m, n)
        self.D.data[:] = self.D / self.D.norm(dim=-2, keepdim=True)
    
    @torch.no_grad()
    def normalise_decoder_weights(self):
        D_normalised = self.D / self.D.norm(dim=-2, keepdim=True)
        D_grad_proj = (self.D.grad * D_normalised).sum(-2, keepdim=True) * D_normalised
        self.D.grad -= D_grad_proj
        self.D.data = D_normalised
    
    def save(self):
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        model_path = SAVE_DIR / f"{self.cfg['name']}.pt"
        torch.save(self.state_dict(), model_path)
    
    def save_dictionary(self):
        DICT_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        D_path = DICT_SAVE_DIR / f"{self.cfg['name']}.pt"
        torch.save(self.D.detach().cpu(), D_path)
    
    @classmethod
    def load(cls, model_path, cfg):
        model = cls(cfg=cfg)
        model.load_state_dict(torch.load(model_path))
        return model