import torch
import torch.nn as nn


class TrajectoryEncoder(nn.Module):
    def __init__(self, input_dim=8, d_model=128, nhead=8, num_layers=2, target_traj_len=20):
        super().__init__()
        self.target_traj_len = target_traj_len
        self.embed = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=target_traj_len)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead),
            num_layers=num_layers
        )

    def forward(self, x):
        # x: (batch, T, 8)
        x = self.embed(x)  # (batch, T, D)
        x = x.permute(1, 0, 2)  # (T, batch, D)
        x = self.pos_encoder(x)
        x = self.transformer(x)  # (T, batch, D)
        return x.permute(1, 0, 2)  # (batch, T, D)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 20):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        self.register_buffer('pe', pe)  # (max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (T, batch_size, d_model) or (batch_size, T, d_model)
        Returns:
            (same shape as x) + positional encoding
        """
        if x.dim() == 3 and x.size(2) == self.pe.size(1):
            # For (T, batch, d_model)
            x = x + self.pe[:x.size(0), :].unsqueeze(1)
        elif x.dim() == 3 and x.size(1) == self.pe.size(1):
            # For (batch, T, d_model)
            x = x + self.pe[:x.size(1), :].unsqueeze(0)
        else:
            raise ValueError("Input shape not supported")
        return x