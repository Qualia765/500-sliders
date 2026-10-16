import json
import struct

import numpy as np
import torch
from torch import nn


class Autoencoder(nn.Module):
    """Sparse input (values + mask) -> latent -> full profile."""

    def __init__(self, features: int, latent: int) -> None:
        super().__init__()
        self.latent = latent
        self.encoder = nn.Sequential(
            nn.Linear(2 * features, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, latent),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, 128),
            nn.ReLU(),
            nn.Linear(128, features),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def save_npz(self, path: str) -> None:
        """Plain numpy arrays, one weight and one bias per layer.
        Each layer computes act(W @ x + b) with W shaped (out, in).
        Encoder input: the feature values (dropped ones filled with the feature average) then the mask.
        Keys: encoder.0 (2*features->512, ReLU), encoder.2 (512->128, ReLU), encoder.4 (128->latent, tanh),
        decoder.0 (latent->128, ReLU), decoder.2 (128->features, tanh)."""
        np.savez(path, **{k: v.cpu().numpy() for k, v in self.state_dict().items()})

    def save_web(self, path: str, fill: torch.Tensor) -> None:
        """Format read by sparse-autoencoder.js: a uint32 header length, a JSON header
        ({array name: shape}, space-padded to a multiple of 4 bytes), then every array as
        little-endian float32 in header order. `fill` is the per-feature average used for dropped values.
        """
        arrays = {"fill": fill, **self.state_dict()}
        header = json.dumps({k: list(v.shape) for k, v in arrays.items()}).encode()
        header += b" " * (-len(header) % 4)
        with open(path, "wb") as f:
            f.write(struct.pack("<I", len(header)) + header)
            for v in arrays.values():
                f.write(v.cpu().numpy().astype("<f4").tobytes())

    @classmethod
    def from_npz(cls, path: str) -> "Autoencoder":
        arrays = np.load(path)
        model = cls(
            arrays["encoder.0.weight"].shape[1] // 2,
            arrays["encoder.4.weight"].shape[0],
        )
        model.load_state_dict({k: torch.from_numpy(arrays[k]) for k in arrays.files})
        return model
