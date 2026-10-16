import csv
import math

import numpy as np
import torch
from torch import Tensor

MIN_STD = 0.02  # 1 unit on the original 0-100 scale; avoids dividing by ~0


def _load_tsv(path: str) -> tuple[list[str], np.ndarray]:
    with open(path, newline="") as f:
        rows = [r for r in csv.reader(f, delimiter="\t") if r]
    labels = [r[0] for r in rows[1:]]
    values = np.array([[float(x) for x in r[1:]] for r in rows[1:]], dtype=np.float32)
    return labels, values


class Dataset:
    def __init__(self, means_path: str, stds_path: str, device: str) -> None:
        self.device = device
        self.labels, means = _load_tsv(means_path)
        _, stds = _load_tsv(stds_path)
        assert means.shape == stds.shape, "means and stds files differ in size"
        self.means = torch.from_numpy(means / 50 - 1).to(device)  # 0..100 -> -1..1
        self.stds = torch.from_numpy(stds / 50).to(device)  # scale only, no shift
        self.fill = self.means.mean(0)  # stands in for dropped values
        self.features = means.shape[1]

    def noisy(self, idx: Tensor, generator: torch.Generator | None = None) -> Tensor:
        """One normally distributed sample per object index."""
        mean, std = self.means[idx], self.stds[idx]
        noise = torch.randn(mean.shape, device=self.device, generator=generator)
        return (mean + std * noise).clamp(-1, 1)

    def sparse_input(self, values: Tensor, mask: Tensor) -> Tensor:
        """Encoder input: values (dropped ones replaced by the fill) followed by the mask."""
        return torch.cat([torch.where(mask, values, self.fill), mask.float()], -1)

    def sample(
        self,
        n: int,
        keep: tuple[float, float],
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Returns (encoder input, clean means, object indices) for n random objects.
        Each sample draws its own keep rate, log-uniform between keep[0] and keep[1]."""
        idx = torch.randint(
            len(self.labels), (n,), device=self.device, generator=generator
        )
        low, high = math.log(keep[0]), math.log(keep[1])
        draw = torch.rand(n, 1, device=self.device, generator=generator)
        values = self.noisy(idx, generator)
        mask = torch.rand(
            values.shape, device=self.device, generator=generator
        ) < torch.exp(low + (high - low) * draw)
        return self.sparse_input(values, mask), self.means[idx], idx

    def distances(self, vec: Tensor) -> tuple[Tensor, Tensor]:
        """Per-object (mean squared error, mean squared z-score) of a 500D vector.
        A z-score distance of ~1 means the vector sits about one std from the object's mean.
        """
        diff = vec - self.means
        return diff.pow(2).mean(1), (diff / self.stds.clamp_min(MIN_STD)).pow(2).mean(1)
