import argparse
from typing import NamedTuple

import torch
from torch import Tensor

from data import Dataset
from model import Autoencoder
from train import Trainer

TOP = 5


class Paths(NamedTuple):
    means: str
    stds: str
    weights: str
    checkpoint: str
    web_model: str


def run(paths: Paths) -> None:
    parser = argparse.ArgumentParser(description="Sparse feature autoencoder")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="train; resumes from the checkpoint if there is one")
    train.add_argument("--keep", type=float, nargs="+", default=[0.01, 0.10], metavar="RATE",
                       help="fraction of features kept: one value, or LOW HIGH for a log-uniform range per sample")
    train.add_argument("--fresh", action="store_true", help="ignore the checkpoint and start over")
    train.add_argument("--latent", type=int, help="latent size for a fresh run (default 50); a resumed run keeps its own")
    obj = commands.add_parser("encode-object", help="row index (0-based) with random features dropped -> latent")
    obj.add_argument("row", type=int)
    obj.add_argument("--keep", type=float, default=0.02, help="chance of keeping each feature")
    values = commands.add_parser("encode-values", help="feature number and value pairs -> latent")
    values.add_argument("pairs", type=float, nargs="+", metavar="FEATURE VALUE",
                        help="feature numbers start at 1; values are in -1..1")
    decode = commands.add_parser("decode", help="latent values -> best matching objects")
    decode.add_argument("latent", type=float, nargs="+", help="latent values in -1..1, as many as the model has")
    commands.add_parser("export-web", help="write the model file for sparse-autoencoder.js")
    args = parser.parse_args()

    if args.command == "train":
        if len(args.keep) > 2:
            parser.error("--keep takes one value or LOW HIGH")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        data = Dataset(paths.means, paths.stds, device)
        keep = (args.keep[0], args.keep[-1])
        Trainer(data, keep, args.latent, paths.weights, paths.checkpoint, resume=not args.fresh).run()
        return

    data = Dataset(paths.means, paths.stds, "cpu")
    model = Autoencoder.from_npz(paths.weights)
    with torch.no_grad():
        if args.command == "encode-object":
            _encode_object(data, model, args.row, args.keep)
        elif args.command == "encode-values":
            _encode_values(data, model, args.pairs)
        elif args.command == "export-web":
            model.save_web(paths.web_model, data.fill)
            print(f"wrote {paths.web_model}")
        else:
            _decode(data, model, torch.tensor(args.latent))


def _print_latent(latent: Tensor) -> None:
    print(" ".join(f"{v:.4f}" for v in latent.tolist()))  # pasteable into `decode`


def _encode_object(data: Dataset, model: Autoencoder, row: int, keep: float) -> None:
    if not 0 <= row < len(data.labels):
        raise SystemExit(f"row must be between 0 and {len(data.labels) - 1}")
    values = data.noisy(torch.tensor([row]))[0]
    mask = torch.rand(data.features) < keep
    print(f"row {row}: {data.labels[row]} (kept {int(mask.sum())} of {data.features} features)")
    _print_latent(model.encoder(data.sparse_input(values, mask)))


def _encode_values(data: Dataset, model: Autoencoder, pairs: list[float]) -> None:
    if len(pairs) % 2:
        raise SystemExit("give feature number and value pairs, like: 1 0.6 93 -0.1")
    values = torch.zeros(data.features)
    mask = torch.zeros(data.features, dtype=torch.bool)
    for feature, value in zip(pairs[0::2], pairs[1::2]):
        if not 1 <= feature <= data.features or feature != int(feature):
            raise SystemExit(f"feature numbers are whole numbers from 1 to {data.features}, got {feature:g}")
        values[int(feature) - 1], mask[int(feature) - 1] = value, True
    _print_latent(model.encoder(data.sparse_input(values, mask)))


def _decode(data: Dataset, model: Autoencoder, latent: Tensor) -> None:
    if len(latent) != model.latent:
        raise SystemExit(f"this model has a {model.latent}D latent, got {len(latent)} values")
    mse, z = data.distances(model.decoder(latent))
    for name, dist in (("MSE vs means", mse), ("std-weighted (mean z^2, ~1 = within 1 std)", z)):
        print(f"\n{name}")
        for d, i in zip(*dist.topk(TOP, largest=False)):
            print(f"  row {i.item():5d}  {data.labels[i]:<20} {d.item():.4f}")