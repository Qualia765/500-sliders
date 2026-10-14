import os

import torch
import torch.nn.functional as F

from data import Dataset
from model import Autoencoder

BATCH = 256
LEARNING_RATE = 1e-3
DEFAULT_LATENT = 50
EVAL_EVERY = 250  # steps
EVAL_RATES = (0.01, 0.025, 0.05, 0.10)  # keep rates the progress line reports
EVAL_SIZE = 2048  # samples per rate
MIN_GAIN = 0.005  # relative eval-loss drop that counts as real progress
PATIENCE = 10  # evals without real progress before suggesting to stop


class Trainer:
    def __init__(
        self,
        data: Dataset,
        keep: tuple[float, float],
        latent: int | None,
        weights_path: str,
        checkpoint_path: str,
        resume: bool,
    ) -> None:
        self.data = data
        self.keep = keep
        self.weights_path = weights_path
        self.checkpoint_path = checkpoint_path
        state = None
        if resume and os.path.exists(checkpoint_path):
            state = torch.load(checkpoint_path, map_location=data.device)
            saved = state["model"]["encoder.4.weight"].shape[0]
            if latent not in (None, saved):
                raise SystemExit(f"the checkpoint has a {saved}D latent; add --fresh to start over with {latent}")
            latent = saved
        self.model = Autoencoder(data.features, latent or DEFAULT_LATENT).to(data.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.step = 0
        self.reference = float("inf")  # eval loss at the last real improvement
        self.stale = 0  # evals since then
        if state:
            self.model.load_state_dict(state["model"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.step, self.reference = state["step"], state["reference"]
        # Fixed eval sets so the eval numbers only move when the model does.
        generator = torch.Generator(device=data.device)
        generator.manual_seed(0)
        self.eval_sets = [data.sample(EVAL_SIZE, (r, r), generator) for r in EVAL_RATES]
        baseline = [F.mse_loss(data.fill.expand_as(t), t).item() for _, t, _ in self.eval_sets]
        self.baseline = sum(baseline) / len(baseline)  # always outputting the average profile

    def run(self) -> None:
        print(
            f"device: {self.data.device} | latent {self.model.latent}D | starting at step {self.step} "
            f"| keep {self.keep[0]:.1%}-{self.keep[1]:.1%} | Ctrl+C to stop and save"
        )
        print(f"eval loss to beat: {self.baseline:.5f} (always outputting the average profile)")
        print("per keep rate: eval loss, then how often the true object is among the 5 closest to the output")
        try:
            self._loop()
        except KeyboardInterrupt:
            pass
        self._save()
        print(f"\nsaved {self.weights_path} and {self.checkpoint_path} (step {self.step})")

    def _loop(self) -> None:
        running = torch.zeros((), device=self.data.device)
        window = 0  # steps since the last report; a resumed run starts mid-window
        while True:
            x, target, _ = self.data.sample(BATCH, self.keep)
            loss = F.mse_loss(self.model(x), target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running += loss.detach()
            window += 1
            self.step += 1
            if self.step % EVAL_EVERY == 0:
                self._report(running.item() / window)
                running.zero_()
                window = 0

    @torch.no_grad()
    def _evaluate(self) -> tuple[list[float], list[float]]:
        losses, top5 = [], []
        for x, target, idx in self.eval_sets:
            out = self.model(x)
            losses.append(F.mse_loss(out, target).item())
            nearest = torch.cdist(out, self.data.means).topk(5, largest=False).indices
            top5.append((nearest == idx[:, None]).any(1).float().mean().item())
        return losses, top5

    def _report(self, train_loss: float) -> None:
        losses, top5 = self._evaluate()
        eval_loss = sum(losses) / len(losses)
        if eval_loss < self.reference * (1 - MIN_GAIN):
            self.reference, self.stale = eval_loss, 0
        else:
            self.stale += 1
        if self.stale >= PATIENCE:
            verdict = "still near the baseline, try a wider --keep" if eval_loss > 0.95 * self.baseline else "probably done"
            status = f"PLATEAU: no real gain in {self.stale * EVAL_EVERY} steps, {verdict}"
        elif self.stale == 0:
            status = "improving"
        else:
            status = f"slowing ({self.stale}/{PATIENCE})"
        rates = " | ".join(f"{r:.1%} {l:.4f} {t:4.0%}" for r, l, t in zip(EVAL_RATES, losses, top5))
        print(f"step {self.step:7d} | train {train_loss:.5f} | eval {eval_loss:.5f} | {rates} | {status}")
        self._save()

    def _save(self) -> None:
        self.model.save_npz(self.weights_path)
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "step": self.step,
                "reference": self.reference,
            },
            self.checkpoint_path,
        )