"""The latent world model: it learns what the screen means, what happens next, and what is worth doing."""

from __future__ import annotations

import re

import torch
import torch.nn as nn
import torch.nn.functional as F

IMG = 64        # the model never sees full resolution, because dynamics is about layout not detail
Z = 128         # the whole screen compressed to 128 numbers, which is what makes imagining 33 futures cheap
GOAL_DIM = 32


def goal_vec(text: str) -> torch.Tensor:
    """Goal text to a fixed vector, so 'climb the tower' and 'buy the upgrade' get different value functions."""
    v = torch.zeros(GOAL_DIM)
    for tok in re.findall(r"[a-z0-9]+", (text or "").lower()):
        v[hash_token(tok) % GOAL_DIM] += 1.0   # a hashed bag of words, no tokenizer and no extra dependency
    return v / v.norm().clamp(min=1e-6)


def hash_token(tok: str) -> int:
    h = 0
    for ch in tok:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF   # rolled by hand because Python's hash() is randomized per process
    return h


def preprocess(rgb) -> torch.Tensor:
    """A captured frame to the (1, 3, 64, 64) float tensor the encoder expects."""
    import numpy as np
    from PIL import Image

    img = Image.fromarray(np.asarray(rgb, dtype="uint8")).resize((IMG, IMG), Image.Resampling.BILINEAR)
    x = torch.from_numpy(np.asarray(img, dtype="float32").copy())
    return (x / 255.0).permute(2, 0, 1).unsqueeze(0)   # PIL gives H,W,C but conv layers want C,H,W


class WorldModel(nn.Module):
    def __init__(self, n_actions: int):
        super().__init__()
        self.n_actions = n_actions
        c = 32
        self.encoder = nn.Sequential(
            nn.Conv2d(3, c, 4, 2, 1), nn.SiLU(),          # 64 to 32
            nn.Conv2d(c, c * 2, 4, 2, 1), nn.SiLU(),      # 32 to 16
            nn.Conv2d(c * 2, c * 4, 4, 2, 1), nn.SiLU(),  # 16 to 8
            nn.Conv2d(c * 4, c * 4, 4, 2, 1), nn.SiLU(),  # 8 to 4
            nn.Flatten(), nn.Linear(c * 4 * 16, Z),
        )
        self.dec_in = nn.Linear(Z, c * 4 * 16)
        self.decoder = nn.Sequential(   # only exists during training, to stop the latent collapsing to a constant
            nn.Unflatten(1, (c * 4, 4, 4)),
            nn.ConvTranspose2d(c * 4, c * 4, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(c * 4, c * 2, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(c * 2, c, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(c, 3, 4, 2, 1), nn.Sigmoid(),
        )
        self.act_emb = nn.Embedding(n_actions, 32)
        self.dyn = nn.GRUCell(32, Z)   # the imagination itself: current latent plus an action gives the next latent
        self.policy = nn.Sequential(nn.Linear(Z + GOAL_DIM, 256), nn.SiLU(), nn.Linear(256, n_actions))
        self.cont = nn.Sequential(nn.Linear(Z, 128), nn.SiLU(), nn.Linear(128, 1))
        self.value = nn.Sequential(nn.Linear(Z + GOAL_DIM, 128), nn.SiLU(), nn.Linear(128, 1))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def step(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Imagine one tick forward entirely in latent space, with no pixels and no screen capture."""
        return self.dyn(self.act_emb(a), z)

    def heads(self, z: torch.Tensor, g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Action logits from imitation, log P(alive) for death, and predicted progress toward the goal."""
        zg = torch.cat([z, g.expand(z.shape[0], -1)], dim=-1)   # the same goal vector is broadcast across the batch
        return self.policy(zg), F.logsigmoid(self.cont(z)).squeeze(-1), self.value(zg).squeeze(-1)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.dec_in(z))

    def loss(self, frames, actions, goal, alive=None, progress=None) -> tuple[torch.Tensor, dict]:
        """One batch of sequences: trains dynamics, the imitation prior, and both heads at once."""
        B, T = actions.shape
        flat = frames.reshape(B * T, *frames.shape[2:])   # the encoder is per-frame, so time is folded into the batch
        z = self.encode(flat)
        recon = F.mse_loss(self.decode(z), flat)
        zs = z.reshape(B, T, Z)

        pred = self.step(zs[:, :-1].reshape(-1, Z), actions[:, :-1].reshape(-1))
        dyn = F.mse_loss(pred, zs[:, 1:].reshape(-1, Z).detach())   # detached so the encoder can't cheat by going flat

        logits, log_alive, val = self.heads(zs.reshape(-1, Z), goal)
        bc = F.cross_entropy(logits, actions.reshape(-1))   # behaviour cloning: predict the key the human actually held

        parts = {"recon": recon.item(), "dyn": dyn.item(), "bc": bc.item()}
        total = recon + dyn + bc

        if alive is not None:
            m = ~torch.isnan(alive.reshape(-1))   # most ticks have no death label, so they are simply skipped
            if m.any():
                p = log_alive[m].exp().clamp(1e-6, 1 - 1e-6)
                cont_loss = F.binary_cross_entropy(p, alive.reshape(-1)[m])
                total = total + cont_loss
                parts["cont"] = cont_loss.item()
        if progress is not None:
            m = ~torch.isnan(progress.reshape(-1))
            if m.any():
                v_loss = F.mse_loss(val[m], progress.reshape(-1)[m])   # this is where Claude's judgement enters the weights
                total = total + v_loss
                parts["value"] = v_loss.item()

        parts["total"] = total.item()
        return total, parts


def load(path: str, n_actions: int, device: str = "cpu") -> WorldModel:
    m = WorldModel(n_actions)
    m.load_state_dict(torch.load(path, map_location=device))
    return m.to(device).eval()


def _demo() -> None:
    """Self-check on shapes, sparse-label masking, and a rollout longer than any real plan."""
    import numpy as np

    torch.manual_seed(0)
    m = WorldModel(12)

    x = preprocess(np.random.randint(0, 255, (240, 320, 3), dtype="uint8"))
    assert x.shape == (1, 3, IMG, IMG) and 0.0 <= x.min() <= x.max() <= 1.0

    B, T = 2, 5
    frames, acts = torch.rand(B, T, 3, IMG, IMG), torch.randint(0, 12, (B, T))
    alive = torch.full((B, T), float("nan")); alive[0, 1], alive[0, 2] = 1.0, 0.0
    prog = torch.full((B, T), float("nan")); prog[1, 3] = 0.7
    total, parts = m.loss(frames, acts, goal_vec("climb the tower"), alive, prog)
    total.backward()
    assert torch.isfinite(total) and set(parts) >= {"recon", "dyn", "bc", "cont", "value"}

    nan = torch.full((B, T), float("nan"))
    t2, p2 = m.loss(frames, acts, goal_vec(""), nan, nan)
    assert torch.isfinite(t2) and "cont" not in p2 and "value" not in p2   # all-unlabelled must not poison the loss

    g = goal_vec("climb the tower")
    assert torch.allclose(g, goal_vec("climb the tower")) and abs(g.norm() - 1) < 1e-5
    assert not torch.allclose(g, goal_vec("buy the upgrade")) and goal_vec("").norm() == 0

    z = m.encode(x)
    for _ in range(30):
        z = m.step(z, torch.tensor([1]))
    assert torch.isfinite(z).all()   # a GRU that drifts to infinity would make every long plan score as nonsense
    print("wm.model ok")


if __name__ == "__main__":
    _demo()
