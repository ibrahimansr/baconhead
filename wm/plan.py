"""Planning by imagination: sample plans from the imitation prior, roll them forward, score them Claude's way."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from wm.actions import N_ACTIONS
from wm.model import Z, goal_vec

DEFAULT_W = {"alive": 3.0, "value": 1.0, "novelty": 0.3, "prior": 0.1}   # Claude overrides these per goal


@dataclass
class Plan:
    actions: list[int]
    score: float
    p_alive: float          # the world model's guess at surviving the whole rollout
    value: float
    latents: torch.Tensor   # the imagined trajectory, kept so it can be previewed as images


@dataclass
class LatentBank:
    """Real latents paired with real thumbnails, which lets us show an imagined future without a video decoder."""
    latents: torch.Tensor = field(default_factory=lambda: torch.zeros(0, Z))
    thumbs: list = field(default_factory=list)
    cap: int = 2000

    def add(self, z: torch.Tensor, thumb: np.ndarray) -> None:
        self.latents = torch.cat([self.latents, z.detach().reshape(1, Z)])[-self.cap:]   # oldest entries fall off
        self.thumbs.append(thumb)
        self.thumbs = self.thumbs[-self.cap:]

    def nearest(self, z: torch.Tensor) -> np.ndarray | None:
        if not self.thumbs:
            return None
        d = ((self.latents - z.reshape(1, Z)) ** 2).sum(-1)
        return self.thumbs[int(d.argmin())]   # the closest thing we have actually seen to this imagined state

    def strip(self, latents: torch.Tensor, every: int = 2) -> np.ndarray | None:
        """An imagined trajectory as one wide filmstrip, which is how a language model reads a latent space."""
        frames = [self.nearest(z) for z in latents[::every]]
        frames = [f for f in frames if f is not None]
        return np.concatenate(frames, axis=1) if frames else None


@torch.no_grad()
def plan(
    model,
    z0: torch.Tensor,
    goal: str,
    horizon: int = 8,
    k: int = 32,
    weights: dict | None = None,
    suggested: list[int] | None = None,
    temperature: float = 1.0,
    gamma: float = 0.95,
) -> list[Plan]:
    """Imagine k futures plus Claude's suggestion, and return them best first."""
    w = {**DEFAULT_W, **(weights or {})}
    g = goal_vec(goal)
    z = z0.reshape(1, Z).expand(k, Z).contiguous()   # every candidate starts from the same real frame

    seqs = torch.zeros(k, horizon, dtype=torch.long)
    if suggested:
        s = (suggested * horizon)[:horizon]   # Claude's idea becomes candidate 0 and has to beat the prior on score
        seqs[0] = torch.tensor(s, dtype=torch.long)

    score = torch.zeros(k)
    logp_alive = torch.zeros(k)
    prior_lp = torch.zeros(k)
    traj = []
    last_value = torch.zeros(k)

    for t in range(horizon):
        logits, la, val = model.heads(z, g)
        lp = torch.log_softmax(logits / max(temperature, 1e-3), dim=-1)
        a = torch.multinomial(lp.exp(), 1).squeeze(-1)   # sampling from the human prior keeps rollouts in-distribution
        if suggested:
            a[0] = seqs[0, t]
        seqs[:, t] = a
        prior_lp += lp.gather(-1, a.unsqueeze(-1)).squeeze(-1)
        logp_alive += la   # summing log P(alive) is the same as multiplying survival odds across the whole plan
        novelty = (z - z0.reshape(1, Z)).norm(dim=-1)   # distance from where we started, which is how it explores
        score += (gamma ** t) * (w["alive"] * la + w["value"] * val + w["novelty"] * novelty.tanh())
        last_value = val
        z = model.step(z, a)
        traj.append(z.clone())

    score += w["prior"] * prior_lp   # a small nudge toward plans a human would plausibly have played
    stacked = torch.stack(traj, dim=1)
    order = torch.argsort(score, descending=True)
    return [
        Plan(
            actions=[int(x) for x in seqs[i]],
            score=float(score[i]),
            p_alive=float(logp_alive[i].exp()),
            value=float(last_value[i]),
            latents=stacked[i],
        )
        for i in order
    ]


def surprise(model, z_pred: torch.Tensor, z_real: torch.Tensor) -> float:
    """How wrong the last imagined step was, which is the cue that it is time to wake Claude."""
    return float((z_pred.reshape(-1) - z_real.reshape(-1)).norm() / (z_real.norm() + 1e-6))


def _demo() -> None:
    """Self-check that the planner ranks correctly and that previews come out as images."""
    from wm.model import WorldModel

    torch.manual_seed(0)
    m = WorldModel(N_ACTIONS).eval()
    z0 = torch.randn(1, Z)
    plans = plan(m, z0, "reach the top", horizon=6, k=16)
    assert len(plans) == 16 and len(plans[0].actions) == 6
    assert all(0 <= a < N_ACTIONS for p in plans for a in p.actions)
    assert plans[0].score >= plans[-1].score

    hi = max(plans, key=lambda p: p.p_alive)
    only_alive = plan(m, z0, "reach the top", horizon=6, k=16,
                      weights={"alive": 100.0, "value": 0.0, "novelty": 0.0, "prior": 0.0})
    assert only_alive[0].p_alive >= hi.p_alive * 0.9   # weighting survival alone must surface the safest plan

    bank = LatentBank()
    for i in range(5):
        bank.add(torch.randn(1, Z), np.full((8, 8, 3), i * 40, dtype=np.uint8))
    strip = bank.strip(plans[0].latents)
    assert strip is not None and strip.shape[0] == 8 and strip.ndim == 3
    assert surprise(m, z0, z0) < 1e-5   # predicting the state you are already in is zero surprise by definition
    print("wm.plan ok")


if __name__ == "__main__":
    _demo()
