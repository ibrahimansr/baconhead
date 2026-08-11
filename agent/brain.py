"""Claude: the goals, reward, and post-mortems that a world model has no way to invent from pixels alone."""

from __future__ import annotations

import base64
import io
import json
import os

import numpy as np

from wm.actions import NAMES

MODEL = os.environ.get("BACONHEAD_MODEL", "claude-opus-5")


def _img_block(frame: np.ndarray, max_edge: int = 512) -> dict:
    from PIL import Image

    img = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    if max(img.size) > max_edge:
        s = max_edge / max(img.size)
        img = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))))   # smaller image, fewer tokens, faster reply
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png",
                   "data": base64.standard_b64encode(buf.getvalue()).decode()},
    }


class Brain:
    """Every method returns a safe default on failure, because a dead API must never stop the world model playing."""

    def __init__(self, api_key: str | None = None, model: str = MODEL):
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])
        self.model = model
        self.calls = 0   # worth watching, since this is the only part of the loop that costs money

    def _ask(self, system: str, blocks: list, max_tokens: int = 400) -> dict:
        self.calls += 1
        r = self.client.messages.create(
            model=self.model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": blocks}],
        )
        text = "".join(b.text for b in r.content if b.type == "text").strip()
        return _parse_json(text)

    def set_goal(self, frame: np.ndarray, memory: str = "") -> dict:
        """Look at the screen and decide what we are trying to do and how to weigh planning."""
        try:
            out = self._ask(
                "You are playing an unknown Roblox game by looking at the screen. "
                "Set the immediate goal (one short phrase) and how to weigh planning.\n"
                f"Action vocabulary (by index): {list(enumerate(NAMES))}\n"
                "A learned world model will imagine each candidate plan and predict whether it "
                "survives, so propose the ambitious move — it will be vetoed if it kills us.\n"
                'Reply JSON only: {"goal": str, "reason": str, "suggested": [int, ...up to 8], '
                '"weights": {"alive": float, "value": float, "novelty": float}}\n'
                "weights: alive 0-10 (raise it when the screen looks lethal — gaps, lava, long "
                "drops), value 0-5, novelty 0-2 (raise it when stuck or exploring.)",
                [_img_block(frame), {"type": "text", "text": memory or "No history yet."}],
            )
            goal = str(out.get("goal") or "explore")
            sug = [int(a) % len(NAMES) for a in (out.get("suggested") or [])][:8]   # clamped so a bad index can't crash the planner
            w = {k: float(v) for k, v in (out.get("weights") or {}).items()
                 if k in ("alive", "value", "novelty")}
            return {"goal": goal, "weights": w, "suggested": sug, "reason": str(out.get("reason") or "")}
        except Exception as e:
            return {"goal": "explore forward", "weights": {}, "suggested": [], "reason": f"brain error: {e}"}

    def choose_plan(self, frame: np.ndarray, previews: list, plans: list, goal: str) -> int:
        """Read filmstrips of what the world model imagines for each plan, and pick one."""
        if not previews:
            return 0
        blocks: list = [{"type": "text", "text": f"NOW (goal: {goal}):"}, _img_block(frame)]
        for i, (prev, pl) in enumerate(zip(previews, plans)):
            blocks.append({"type": "text", "text":
                           f"PLAN {i}: {[NAMES[a] for a in pl.actions]} — world model says "
                           f"P(alive)={pl.p_alive:.2f}, progress={pl.value:.2f}. "
                           "Imagined future, left to right:"})
            if prev is not None:
                blocks.append(_img_block(prev, max_edge=768))
        try:
            out = self._ask(
                "Each PLAN shows what a learned world model imagines will happen, rendered by "
                "matching its imagined states to real frames it has seen before — so the images "
                "are approximate, and its numbers can be wrong on anything unfamiliar. "
                "Pick the plan that best serves the goal without dying.\n"
                'Reply JSON only: {"pick": int, "why": str}',
                blocks, max_tokens=300,
            )
            return int(out.get("pick", 0)) % len(previews)
        except Exception:
            return 0   # falling back to the world model's own top pick is always safe

    def score_progress(self, frame: np.ndarray, goal: str) -> float | None:
        """The reward signal, which wm/train.py later distils into the value head."""
        try:
            out = self._ask(
                "Rate progress toward the stated goal on this screen, 0-10. 0 = no closer or worse, "
                '10 = goal achieved. Reply JSON only: {"score": float, "why": str}',
                [_img_block(frame), {"type": "text", "text": f"Goal: {goal}"}], max_tokens=150,
            )
            return max(0.0, min(10.0, float(out["score"])))
        except Exception:
            return None

    def explain_death(self, frames: list, actions: list, goal: str) -> str:
        """Turn a death into one sentence the agent carries into its next plan."""
        try:
            blocks: list = [{"type": "text", "text":
                             f"Goal was: {goal}. Last actions: {[NAMES[a] for a in actions]}. "
                             "Frames leading to death, oldest first:"}]
            blocks += [_img_block(f) for f in frames[-4:]]   # the shared clock is what makes these the right four frames
            out = self._ask(
                "The agent just died. Say what killed it and the one rule to follow next time.\n"
                'Reply JSON only: {"cause": str, "rule": str}',
                blocks, max_tokens=250,
            )
            return f"{out.get('cause', 'died')} → {out.get('rule', 'be careful')}"
        except Exception:
            return "died (cause unknown)"


def _parse_json(text: str) -> dict:
    if text.startswith("```"):
        text = text.split("```")[1].removeprefix("json").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        i, j = text.find("{"), text.rfind("}")
        return json.loads(text[i: j + 1]) if 0 <= i < j else {}   # last resort: dig the object out of surrounding prose


def _demo() -> None:
    """Self-check on parsing and the fallbacks, which needs no API key by design."""
    assert _parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert _parse_json('sure!\n{"pick": 2}\ndone') == {"pick": 2}
    assert _parse_json("not json at all") == {}
    b = Brain.__new__(Brain)   # built without __init__ so there is no client and every call must degrade
    b.client, b.model, b.calls = None, "x", 0
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    assert b.set_goal(frame)["goal"] == "explore forward"
    assert b.choose_plan(frame, [], [], "g") == 0
    assert b.score_progress(frame, "g") is None
    assert "died" in b.explain_death([frame], [0], "g")
    print("agent.brain ok")


if __name__ == "__main__":
    _demo()
