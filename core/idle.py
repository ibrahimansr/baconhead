"""How long since the human touched anything, which is what decides who is driving."""

from __future__ import annotations

from core.clock import now_ns


class HumanIdle:
    """The macOS idle timer can't answer this, because our own injected events reset it too."""

    def __init__(self):
        self._last_human_ns = now_ns()

    def observe(self, ev) -> None:
        if ev.source == "human":   # the injector tags its own events, so this is how the two are told apart
            self._last_human_ns = ev.t_ns

    def seconds(self) -> float:
        return (now_ns() - self._last_human_ns) / 1e9

    def touched(self) -> None:
        self._last_human_ns = now_ns()
