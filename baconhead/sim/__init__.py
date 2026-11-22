"""Ten Roblox-faithful training games sharing one humanoid, one camera, one renderer, one action space."""

from sim.adoptme import AdoptMe
from sim.brainrot import StealABrainrot
from sim.disaster import NaturalDisaster
from sim.doors import Doors
from sim.fisch import Fisch
from sim.forest import NinetyNineNights
from sim.garden import GrowAGarden
from sim.mm2 import MurderMystery2
from sim.obby import Obby
from sim.rivals import Rivals

ENVS = {"obby": Obby, "brainrot": StealABrainrot, "adoptme": AdoptMe, "disaster": NaturalDisaster, "garden": GrowAGarden,
        "forest": NinetyNineNights, "mm2": MurderMystery2, "doors": Doors, "rivals": Rivals, "fisch": Fisch}


def make(name: str, seed: int = 0, **kw):
    return ENVS[name](seed=seed, **kw)
