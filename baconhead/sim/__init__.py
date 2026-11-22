from baconhead.sim.adoptme import AdoptMe
from baconhead.sim.brainrot import StealABrainrot
from baconhead.sim.disaster import NaturalDisaster
from baconhead.sim.doors import Doors
from baconhead.sim.fisch import Fisch
from baconhead.sim.forest import NinetyNineNights
from baconhead.sim.garden import GrowAGarden
from baconhead.sim.mm2 import MurderMystery2
from baconhead.sim.obby import Obby
from baconhead.sim.rivals import Rivals

ENVS = {"obby": Obby, "brainrot": StealABrainrot, "adoptme": AdoptMe, "disaster": NaturalDisaster, "garden": GrowAGarden,
        "forest": NinetyNineNights, "mm2": MurderMystery2, "doors": Doors, "rivals": Rivals, "fisch": Fisch}


def make(name, seed=0, **kw):
    return ENVS[name](seed=seed, **kw)
