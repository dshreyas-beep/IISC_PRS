from .base import SpeciesProfile
from .human import PROFILE as HUMAN
from .elephant import PROFILE as ELEPHANT
from .tiger import PROFILE as TIGER
from .leopard import PROFILE as LEOPARD
from .sloth_bear import PROFILE as SLOTH_BEAR

REGISTRY = {
    "human": HUMAN,
    "elephant": ELEPHANT,
    "tiger": TIGER,
    "leopard": LEOPARD,
    "sloth_bear": SLOTH_BEAR,
}

def list_species():
    return list(REGISTRY.keys())

def get_profile(kind: str):
    return REGISTRY[kind]
