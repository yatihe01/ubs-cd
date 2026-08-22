from importlib import import_module

from flask import Blueprint

from challenges.adaptive_gateway import blueprint as adaptive_gateway
from challenges.ghost_chains import blueprint as ghost_chains
from challenges.showdown import blueprint as showdown


kan_cheong_delivery_driver = import_module(
    ".kan-cheong-delivery-driver", __name__
).blueprint


BLUEPRINTS: list[tuple[Blueprint, str]] = [
    (adaptive_gateway, "/adaptive-gateway"),
    (ghost_chains, "/ghost-chains"),
    (showdown, "/showdown"),
    (kan_cheong_delivery_driver, ""),
]
