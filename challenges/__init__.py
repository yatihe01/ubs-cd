from importlib import import_module

from flask import Blueprint

from challenges.adaptive_gateway import blueprint as adaptive_gateway


kan_cheong_delivery_driver = import_module(
    ".kan-cheong-delivery-driver", __name__
).blueprint


BLUEPRINTS: list[tuple[Blueprint, str]] = [
    (adaptive_gateway, "/adaptive-gateway"),
    (kan_cheong_delivery_driver, ""),
]
