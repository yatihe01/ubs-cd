from flask import Blueprint

from challenges.adaptive_gateway import blueprint as adaptive_gateway
from challenges.showdown import blueprint as showdown


BLUEPRINTS: list[tuple[Blueprint, str]] = [
    (adaptive_gateway, "/adaptive-gateway"),
    (showdown, "/showdown"),
]
