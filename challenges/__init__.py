from flask import Blueprint

from challenges.adaptive_gateway import blueprint as adaptive_gateway


BLUEPRINTS: list[tuple[Blueprint, str]] = [
    (adaptive_gateway, "/adaptive-gateway"),
]
