from flask import Blueprint


blueprint = Blueprint("kan_cheong_delivery_driver", __name__)

from . import routes  # noqa: E402, F401
