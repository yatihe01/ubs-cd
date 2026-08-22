from flask import Blueprint


blueprint = Blueprint("stonks", __name__)

from . import routes  # noqa: E402, F401
