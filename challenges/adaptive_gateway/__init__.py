from flask import Blueprint


blueprint = Blueprint("adaptive_gateway", __name__)

from challenges.adaptive_gateway import routes  # noqa: E402, F401
