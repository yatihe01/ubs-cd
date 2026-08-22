from flask import Blueprint


blueprint = Blueprint("showdown", __name__)

from challenges.showdown import phase_1  # noqa: E402, F401
