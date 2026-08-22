from flask import Blueprint


blueprint = Blueprint("ghost_chains", __name__)

from challenges.ghost_chains import routes  # noqa: E402, F401
