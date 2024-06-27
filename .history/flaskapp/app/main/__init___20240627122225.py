from flask import Blueprint

bp = Blueprint('main', __name__)

# Import routes after defining the blueprint
from app.main import routes  # This line is crucial! 