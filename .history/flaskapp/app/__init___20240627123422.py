from flask import Flask
from app.config import Config
from app.main import bp as main_bp

def create_app(config_class=Config):
    """Creates the Flask application."""
    app = Flask(__name__)
    app.config.from_object(config_class)
    app.register_blueprint(main_bp, url_prefix='/api') # Register the main blueprint



    return app 