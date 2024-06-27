from flask import Flask
from app.config import Config
from app.main import bp as main_bp

def create_app(config_class=Config):
    """Creates the Flask application."""
    app = Flask(__name__)
    app.config.from_object(config_class)
    app.register_blueprint(main_bp, url_prefix='/api') # Register the main blueprint

    with app.app_context():
        from app import models

    # Register error handlers
    @app.errorhandler(404)
    def not_found(error):
        return {'error': 'Not found'}, 404

    @app.errorhandler(500)
    def internal_error(error):
        return {'error': 'Internal server error'}, 500
    return app 