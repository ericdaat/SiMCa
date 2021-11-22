from flask import Flask

from src.widget import home


def create_app(config=None):
    """ Flask app factory that creates and configure the app.
    Args:
        test_config (str): python configuration filepath
    Returns: Flask application
    """
    app = Flask(__name__)
    app.config.from_pyfile("config.py")

    if config:
        app.config.update(config)

    # blueprints
    app.register_blueprint(home.bp)

    return app
