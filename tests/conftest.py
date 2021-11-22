import pytest
from src.widget.app import create_app


@pytest.fixture
def app():
    app = create_app(dict(
        TESTING=True,
        SERVER_NAME='127.1'
    ))

    with app.app_context():
        yield app


@pytest.fixture
def client(app):
    """A test client for the app."""
    return app.test_client()


@pytest.fixture
def runner(app):
    """A test runner for the app's Click commands."""
    return app.test_cli_runner()
