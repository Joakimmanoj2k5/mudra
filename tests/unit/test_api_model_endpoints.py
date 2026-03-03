from backend.api import app


def test_health_endpoint_exists():
    routes = {r.path for r in app.routes}
    assert "/health" in routes
    assert "/ready" in routes
    assert "/models" in routes
    assert "/models/register" in routes
