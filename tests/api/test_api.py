"""
Tests for the MCP Agents API.
Basic API endpoint functionality tests.
"""

import uuid
from pathlib import Path


def test_health_endpoint(api_test_client):
    """Test that the health endpoint returns status ok."""
    client, _ = api_test_client
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert response.headers["x-request-id"].startswith("req_")


def test_request_id_header_is_preserved(api_test_client):
    """Test that incoming request IDs are propagated back in the response."""
    client, _ = api_test_client
    response = client.get("/health", headers={"x-request-id": "req_test_123"})
    assert response.status_code == 200
    assert response.headers["x-request-id"] == "req_test_123"


def test_root_endpoint(api_test_client):
    """Test that the root endpoint returns API information."""
    client, _ = api_test_client
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "api" in data
    assert "version" in data
    assert "endpoints" in data
    assert isinstance(data["endpoints"], dict)


def test_config_skills_endpoint(api_test_client, monkeypatch):
    """Test discoverable skill listing from configured roots."""
    client, _ = api_test_client
    skill_root = Path.cwd() / ".test_tmp" / f"config_skills_{uuid.uuid4().hex}"
    skill_dir = skill_root / "analyst"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: analyst\ndescription: Analyze data\n---\n\nUse data workflows.\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MAO_SKILL_PATHS", str(skill_root))

    response = client.get("/config/skills")
    assert response.status_code == 200
    payload = response.json()
    assert str(skill_root) in payload["skill_roots"]
    assert any(skill["name"] == "analyst" for skill in payload["skills"])
