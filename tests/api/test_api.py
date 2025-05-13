"""
Tests for the MCP Agents API.
Test the basic API endpoints and functionality.
"""



def test_health_endpoint(api_test_client):
    """Test that the health endpoint returns status ok."""
    client, _ = api_test_client
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


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


def test_json_handling(api_test_client):
    """
    Test that the API correctly processes JSON data.
    This is a placeholder - replace with a real endpoint that accepts JSON.
    """
    client, _ = api_test_client
    
    # Example test data - update based on your actual API endpoints
    test_data = {
        "name": "test_agent",
        "type": "assistant",
        "config": {
            "model": "test_model",
            "settings": {"temperature": 0.7}
        }
    }
    
    # This should be replaced with a real endpoint path that accepts JSON
    # For demonstration purposes only
    response = client.post(
        "/agents", 
        json=test_data
    )
    
    # Update these assertions based on your actual API behavior
    assert response.status_code in (200, 201, 404, 422)
    
    # If endpoint exists and succeeds:
    if response.status_code in (200, 201):
        data = response.json()
        assert data is not None 