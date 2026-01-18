---
applyTo: "**/tests/**/*.py"
description: "Guidelines for pytest test development with async support"
---

# Test Development Guidelines

When writing tests in the `tests/` directory, follow these pytest best practices:

## Test Structure

1. **Use descriptive test names** that explain what is being tested
2. **Follow AAA pattern**: Arrange, Act, Assert
3. **One assertion per test** when possible (or logically grouped assertions)
4. **Use fixtures** from `conftest.py` for common test setup
5. **Mark async tests** with `@pytest.mark.asyncio`

Example:
```python
import pytest
from mao.agents import create_agent

@pytest.mark.asyncio
async def test_agent_creation_with_valid_config():
    # Arrange
    config = {
        "name": "test_agent",
        "model": "claude-3-opus-20240229"
    }
    
    # Act
    agent = await create_agent(**config)
    
    # Assert
    assert agent.name == "test_agent"
    assert agent.model == "claude-3-opus-20240229"
```

## Async Testing

1. **All async tests must have `@pytest.mark.asyncio` decorator**
2. **Use `async def` for test functions** that test async code
3. **Use `await`** for all async operations
4. **Leverage async fixtures** for shared async resources

Example:
```python
@pytest.fixture
async def initialized_agent():
    """Fixture that provides an initialized agent instance."""
    agent = await create_agent(name="fixture_agent")
    yield agent
    await agent.cleanup()

@pytest.mark.asyncio
async def test_agent_query(initialized_agent):
    response = await initialized_agent.query("test question")
    assert response is not None
```

## Mocking

1. **Mock external dependencies** (LLM API calls, database operations)
2. **Use `unittest.mock`** or `pytest-mock` for mocking
3. **Mock at the appropriate level** (prefer mocking HTTP clients over internal functions)
4. **Make tests deterministic** by mocking non-deterministic operations

Example:
```python
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_agent_with_mocked_llm():
    # Mock the LLM call to return a fixed response
    with patch('mao.agents.ChatAnthropic') as mock_llm:
        mock_instance = AsyncMock()
        mock_instance.ainvoke.return_value = "Mocked response"
        mock_llm.return_value = mock_instance
        
        agent = await create_agent(name="test")
        response = await agent.query("test")
        
        assert response == "Mocked response"
        mock_instance.ainvoke.assert_called_once()
```

## API Testing

1. **Use `httpx.AsyncClient`** for testing FastAPI endpoints
2. **Create client fixtures** in `conftest.py`
3. **Test all HTTP status codes** (success and error paths)
4. **Test request validation** with both valid and invalid data

Example:
```python
import pytest
from httpx import AsyncClient
from fastapi import status

@pytest.mark.asyncio
async def test_create_agent_success(async_client: AsyncClient):
    response = await async_client.post(
        "/agents",
        json={"name": "test", "model": "claude-3-opus"}
    )
    assert response.status_code == status.HTTP_201_CREATED
    assert response.json()["name"] == "test"

@pytest.mark.asyncio
async def test_create_agent_invalid_data(async_client: AsyncClient):
    response = await async_client.post(
        "/agents",
        json={"name": ""}  # Invalid: empty name
    )
    assert response.status_code == status.HTTP_400_BAD_REQUEST
```

## Fixtures

1. **Define fixtures in `conftest.py`** for reusability across test modules
2. **Use appropriate fixture scopes**: `function` (default), `class`, `module`, `session`
3. **Leverage pytest's dependency injection** for fixtures
4. **Clean up resources** in fixture teardown (use `yield`)

Example:
```python
# In conftest.py
@pytest.fixture
async def qdrant_client():
    """Fixture providing a Qdrant client."""
    from qdrant_client import AsyncQdrantClient
    
    client = AsyncQdrantClient(url="localhost:6333")
    yield client
    await client.close()

@pytest.fixture
async def test_collection(qdrant_client):
    """Fixture providing a test collection."""
    collection_name = "test_collection"
    await qdrant_client.create_collection(collection_name, ...)
    yield collection_name
    await qdrant_client.delete_collection(collection_name)
```

## Test Organization

1. **Mirror source structure** in test directory (`tests/api/` for `src/mao/api/`)
2. **Group related tests** in test classes when appropriate
3. **Use parametrize** for testing multiple inputs

Example:
```python
@pytest.mark.parametrize("model_name,expected_provider", [
    ("claude-3-opus-20240229", "anthropic"),
    ("gpt-4", "openai"),
    ("llama2", "ollama"),
])
def test_model_provider_detection(model_name, expected_provider):
    provider = detect_provider(model_name)
    assert provider == expected_provider
```

## Coverage

1. **Aim for high test coverage** but prioritize critical paths
2. **Run coverage reports**: `uv run pytest --cov=mao`
3. **Focus on edge cases** and error handling
4. **Don't test framework code** (e.g., FastAPI routing)

## Integration Tests

1. **Mark integration tests** with custom marker: `@pytest.mark.integration`
2. **Separate from unit tests** for faster CI feedback
3. **Use real dependencies** (actual Qdrant, not mocked) when appropriate
4. **Clean up test data** after integration tests

Example:
```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_agent_workflow():
    # Test with real dependencies
    agent = await create_agent(name="integration_test")
    storage = await create_vector_storage()
    
    try:
        # Test full workflow
        await agent.store_memory("Test memory", storage)
        results = await agent.query_memory("Test", storage)
        assert len(results) > 0
    finally:
        # Cleanup
        await storage.delete_collection()
```

## Test Data

1. **Use factories or builders** for test data creation
2. **Keep test data minimal** but representative
3. **Avoid hard-coding** test data in multiple places
4. **Use constants** for frequently used test values

Example:
```python
# In conftest.py or test_helpers.py
class AgentConfigBuilder:
    def __init__(self):
        self.config = {
            "name": "default_test_agent",
            "model": "claude-3-opus-20240229"
        }
    
    def with_name(self, name: str):
        self.config["name"] = name
        return self
    
    def build(self):
        return self.config

# In tests
def test_something():
    config = AgentConfigBuilder().with_name("custom").build()
    # Use config
```
