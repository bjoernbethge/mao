---
applyTo: "**/api/**/*.py"
description: "Guidelines for FastAPI endpoint development"
---

# API Endpoint Development Guidelines

When working with FastAPI endpoints in the `mao/api/` directory, follow these guidelines:

## Request/Response Models

1. **Always use Pydantic models** for request and response validation
2. **Define models in `models.py`** to keep endpoints clean and reusable
3. **Use type hints** for all fields with appropriate validators
4. **Include example values** in schema definitions using `Config.schema_extra`

Example:
```python
from pydantic import BaseModel, Field

class AgentRequest(BaseModel):
    name: str = Field(..., description="Agent name", min_length=1)
    model: str = Field(..., description="LLM model identifier")
    system_prompt: str | None = Field(None, description="Optional system prompt")
    
    class Config:
        schema_extra = {
            "example": {
                "name": "research_assistant",
                "model": "claude-3-opus-20240229",
                "system_prompt": "You are a research assistant."
            }
        }
```

## Endpoint Design

1. **Use appropriate HTTP methods**: GET (read), POST (create), PUT/PATCH (update), DELETE (delete)
2. **Return appropriate status codes**: 200 (OK), 201 (Created), 400 (Bad Request), 404 (Not Found), 500 (Internal Error)
3. **Use dependency injection** for shared resources (database connections, config)
4. **Add OpenAPI tags** to group related endpoints
5. **Include comprehensive docstrings** describing endpoint behavior

Example:
```python
@router.post("/agents", response_model=AgentResponse, status_code=201, tags=["agents"])
async def create_agent(
    request: AgentRequest,
    db: Database = Depends(get_db)
) -> AgentResponse:
    """
    Create a new agent with the specified configuration.
    
    Args:
        request: Agent creation request with name, model, and optional system prompt
        db: Database connection (injected)
        
    Returns:
        AgentResponse: Created agent details including generated ID
        
    Raises:
        HTTPException: 400 if validation fails, 500 if creation fails
    """
    # Implementation
```

## Error Handling

1. **Use HTTPException** for all API errors
2. **Include descriptive error messages** for debugging
3. **Log errors** using appropriate log levels
4. **Never expose sensitive information** in error messages

Example:
```python
from fastapi import HTTPException, status

try:
    result = await agent.execute(query)
except ValueError as e:
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Invalid query: {str(e)}"
    )
except Exception as e:
    logger.error(f"Agent execution failed: {e}")
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Agent execution failed"
    )
```

## Async Operations

1. **All endpoint handlers must be async** (`async def`)
2. **Use `await`** for all I/O operations (database, LLM calls, file operations)
3. **Handle timeouts** using `asyncio.wait_for()`
4. **Use background tasks** for long-running operations that don't need immediate results

## Testing

1. **Write tests using FastAPI's TestClient** or `httpx.AsyncClient`
2. **Mock external dependencies** (LLM calls, database operations)
3. **Test all status codes** (success and error cases)
4. **Test request validation** with valid and invalid inputs
5. **Use fixtures** from `tests/conftest.py` for common test setup

Example:
```python
import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_create_agent(async_client: AsyncClient):
    response = await async_client.post(
        "/agents",
        json={
            "name": "test_agent",
            "model": "claude-3-opus-20240229"
        }
    )
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "test_agent"
    assert "id" in data
```

## WebSocket Endpoints

For streaming responses:

1. **Use WebSocket** for real-time streaming
2. **Handle connection lifecycle** (connect, disconnect, errors)
3. **Implement heartbeat** for long-lived connections
4. **Stream responses incrementally** for better UX

Example:
```python
@router.websocket("/agents/{agent_id}/stream")
async def agent_stream(websocket: WebSocket, agent_id: str):
    await websocket.accept()
    try:
        while True:
            message = await websocket.receive_text()
            async for chunk in agent.stream(message):
                await websocket.send_text(chunk)
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for agent {agent_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close(code=1011, reason="Internal error")
```
