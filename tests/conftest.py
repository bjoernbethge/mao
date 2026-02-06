"""
Pytest fixtures for mcp_agents tests.
Production-ready, robust, DRY.
"""

import os
import pytest
import asyncio
import logging
import httpx
import socket
import time
import threading
import uvicorn
from dotenv import load_dotenv
from mao.storage import KnowledgeTree, ExperienceTree
from mao.mcp import MCPClient
from fastapi.testclient import TestClient
from mao.api.api import MCPAgentsAPI

load_dotenv()

try:
    from pytest_asyncio import fixture as asyncio_fixture
except ImportError:
    asyncio_fixture = pytest.fixture  # type: ignore


@pytest.fixture(scope="session")
def qdrant_url():
    """Returns the Qdrant URL from environment or default."""
    return os.environ.get("QDRANT_URL", "http://localhost:6333")


@asyncio_fixture(scope="function")
async def knowledge_tree(qdrant_url):
    """Returns a fresh KnowledgeTree for each test, with clean collection.
    Uses FastEmbedEmbeddings (local, no API key needed).
    """
    tree = await KnowledgeTree.create(
        url=qdrant_url,
        collection_name="test_knowledge_collection",
        recreate_on_dim_mismatch=True,
    )

    await tree.clear_all_points_async()
    yield tree
    await tree.clear_all_points_async()


@asyncio_fixture(scope="function")
async def experience_tree(qdrant_url):
    """Returns a fresh ExperienceTree for each test, with clean collection.
    Uses FastEmbedEmbeddings (local, no API key needed).
    """
    tree = await ExperienceTree.create(
        url=qdrant_url,
        collection_name="test_experience_collection",
        recreate_on_dim_mismatch=True,
    )

    await tree.clear_all_points_async()
    yield tree
    await tree.clear_all_points_async()


@asyncio_fixture(scope="function")
async def mcp_client():
    """Provides an MCPClient instance for testing with proper cleanup."""
    client = None
    try:
        client = MCPClient()
        logging.info(
            f"MCPClient for tests with {len(client.list_servers())} configured servers"
        )
        for server in client.list_servers():
            logging.info(f"  - Server: {server}")

        yield client
    finally:
        pass


@pytest.fixture(scope="function")
def api_test_client():
    """
    Provides a test client for the MCP Agents API using TestClient.

    Returns:
        tuple: (TestClient, MCPAgentsAPI) - The test client and API instance
    """
    test_api = MCPAgentsAPI(
        db_path=":memory:", title="Test MCP Agents API", version="test"
    )
    client = TestClient(test_api)
    return client, test_api


def find_free_port():
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class UvicornTestServer(uvicorn.Server):
    """Uvicorn test server helper for controlled test startup/shutdown."""

    def install_signal_handlers(self):
        """Override to avoid conflicts with pytest running in the same thread."""
        pass

    @property
    def is_running(self):
        """Check if server is running."""
        return self.started


@pytest.fixture(scope="session")
def live_api_server():
    """
    Provides a real running API server for tests.

    Returns:
        str: Base URL of the running server
    """
    api = MCPAgentsAPI(
        db_path="test_live_api.duckdb", title="Live Test MCP Agents API", version="test"
    )

    port = find_free_port()
    host = "127.0.0.1"

    config = uvicorn.Config(app=api, host=host, port=port, log_level="error")
    server = UvicornTestServer(config=config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    startup_timeout = 5.0
    start_time = time.time()
    while not server.is_running and time.time() - start_time < startup_timeout:
        time.sleep(0.1)

    if not server.is_running:
        raise RuntimeError("Failed to start API server within timeout period")

    base_url = f"http://{host}:{port}"

    client = httpx.Client(base_url=base_url, timeout=30.0)

    try:
        response = client.get("/health")
        response.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Failed to connect to API server: {e}")

    yield base_url

    server.should_exit = True
    thread.join(timeout=5)


def pytest_sessionfinish(session, exitstatus):
    """Hook to run at the very end of the test session for resource cleanup."""
    if os.name == "nt":
        logging.info("Test session ended. Adding cleanup delay...")
        try:
            asyncio.run(asyncio.sleep(1))
        except Exception as e:
            logging.warning(f"Could not run cleanup delay via asyncio: {e}")
            import time

            time.sleep(1)

    test_db_path = "test_live_api.duckdb"
    if os.path.exists(test_db_path):
        try:
            os.remove(test_db_path)
        except Exception as e:
            logging.warning(f"Could not remove test database: {e}")
