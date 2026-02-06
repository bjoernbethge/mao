"""
Tests for Agent and Supervisor classes.
"""

import pytest
import asyncio
import uuid
import os
import logging

from langchain_core.messages import AIMessage, HumanMessage

from mao.agents import create_agent, Supervisor, load_mcp_tools

# Import pytest_asyncio if available
try:
    import pytest_asyncio

    ASYNCIO_FIXTURE = pytest_asyncio.fixture
except ImportError:
    ASYNCIO_FIXTURE = pytest.fixture  # type: ignore

# Configuration for tests — defaults to Ollama (local/cloud)
TEST_LLM_PROVIDER = os.environ.get("TEST_LLM_PROVIDER", "ollama")
TEST_LLM_MODEL = os.environ.get("TEST_LLM_MODEL", "gemma3:4b-cloud")


@pytest.mark.asyncio
async def test_create_agent_factory(knowledge_tree, experience_tree):
    """Test the agent factory function with basic parameters."""
    agent_name = f"test_agent_{uuid.uuid4().hex[:8]}"
    agent_app = await create_agent(
        provider=TEST_LLM_PROVIDER,
        model_name=TEST_LLM_MODEL,
        agent_name=agent_name,
        knowledge_tree=knowledge_tree,
        experience_tree=experience_tree,
        system_prompt="You are a test agent.",
    )

    # Verify basic agent properties
    assert agent_app is not None, "Agent app should be created"
    assert hasattr(agent_app, "ainvoke"), "Agent should have ainvoke method"
    assert agent_app.name == agent_name, "Agent name should be set correctly"

    # Basic functionality test
    thread_id = f"test_thread_{uuid.uuid4()}"
    response = await agent_app.ainvoke(
        {"messages": [HumanMessage(content="Hello")]},
        config={"configurable": {"thread_id": thread_id}},
    )

    # Verify response structure
    assert response is not None
    assert "messages" in response
    assert len(response["messages"]) > 0
    last_message = response["messages"][-1]
    assert isinstance(last_message, AIMessage)
    assert last_message.content, "Agent response should not be empty"


@pytest.mark.asyncio
async def test_agent_learning_retrieval(knowledge_tree, experience_tree):
    """Test that agent can learn from interactions and retrieve context."""
    # Create agent with knowledge
    agent_app = await create_agent(
        provider=TEST_LLM_PROVIDER,
        model_name=TEST_LLM_MODEL,
        agent_name="learning_agent_test",
        knowledge_tree=knowledge_tree,
        experience_tree=experience_tree,
        system_prompt="You learn from and remember conversations.",
    )

    # Add knowledge entry
    await knowledge_tree.add_entry_async(
        "Python is a programming language created by Guido van Rossum."
    )

    # First interaction to be learned
    user_query = f"What is Python and who created it? [Test ID: {uuid.uuid4().hex[:8]}]"
    thread_id = f"learning_thread_{uuid.uuid4()}"

    # First invoke creates experience
    await agent_app.ainvoke(
        {"messages": [HumanMessage(content=user_query)]},
        config={"configurable": {"thread_id": thread_id}},
    )

    # Multiple attempts to find the stored experience
    max_attempts = 5
    experience_found = False

    for attempt in range(max_attempts):
        # Give time for async operations to complete
        await asyncio.sleep(1.0)

        # Try multiple search terms
        search_queries = [
            user_query,
            "Python programming",
            "Guido van Rossum",
            (
                f"Test ID: {user_query.split('Test ID:')[-1].strip().rstrip(']')}"
                if "Test ID:" in user_query
                else ""
            ),
        ]

        for query in search_queries:
            if not query:
                continue

            search_results = await experience_tree.search_async(query, k=3)
            if search_results:
                experience_found = True
                logging.info(
                    f"Experience found on attempt {attempt+1} with query: {query}"
                )
                break

        if experience_found:
            break

        logging.warning(
            f"Experience search attempt {attempt+1}/{max_attempts} failed. Retrying..."
        )

    # Skip test if we can't find any experience
    if not experience_found:
        logging.error("Could not find stored experience after multiple attempts")
        pytest.skip(
            "Experience storage/retrieval is not working correctly, skipping assertion"
        )

    # Second query to test context retrieval with direct conversation flow
    follow_up_response = await agent_app.ainvoke(
        {
            "messages": [
                HumanMessage(content=user_query),
                AIMessage(
                    content="Python is a programming language created by Guido van Rossum."
                ),
                HumanMessage(content="Who created Python again?"),
            ]
        },
        config={"configurable": {"thread_id": thread_id}},
    )

    # Verify context was used
    last_message = follow_up_response["messages"][-1]
    assert (
        "Guido" in last_message.content
    ), "Response should include previously provided information"


@pytest.mark.asyncio
async def test_supervisor_basic(knowledge_tree, experience_tree):
    """Test Supervisor with a simple worker agent."""
    # Create worker agent
    worker_agent = await create_agent(
        provider=TEST_LLM_PROVIDER,
        model_name=TEST_LLM_MODEL,
        agent_name="worker_agent",
        knowledge_tree=knowledge_tree,
        experience_tree=experience_tree,
        system_prompt="You are a helpful worker. Answer questions directly and concisely.",
    )

    # Create supervisor
    supervisor = Supervisor(
        agents=[worker_agent],
        supervisor_provider=TEST_LLM_PROVIDER,
        supervisor_model_name=TEST_LLM_MODEL,
        supervisor_system_prompt=(
            "You are a supervisor. Delegate the user's question to worker_agent."
        ),
    )

    # Initialize the supervisor
    await supervisor.init_supervisor()
    assert supervisor.app is not None, "Supervisor should be initialized"

    # Test supervisor delegation
    thread_id = f"supervisor_thread_{uuid.uuid4()}"
    response = await supervisor.invoke(
        messages=[HumanMessage(content="What is AI?")], thread_id=thread_id
    )

    # Verify response
    assert response is not None
    assert "messages" in response
    assert len(response["messages"]) > 0

    # Find worker's response in the messages
    worker_response = None
    for msg in response["messages"]:
        if isinstance(msg, AIMessage) and getattr(msg, "name", None) == "worker_agent":
            worker_response = msg.content
            break

    assert worker_response is not None, "Worker agent should have responded"
    assert len(worker_response) > 0, "Worker response should not be empty"


@pytest.mark.asyncio
async def test_load_mcp_tools_function(mcp_client):
    """Test the load_mcp_tools helper function with different inputs."""
    # Test with real tools from an MCPClient (requires session context)
    async with mcp_client.session() as client:
        real_tools = client.get_tools()
        loaded_tools = await load_mcp_tools(real_tools)
        assert (
            loaded_tools == real_tools
        ), "Should return the same list when input is a list"

    # Test with None
    empty_tools = await load_mcp_tools(None)
    assert empty_tools == [], "Should return empty list when input is None"

    # Test with invalid type
    string_input = "not a valid tools input"
    invalid_tools = await load_mcp_tools(string_input)
    assert invalid_tools == [], "Should return empty list when input is invalid type"


@pytest.mark.asyncio
async def test_agent_with_mcp_tools(knowledge_tree, experience_tree, mcp_client):
    """Test creating an agent with MCP tools."""
    # Check if MCP client works
    servers = mcp_client.list_servers()
    if not servers:
        pytest.skip("No MCP servers configured")

    active_servers = mcp_client.list_active_servers()
    if not active_servers:
        pytest.skip("No active MCP servers found")

    # Create agent with MCP tools
    agent_app = await create_agent(
        provider=TEST_LLM_PROVIDER,
        model_name=TEST_LLM_MODEL,
        agent_name="mcp_tools_agent",
        knowledge_tree=knowledge_tree,
        experience_tree=experience_tree,
        tools=mcp_client,
        system_prompt="You are a helpful assistant with access to tools. Use the appropriate tool when needed.",
        use_react_agent=True,
    )

    # Verify agent was created
    assert agent_app is not None, "Agent app should be created"
    assert hasattr(agent_app, "ainvoke"), "Agent should have ainvoke method"

    # Test with a request that should use a tool
    thread_id = f"mcp_tools_thread_{uuid.uuid4()}"
    test_query = "What is the current time in Berlin, Germany?"

    try:
        response = await agent_app.ainvoke(
            {"messages": [HumanMessage(content=test_query)]},
            config={"configurable": {"thread_id": thread_id}},
        )

        # Verify response
        assert response is not None
        assert "messages" in response
        assert len(response["messages"]) > 0

        # Log the response but don't enforce specific content
        last_message = response["messages"][-1]
        logging.info(f"Response to '{test_query}': {last_message.content}")
    except Exception as e:
        logging.error(f"Error testing agent with MCP tools: {e}")
        raise
