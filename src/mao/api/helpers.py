"""
Helper functions for API endpoints to reduce code duplication.
"""

import logging
from typing import Any

from ..agents import create_agent
from .db import ConfigDB


async def convert_db_tools_to_list(
    agent_tools: list[dict[str, Any]],
) -> list[dict[str, Any]] | None:
    """
    Convert database tool objects to a list of tool descriptions.

    Args:
        agent_tools: List of tool dictionaries from the database

    Returns:
        List of tool descriptions or None if conversion fails
    """
    if not agent_tools:
        return None

    try:
        return [
            {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {}),
            }
            for tool in agent_tools
        ]
    except Exception as e:
        logging.warning(f"Failed to convert tools: {e}")
        return None


async def create_and_start_agent(
    db: ConfigDB,
    agent_id: str,
    agent_config: dict[str, Any],
    active_agents: dict[str, dict[str, Any]],
) -> Any:
    """
    Create and start an agent, loading its tools from the database.

    Args:
        db: Database connection
        agent_id: ID of the agent
        agent_config: Agent configuration dictionary
        active_agents: Dictionary of active agents to update

    Returns:
        The created agent app
    """
    # Get tools for the agent
    agent_tools = await db.get_agent_tools(agent_id, enabled_only=True)
    tools = await convert_db_tools_to_list(agent_tools) if agent_tools else None

    # Create and start the agent
    agent_app = await create_agent(
        provider=agent_config["provider"],
        model_name=agent_config["model_name"],
        agent_name=agent_config["name"],
        system_prompt=agent_config.get("system_prompt"),
        use_react_agent=agent_config.get("use_react_agent", True),
        max_tokens_trimmed=agent_config.get("max_tokens_trimmed", 3000),
        llm_specific_kwargs=agent_config.get("llm_specific_kwargs"),
        tools=tools,
    )

    # Add agent to active agents
    active_agents[agent_id] = {"agent": agent_app, "config": agent_config}

    return agent_app
