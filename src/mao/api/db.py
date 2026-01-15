"""
Configuration database manager using DuckDB.
Provides a lightweight, file-based database for agent and MCP configuration.
"""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager, contextmanager
from typing import Any


import duckdb


class ConfigDB:
    """
    Configuration database for MCP agents using DuckDB.
    Stores and manages:
    - Agent configurations
    - Tool configurations
    - Server configurations
    - Team configurations
    - Supervisor configurations
    - Default system prompts
    """

    _instances: dict[str, "ConfigDB"] = {}
    _lock = asyncio.Lock()
    _lock = asyncio.Lock()

    @classmethod
    async def get_instance(cls, db_path: str = "mcp_config.duckdb") -> "ConfigDB":
        """
        Get or create a ConfigDB instance with the given path.
        Uses a singleton pattern to reuse database connections.

        Args:
            db_path: Path to the DuckDB database file

        Returns:
            ConfigDB instance
        """
        async with cls._lock:
            if db_path not in cls._instances:
                # Create new instance
                instance = cls(db_path)
                cls._instances[db_path] = instance
            return cls._instances[db_path]

    def __init__(self, db_path: str = "mcp_config.duckdb"):
        """
        Initialize the configuration database.

        Args:
            db_path: Path to the DuckDB database file
        """
        # Check for environment variable override
        env_db_path = os.environ.get("MCP_DB_PATH")
        if env_db_path:
            db_path = env_db_path

        self.db_path = db_path
        self._conn = None
        self._conn_lock = asyncio.Lock()
        self._initialize_db()

    def _connect(self):
        """Establish database connection"""
        if self._conn is None:
            self._conn = duckdb.connect(self.db_path)
        return self._conn

    @contextmanager
    def connection(self):
        """Context manager for database connections"""
        conn = self._connect()
        try:
            yield conn
        except Exception as e:
            logging.error(f"Database error: {e}")
            raise

    @asynccontextmanager
    async def async_connection(self):
        """Async context manager for database connections"""
        async with self._conn_lock:
            conn = self._connect()
            try:
                yield conn
            except Exception as e:
                logging.error(f"Database error: {e}")
                raise

    def _initialize_db(self):
        """Create required tables if they don't exist"""
        with self.connection() as conn:
            # Create the agents table
            conn.execute(
                """
            CREATE TABLE IF NOT EXISTS agents (
                id VARCHAR PRIMARY KEY,
                name VARCHAR NOT NULL,
                provider VARCHAR NOT NULL,
                model_name VARCHAR NOT NULL,
                system_prompt TEXT,
                use_react_agent BOOLEAN DEFAULT TRUE,
                max_tokens_trimmed INTEGER DEFAULT 3000,
                llm_specific_kwargs JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            )

            # Create the tools table
            conn.execute(
                """
            CREATE TABLE IF NOT EXISTS tools (
                id VARCHAR PRIMARY KEY,
                name VARCHAR NOT NULL,
                enabled BOOLEAN DEFAULT TRUE,
                server_id VARCHAR,
                description TEXT,
                parameters JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            )

            # Create the servers table
            conn.execute(
                """
            CREATE TABLE IF NOT EXISTS mcp_servers (
                id VARCHAR PRIMARY KEY,
                name VARCHAR NOT NULL UNIQUE,
                transport VARCHAR NOT NULL, 
                enabled BOOLEAN DEFAULT TRUE,
                url VARCHAR,
                command VARCHAR,
                args JSON,
                headers JSON,
                env_vars JSON,
                timeout INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            )

            # Create the agent-tool associations table
            conn.execute(
                """
            CREATE TABLE IF NOT EXISTS agent_tools (
                agent_id VARCHAR NOT NULL,
                tool_id VARCHAR NOT NULL,
                enabled BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (agent_id, tool_id)
            )
            """
            )

            # Create the supervisors table
            conn.execute(
                """
            CREATE TABLE IF NOT EXISTS supervisors (
                id VARCHAR PRIMARY KEY,
                agent_id VARCHAR NOT NULL,
                system_prompt TEXT,
                strategy VARCHAR NOT NULL DEFAULT 'team_manager',
                add_handoff_back_messages BOOLEAN DEFAULT TRUE,
                parallel_tool_calls BOOLEAN DEFAULT TRUE,
                config JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (agent_id) REFERENCES agents(id)
            )
            """
            )

            # Create the teams table
            conn.execute(
                """
            CREATE TABLE IF NOT EXISTS teams (
                id VARCHAR PRIMARY KEY,
                name VARCHAR NOT NULL,
                description TEXT,
                workflow_type VARCHAR NOT NULL DEFAULT 'sequential',
                supervisor_id VARCHAR,
                config JSON,
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (supervisor_id) REFERENCES supervisors(id)
            )
            """
            )

            # Create the team members table
            conn.execute(
                """
            CREATE TABLE IF NOT EXISTS team_members (
                team_id VARCHAR NOT NULL,
                agent_id VARCHAR NOT NULL,
                role VARCHAR NOT NULL,
                order_index INTEGER,
                is_active BOOLEAN DEFAULT TRUE,
                params JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (team_id, agent_id),
                FOREIGN KEY (team_id) REFERENCES teams(id),
                FOREIGN KEY (agent_id) REFERENCES agents(id)
            )
            """
            )

            # Create the configs table for global settings
            conn.execute(
                """
            CREATE TABLE IF NOT EXISTS global_configs (
                key VARCHAR PRIMARY KEY,
                value JSON,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            )

    def _process_result(
        self, result: tuple | Any, table_name: str
    ) -> dict[str, Any]:
        """Process a database result into a dictionary"""
        if result is None:
            return {}

        # Define column mappings for each table
        column_mappings = {
            "agents": [
                "id",
                "name",
                "provider",
                "model_name",
                "system_prompt",
                "use_react_agent",
                "max_tokens_trimmed",
                "llm_specific_kwargs",
                "created_at",
                "updated_at",
            ],
            "tools": [
                "id",
                "name",
                "enabled",
                "server_id",
                "description",
                "parameters",
                "created_at",
                "updated_at",
            ],
            "mcp_servers": [
                "id",
                "name",
                "transport",
                "enabled",
                "url",
                "command",
                "args",
                "headers",
                "env_vars",
                "timeout",
                "created_at",
                "updated_at",
            ],
            "agent_tools": ["agent_id", "tool_id", "enabled", "created_at"],
            "supervisors": [
                "id",
                "agent_id",
                "system_prompt",
                "strategy",
                "add_handoff_back_messages",
                "parallel_tool_calls",
                "config",
                "created_at",
                "updated_at",
            ],
            "teams": [
                "id",
                "name",
                "description",
                "workflow_type",
                "supervisor_id",
                "config",
                "is_active",
                "created_at",
                "updated_at",
            ],
            "team_members": [
                "team_id",
                "agent_id",
                "role",
                "order_index",
                "is_active",
                "params",
                "created_at",
                "updated_at",
            ],
            "global_configs": [
                "key",
                "value",
                "description",
                "created_at",
                "updated_at",
            ],
        }

        # Convert to dict if tuple
        if isinstance(result, tuple):
            # Get column names
            if hasattr(result, "_fields"):
                # Named tuple
                data = dict(zip(result._fields, result))
            else:
                # Use predefined column mappings
                if table_name in column_mappings:
                    columns = column_mappings[table_name]
                    data = dict(zip(columns, result))
                else:
                    # Fallback - just use indices
                    data = {f"col{i}": val for i, val in enumerate(result)}
        else:
            # Already a dict or dict-like
            data = dict(result)

        # Process JSON fields
        json_fields = [
            "llm_specific_kwargs",
            "config",
            "params",
            "args",
            "headers",
            "env_vars",
            "parameters",
        ]
        for field in json_fields:
            if field in data and data[field] is not None:
                if isinstance(data[field], str):
                    try:
                        data[field] = json.loads(data[field])
                    except json.JSONDecodeError:
                        # If not valid JSON, keep as string
                        pass

        return data

    def _build_update_query(
        self,
        table: str,
        id_column: str,
        id_value: str,
        json_fields: list[str],
        **kwargs: Any,
    ) -> tuple[str, list[Any]]:
        """
        Build an UPDATE query with SET clauses and parameters.

        Args:
            table: Table name
            id_column: Name of the ID column for the WHERE clause
            id_value: Value for the ID column
            json_fields: List of field names that should be JSON serialized
            **kwargs: Field names and values to update

        Returns:
            Tuple of (query string, parameters list)
        """
        set_clauses = []
        params: list[Any] = []

        for key, value in kwargs.items():
            if key in json_fields:
                set_clauses.append(f"{key} = ?")
                params.append(json.dumps(value) if value is not None else None)
            else:
                set_clauses.append(f"{key} = ?")
                params.append(value)

        set_clauses.append("updated_at = CURRENT_TIMESTAMP")
        params.append(id_value)

        query = f"UPDATE {table} SET {', '.join(set_clauses)} WHERE {id_column} = ?"
        return query, params

    # Agent Methods - Asynchronous Implementation
    async def create_agent(
        self,
        agent_id: str,
        name: str,
        provider: str,
        model_name: str,
        system_prompt: str | None = None,
        use_react_agent: bool = True,
        max_tokens_trimmed: int = 3000,
        llm_specific_kwargs: dict[str, Any] | None = None,
    ) -> str:
        """
        Create a new agent configuration asynchronously.

        Args:
            agent_id: Unique identifier for the agent
            name: Display name for the agent
            provider: LLM provider (openai, anthropic, etc.)
            model_name: Model name to use
            system_prompt: System prompt for the agent
            use_react_agent: Whether to use ReAct agent
            max_tokens_trimmed: Maximum tokens to keep in context
            llm_specific_kwargs: Provider-specific arguments

        Returns:
            The agent_id of the created agent
        """
        async with self.async_connection() as conn:
            kwargs_json = (
                json.dumps(llm_specific_kwargs) if llm_specific_kwargs else None
            )

            conn.execute(
                """
            INSERT INTO agents (id, name, provider, model_name, system_prompt, use_react_agent, max_tokens_trimmed, llm_specific_kwargs)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                [
                    agent_id,
                    name,
                    provider,
                    model_name,
                    system_prompt,
                    use_react_agent,
                    max_tokens_trimmed,
                    kwargs_json,
                ],
            )

            return agent_id

    async def get_agent(self, agent_id: str) -> dict[str, Any] | None:
        """Gets an agent by ID asynchronously"""
        async with self.async_connection() as conn:
            result = conn.execute(
                "SELECT * FROM agents WHERE id = ?", [agent_id]
            ).fetchone()

            if not result:
                return None

            agent = self._process_result(result, "agents")

            # No need to deserialize JSON fields again as _process_result already does this
            return agent

    async def list_agents(
        self, limit: int | None = None, offset: int | None = 0
    ) -> list[dict[str, Any]]:
        """Lists all agents asynchronously, with optional pagination"""
        async with self.async_connection() as conn:
            query = "SELECT * FROM agents"
            params: list[Any] = []

            if limit is not None:
                query += " LIMIT ?"
                params.append(limit)

                if offset is not None:
                    query += " OFFSET ?"
                    params.append(offset)

            results = conn.execute(query, params).fetchall()

            agents = [self._process_result(row, "agents") for row in results]

            # No need to deserialize JSON fields again as _process_result already does this
            return agents

    async def update_agent(self, agent_id: str, **kwargs: Any) -> bool:
        """Updates an agent asynchronously"""
        if not kwargs:
            return False

        async with self.async_connection() as conn:
            query, params = self._build_update_query(
                "agents", "id", agent_id, ["llm_specific_kwargs"], **kwargs
            )
            conn.execute(query, params)
            return True

    async def delete_agent(self, agent_id: str) -> bool:
        """Deletes an agent asynchronously"""
        async with self.async_connection() as conn:
            # First delete agent-tool associations
            conn.execute("DELETE FROM agent_tools WHERE agent_id = ?", [agent_id])

            # Delete agent from all teams
            conn.execute("DELETE FROM team_members WHERE agent_id = ?", [agent_id])

            # Get supervisors using this agent
            supervisors = conn.execute(
                "SELECT id FROM supervisors WHERE agent_id = ?", [agent_id]
            ).fetchall()
            supervisor_ids = [s[0] for s in supervisors]

            # Update teams using those supervisors to set supervisor_id to NULL
            for supervisor_id in supervisor_ids:
                conn.execute(
                    "UPDATE teams SET supervisor_id = NULL WHERE supervisor_id = ?",
                    [supervisor_id],
                )

            # Delete supervisors using this agent
            conn.execute("DELETE FROM supervisors WHERE agent_id = ?", [agent_id])

            # Then delete the agent
            conn.execute("DELETE FROM agents WHERE id = ?", [agent_id])

            return True

    async def get_agent_tools(
        self, agent_id: str, enabled_only: bool = False
    ) -> list[dict[str, Any]]:
        """Gets all tools assigned to an agent asynchronously"""
        async with self.async_connection() as conn:
            query = """
            SELECT t.*, at.enabled as agent_tool_enabled
            FROM agent_tools at
            JOIN tools t ON at.tool_id = t.id
            WHERE at.agent_id = ?
            """

            params = [agent_id]

            if enabled_only:
                query += " AND at.enabled = TRUE AND t.enabled = TRUE"

            query += " ORDER BY t.name"

            results = conn.execute(query, params).fetchall()
            return [self._process_result(r, "tools") for r in results]

    # Team Methods - Asynchronous Implementation
    async def get_team(self, team_id: str) -> dict[str, Any] | None:
        """Gets a team by its ID asynchronously"""
        async with self.async_connection() as conn:
            result = conn.execute(
                "SELECT * FROM teams WHERE id = ?", [team_id]
            ).fetchone()
            return self._process_result(result, "teams") if result else None

    async def list_teams(
        self, supervisor_id: str | None = None, active_only: bool = False
    ) -> list[dict[str, Any]]:
        """Lists all teams asynchronously, optionally filtered"""
        async with self.async_connection() as conn:
            query = "SELECT * FROM teams"
            params: list[Any] = []

            conditions = []
            if supervisor_id:
                conditions.append("supervisor_id = ?")
                params.append(supervisor_id)

            if active_only:
                conditions.append("is_active = TRUE")

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            query += " ORDER BY name"

            results = conn.execute(query, params).fetchall()
            return [self._process_result(r, "teams") for r in results]

    async def create_team(
        self,
        team_id: str,
        name: str,
        description: str | None = None,
        workflow_type: str = "sequential",
        supervisor_id: str | None = None,
        config: dict[str, Any] | None = None,
        is_active: bool = True,
    ) -> str:
        """Creates a team asynchronously"""
        async with self.async_connection() as conn:
            conn.execute(
                """
                INSERT INTO teams (
                    id, name, description, workflow_type, supervisor_id, config, is_active
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    team_id,
                    name,
                    description,
                    workflow_type,
                    supervisor_id,
                    json.dumps(config) if config else None,
                    is_active,
                ],
            )
            return team_id

    async def update_team(self, team_id: str, **kwargs: Any) -> bool:
        """Updates a team asynchronously"""
        if not kwargs:
            return False

        async with self.async_connection() as conn:
            query, params = self._build_update_query(
                "teams", "id", team_id, ["config"], **kwargs
            )
            conn.execute(query, params)
            return True

    async def delete_team(self, team_id: str) -> bool:
        """Deletes a team asynchronously"""
        async with self.async_connection() as conn:
            # Delete team members first (due to foreign key constraint)
            conn.execute("DELETE FROM team_members WHERE team_id = ?", [team_id])
            # Delete the team
            conn.execute("DELETE FROM teams WHERE id = ?", [team_id])
            return True

    async def add_team_member(
        self,
        team_id: str,
        agent_id: str,
        role: str,
        order_index: int | None = None,
        is_active: bool = True,
        params: dict[str, Any] | None = None,
    ) -> bool:
        """Adds an agent to a team asynchronously"""
        async with self.async_connection() as conn:
            # Check if the member already exists
            existing = conn.execute(
                "SELECT 1 FROM team_members WHERE team_id = ? AND agent_id = ?",
                [team_id, agent_id],
            ).fetchone()

            if existing:
                # Update existing member
                set_clauses = [
                    "role = ?",
                    "is_active = ?",
                    "updated_at = CURRENT_TIMESTAMP",
                ]
                params_list = [role, is_active]

                if order_index is not None:
                    set_clauses.append("order_index = ?")
                    params_list.append(order_index)

                if params is not None:
                    set_clauses.append("params = ?")
                    params_list.append(json.dumps(params))

                # Add team_id and agent_id to parameters
                params_list.extend([team_id, agent_id])

                conn.execute(
                    f"UPDATE team_members SET {', '.join(set_clauses)} WHERE team_id = ? AND agent_id = ?",
                    params_list,
                )
            else:
                # Insert new member
                conn.execute(
                    """
                    INSERT INTO team_members (
                        team_id, agent_id, role, order_index, is_active, params
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        team_id,
                        agent_id,
                        role,
                        order_index,
                        is_active,
                        json.dumps(params) if params else None,
                    ],
                )

            return True

    async def get_team_members(
        self, team_id: str, active_only: bool = False
    ) -> list[dict[str, Any]]:
        """Gets all members of a team asynchronously"""
        async with self.async_connection() as conn:
            query = """
            SELECT tm.*, a.name as agent_name
            FROM team_members tm
            LEFT JOIN agents a ON tm.agent_id = a.id
            WHERE tm.team_id = ?
            """

            params = [team_id]

            if active_only:
                query += " AND tm.is_active = TRUE"

            query += " ORDER BY tm.order_index, tm.created_at"

            results = conn.execute(query, params).fetchall()
            return [self._process_result(r, "team_members") for r in results]

    async def update_team_member(self, team_id: str, agent_id: str, **kwargs: Any) -> bool:
        """Updates a team member asynchronously"""
        if not kwargs:
            return False

        async with self.async_connection() as conn:
            # Build the SET clause and parameters
            set_clauses = []
            params: list[Any] = []

            for key, value in kwargs.items():
                if key == "params":
                    set_clauses.append(f"{key} = ?")
                    params.append(json.dumps(value) if value is not None else None)
                else:
                    set_clauses.append(f"{key} = ?")
                    params.append(value)

            # Add updated_at timestamp
            set_clauses.append("updated_at = CURRENT_TIMESTAMP")

            # Add team_id and agent_id to parameters
            params.extend([team_id, agent_id])

            # Execute the update
            conn.execute(
                f"UPDATE team_members SET {', '.join(set_clauses)} WHERE team_id = ? AND agent_id = ?",
                params,
            )

            return True

    async def remove_team_member(self, team_id: str, agent_id: str) -> bool:
        """Removes an agent from a team asynchronously"""
        async with self.async_connection() as conn:
            conn.execute(
                "DELETE FROM team_members WHERE team_id = ? AND agent_id = ?",
                [team_id, agent_id],
            )
            return True

    # Supervisor Methods - Asynchronous Implementation
    async def create_supervisor(
        self,
        supervisor_id: str,
        agent_id: str,
        system_prompt: str | None = None,
        strategy: str = "team_manager",
        add_handoff_back_messages: bool = True,
        parallel_tool_calls: bool = True,
        config: dict[str, Any] | None = None,
    ) -> str:
        """Creates a supervisor asynchronously"""
        async with self.async_connection() as conn:
            conn.execute(
                """
                INSERT INTO supervisors (
                    id, agent_id, system_prompt, strategy, 
                    add_handoff_back_messages, parallel_tool_calls, config
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    supervisor_id,
                    agent_id,
                    system_prompt,
                    strategy,
                    add_handoff_back_messages,
                    parallel_tool_calls,
                    json.dumps(config) if config else None,
                ],
            )
            return supervisor_id

    async def get_supervisor(self, supervisor_id: str) -> dict[str, Any] | None:
        """Gets a supervisor by its ID asynchronously"""
        async with self.async_connection() as conn:
            result = conn.execute(
                """
                SELECT s.*, a.name as agent_name 
                FROM supervisors s
                LEFT JOIN agents a ON s.agent_id = a.id
                WHERE s.id = ?
                """,
                [supervisor_id],
            ).fetchone()
            return self._process_result(result, "supervisors") if result else None

    async def list_supervisors(
        self, agent_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Lists all supervisors asynchronously, optionally filtered by agent_id"""
        async with self.async_connection() as conn:
            query = """
            SELECT s.*, a.name as agent_name 
            FROM supervisors s
            LEFT JOIN agents a ON s.agent_id = a.id
            """

            params = []

            if agent_id:
                query += " WHERE s.agent_id = ?"
                params.append(agent_id)

            query += " ORDER BY s.created_at DESC"

            results = conn.execute(query, params).fetchall()
            return [self._process_result(r, "supervisors") for r in results]

    async def update_supervisor(self, supervisor_id: str, **kwargs: Any) -> bool:
        """Updates a supervisor asynchronously"""
        if not kwargs:
            return False

        async with self.async_connection() as conn:
            query, params = self._build_update_query(
                "supervisors", "id", supervisor_id, ["config"], **kwargs
            )
            conn.execute(query, params)
            return True

    async def delete_supervisor(self, supervisor_id: str) -> bool:
        """Deletes a supervisor asynchronously"""
        async with self.async_connection() as conn:
            # Update teams that use this supervisor to remove the reference
            conn.execute(
                "UPDATE teams SET supervisor_id = NULL, updated_at = CURRENT_TIMESTAMP WHERE supervisor_id = ?",
                [supervisor_id],
            )

            # Delete the supervisor
            conn.execute("DELETE FROM supervisors WHERE id = ?", [supervisor_id])
            return True

    # Server Methods - Asynchronous Implementation
    async def create_server(
        self,
        server_id: str,
        name: str,
        transport: str,
        enabled: bool = True,
        url: str | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        headers: dict[str, str] | None = None,
        env_vars: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> str:
        """Creates a server asynchronously"""
        async with self.async_connection() as conn:
            conn.execute(
                """
                INSERT INTO mcp_servers (
                    id, name, transport, enabled, url, command, args, headers, env_vars, timeout
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    server_id,
                    name,
                    transport,
                    enabled,
                    url,
                    command,
                    json.dumps(args) if args else None,
                    json.dumps(headers) if headers else None,
                    json.dumps(env_vars) if env_vars else None,
                    timeout,
                ],
            )
            return server_id

    async def get_server(self, server_id: str) -> dict[str, Any] | None:
        """Gets a server by its ID asynchronously"""
        async with self.async_connection() as conn:
            result = conn.execute(
                "SELECT * FROM mcp_servers WHERE id = ?", [server_id]
            ).fetchone()
            return self._process_result(result, "mcp_servers") if result else None

    async def list_servers(self, enabled_only: bool = False) -> list[dict[str, Any]]:
        """Lists all servers asynchronously, optionally filtered by enabled status"""
        async with self.async_connection() as conn:
            query = "SELECT * FROM mcp_servers"
            params: list[Any] = []

            if enabled_only:
                query += " WHERE enabled = TRUE"

            query += " ORDER BY name"

            results = conn.execute(query, params).fetchall()
            return [self._process_result(r, "mcp_servers") for r in results]

    async def update_server(self, server_id: str, **kwargs: Any) -> bool:
        """Updates a server asynchronously"""
        if not kwargs:
            return False

        async with self.async_connection() as conn:
            query, params = self._build_update_query(
                "mcp_servers", "id", server_id, ["args", "headers", "env_vars"], **kwargs
            )
            conn.execute(query, params)
            return True

    async def delete_server(self, server_id: str) -> bool:
        """Deletes a server asynchronously"""
        async with self.async_connection() as conn:
            # Update tools that use this server to set server_id to NULL
            conn.execute(
                "UPDATE tools SET server_id = NULL, updated_at = CURRENT_TIMESTAMP WHERE server_id = ?",
                [server_id],
            )

            # Delete the server
            conn.execute("DELETE FROM mcp_servers WHERE id = ?", [server_id])
            return True

    # Tool Methods - Asynchronous Implementation
    async def create_tool(
        self,
        tool_id: str,
        name: str,
        enabled: bool = True,
        server_id: str | None = None,
        description: str | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> str:
        """Creates a tool asynchronously"""
        async with self.async_connection() as conn:
            conn.execute(
                """
                INSERT INTO tools (
                    id, name, enabled, server_id, description, parameters
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    tool_id,
                    name,
                    enabled,
                    server_id,
                    description,
                    json.dumps(parameters) if parameters else None,
                ],
            )
            return tool_id

    async def get_tool(self, tool_id: str) -> dict[str, Any] | None:
        """Gets a tool by its ID asynchronously"""
        async with self.async_connection() as conn:
            result = conn.execute(
                "SELECT * FROM tools WHERE id = ?", [tool_id]
            ).fetchone()
            return self._process_result(result, "tools") if result else None

    async def list_tools(
        self, server_id: str | None = None, enabled_only: bool = False
    ) -> list[dict[str, Any]]:
        """Lists all tools asynchronously, optionally filtered"""
        async with self.async_connection() as conn:
            query = "SELECT * FROM tools"
            params: list[Any] = []

            conditions = []
            if server_id:
                conditions.append("server_id = ?")
                params.append(server_id)

            if enabled_only:
                conditions.append("enabled = TRUE")

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            query += " ORDER BY name"

            results = conn.execute(query, params).fetchall()
            return [self._process_result(r, "tools") for r in results]

    async def update_tool(self, tool_id: str, **kwargs: Any) -> bool:
        """Updates a tool asynchronously"""
        if not kwargs:
            return False

        async with self.async_connection() as conn:
            query, params = self._build_update_query(
                "tools", "id", tool_id, ["parameters"], **kwargs
            )
            conn.execute(query, params)
            return True

    async def delete_tool(self, tool_id: str) -> bool:
        """Deletes a tool asynchronously"""
        async with self.async_connection() as conn:
            # Delete tool-agent associations first
            conn.execute("DELETE FROM agent_tools WHERE tool_id = ?", [tool_id])

            # Delete the tool
            conn.execute("DELETE FROM tools WHERE id = ?", [tool_id])
            return True

    async def assign_tool_to_agent(
        self, agent_id: str, tool_id: str, enabled: bool = True
    ) -> bool:
        """Assigns a tool to an agent asynchronously"""
        async with self.async_connection() as conn:
            # Check if the association already exists
            existing = conn.execute(
                "SELECT 1 FROM agent_tools WHERE agent_id = ? AND tool_id = ?",
                [agent_id, tool_id],
            ).fetchone()

            if existing:
                # Update existing association
                conn.execute(
                    "UPDATE agent_tools SET enabled = ? WHERE agent_id = ? AND tool_id = ?",
                    [enabled, agent_id, tool_id],
                )
            else:
                # Insert new association
                conn.execute(
                    "INSERT INTO agent_tools (agent_id, tool_id, enabled) VALUES (?, ?, ?)",
                    [agent_id, tool_id, enabled],
                )

            return True

    async def remove_tool_from_agent(self, agent_id: str, tool_id: str) -> bool:
        """Removes a tool from an agent asynchronously"""
        async with self.async_connection() as conn:
            conn.execute(
                "DELETE FROM agent_tools WHERE agent_id = ? AND tool_id = ?",
                [agent_id, tool_id],
            )
            return True

    # Synchrone Wrapper-Methoden
    def create_agent_sync(
        self,
        agent_id: str,
        name: str,
        provider: str,
        model_name: str,
        system_prompt: str | None = None,
        use_react_agent: bool = True,
        max_tokens_trimmed: int = 3000,
        llm_specific_kwargs: dict[str, Any] | None = None,
    ) -> str:
        """
        Create a new agent configuration.

        Args:
            agent_id: Unique identifier for the agent
            name: Display name for the agent
            provider: LLM provider (openai, anthropic, etc.)
            model_name: Model name to use
            system_prompt: System prompt for the agent
            use_react_agent: Whether to use ReAct agent
            max_tokens_trimmed: Maximum tokens to keep in context
            llm_specific_kwargs: Provider-specific arguments

        Returns:
            The agent_id of the created agent
        """
        with self.connection() as conn:
            kwargs_json = (
                json.dumps(llm_specific_kwargs) if llm_specific_kwargs else None
            )

            conn.execute(
                """
            INSERT INTO agents (id, name, provider, model_name, system_prompt, use_react_agent, max_tokens_trimmed, llm_specific_kwargs)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                [
                    agent_id,
                    name,
                    provider,
                    model_name,
                    system_prompt,
                    use_react_agent,
                    max_tokens_trimmed,
                    kwargs_json,
                ],
            )

            return agent_id

    async def close_async(self):
        """Async version of close"""
        async with self._conn_lock:
            if self._conn:
                self._conn.close()
                self._conn = None

    def close(self):
        """Close the database connection"""
        if self._conn:
            self._conn.close()
            self._conn = None

    @classmethod
    async def cleanup(cls):
        """Close all database connections"""
        async with cls._lock:
            for db_path, instance in cls._instances.items():
                await instance.close_async()
            cls._instances.clear()

    # --- Global Config Methods (Stubs für API-Kompatibilität und mypy) ---
    def set_config(
        self, key: str, value: Any, description: str | None = None
    ) -> None:
        """Set a global configuration value."""
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO global_configs (key, value, description, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value, description=excluded.description, updated_at=CURRENT_TIMESTAMP
                """,
                [key, json.dumps(value), description],
            )

    def get_config(self, key: str) -> Any | None:
        """Get a global configuration value by key."""
        with self.connection() as conn:
            result = conn.execute(
                "SELECT value FROM global_configs WHERE key = ?", [key]
            ).fetchone()
            if result:
                try:
                    return json.loads(result[0])
                except Exception:
                    return result[0]
            return None

    def delete_config(self, key: str) -> None:
        """Delete a global configuration value by key."""
        with self.connection() as conn:
            conn.execute("DELETE FROM global_configs WHERE key = ?", [key])

    def export_config(self, export_path: str | None = None) -> str:
        """Export all global configs to a JSON file."""
        with self.connection() as conn:
            results = conn.execute(
                "SELECT key, value, description FROM global_configs"
            ).fetchall()
            configs = [
                {
                    "key": row[0],
                    "value": json.loads(row[1]) if row[1] else None,
                    "description": row[2],
                }
                for row in results
            ]
            export_file = export_path or "mcp_config_export.json"
            with open(export_file, "w", encoding="utf-8") as f:
                json.dump(configs, f, indent=2)
            return export_file

    def import_config(self, import_path: str) -> bool:
        """Import global configs from a JSON file."""
        if not os.path.exists(import_path):
            return False
        with open(import_path, "r", encoding="utf-8") as f:
            configs = json.load(f)
        with self.connection():
            for entry in configs:
                self.set_config(entry["key"], entry["value"], entry.get("description"))
        return True
