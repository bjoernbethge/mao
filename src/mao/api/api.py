"""
MCP Agents API Application.
Provides a REST API for managing and interacting with MCP agents.
"""

import logging
import os
from contextlib import suppress
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .db import ConfigDB
from ..observability import next_request_id, reset_request_context, set_request_context

# Global state for active agents
active_agents: dict[str, dict[str, Any]] = {}


# Global dependency functions
async def get_config_db() -> AsyncGenerator[ConfigDB, None]:
    """
    Dependency for accessing the configuration database.
    Will be automatically closed after the request is processed.
    """
    db_path_env = os.environ.get("MCP_DB_PATH")
    db_path = db_path_env if db_path_env is not None else "mcp_config.duckdb"
    db = await ConfigDB.get_instance(db_path=db_path)
    try:
        yield db
    except Exception as e:
        logging.error(f"Database error in dependency: {e}")
        raise


def get_active_agents():
    """Dependency for accessing the active agents registry"""
    return active_agents


class MCPAgentsAPI(FastAPI):
    """
    FastAPI extension for MCP Agents API with integrated dependencies
    and router management.
    """

    def __init__(
        self,
        title: str = "MCP Agents API",
        description: str = "API for managing and interacting with MCP agents",
        version: str = "1.0.0",
        db_path: str | None = None,
        *args: Any,
        **kwargs: Any,
    ):
        """
        Initialize the API application with configurations and routers.

        Args:
            title: API title for documentation
            description: API description for documentation
            version: API version
            db_path: Path to the DuckDB database file
            *args, **kwargs: Additional arguments for FastAPI
        """
        super().__init__(
            title=title, description=description, version=version, *args, **kwargs
        )

        self.db_path = db_path or os.environ.get("MCP_DB_PATH", "mcp_config.duckdb")

        # Add middleware
        self._setup_middleware()

        # Add exception handlers
        self._setup_exception_handlers()

        # Add routers
        self._setup_routers()

        # Add base endpoints
        self._add_base_endpoints()

    def _setup_middleware(self):
        """Setup middleware for the API"""
        # CORS middleware
        cors_origins = os.environ.get("MAO_CORS_ORIGINS", "*").split(",")
        self.add_middleware(
            CORSMiddleware,
            allow_origins=[o.strip() for o in cors_origins],
            allow_credentials=cors_origins != ["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Request/response logging middleware
        @self.middleware("http")
        async def log_requests(request: Request, call_next):
            """Log request and response details"""
            request_id = request.headers.get("x-request-id") or next_request_id()
            token = set_request_context(
                request_id=request_id,
                http_method=request.method,
                http_path=request.url.path,
            )
            # Log request
            logging.debug(f"Request: {request.method} {request.url.path}")
            try:
                response = await call_next(request)
            finally:
                with suppress(Exception):
                    reset_request_context(token)

            response.headers["x-request-id"] = request_id
            logging.debug(f"Response: {response.status_code}")
            return response

    def _setup_exception_handlers(self):
        """Setup exception handlers for the API"""

        @self.exception_handler(RequestValidationError)
        async def validation_exception_handler(
            request: Request, exc: RequestValidationError
        ):
            """Handle validation errors"""
            errors = []
            for error in exc.errors():
                errors.append(
                    {"loc": error["loc"], "msg": error["msg"], "type": error["type"]}
                )

            return JSONResponse(
                status_code=422,
                content={"detail": "Validation error", "errors": errors},
            )

        @self.exception_handler(Exception)
        async def general_exception_handler(request: Request, exc: Exception):
            """Handle general exceptions"""
            logging.exception(f"Unhandled exception: {exc}")
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error", "error": str(exc)},
            )

    async def get_config_db(self) -> AsyncGenerator[ConfigDB, None]:
        """
        Dependency for accessing the configuration database.
        Will be automatically closed after the request is processed.
        """
        db_path_env = os.environ.get("MCP_DB_PATH")
        db_path = db_path_env if db_path_env is not None else "mcp_config.duckdb"
        db = await ConfigDB.get_instance(db_path=db_path)
        try:
            yield db
        except Exception as e:
            logging.error(f"Database error in dependency: {e}")
            raise

    def get_active_agents(self) -> dict[str, dict[str, Any]]:
        """Dependency for accessing the active agents registry"""
        return active_agents

    def _setup_routers(self):
        """Register all routers from submodules"""
        # Import here to avoid circular references
        from .agents import router as agents_router
        from .mcp import router as mcp_router
        from .storage import config_router, export_router
        from .teams import router as teams_router

        self.include_router(agents_router)
        self.include_router(mcp_router)
        self.include_router(teams_router)
        self.include_router(config_router)
        self.include_router(export_router)

    def _add_base_endpoints(self):
        """Add basic information and health check endpoints"""

        @self.get("/health", tags=["health"])
        async def health_check():
            """Simple health check endpoint"""
            return {"status": "ok", "version": self.version}

        @self.get("/", tags=["root"])
        async def get_api_info():
            """Returns basic API information and available endpoints"""
            return {
                "api": self.title,
                "version": self.version,
                "endpoints": {
                    "agents": "/agents - Agent management",
                    "teams": "/teams - Team management",
                    "supervisors": "/teams/supervisors - Supervisor management",
                    "mcp": "/mcp - MCP server and tool management",
                    "config": "/config - Global configuration and skill discovery",
                    "import/export": "/export, /import - Configuration import/export",
                },
                "documentation": "/docs - Swagger UI documentation",
                "redoc": "/redoc - ReDoc documentation",
            }

    async def shutdown(self):
        """Shutdown the API and clean up resources"""
        # Close all database connections
        await ConfigDB.cleanup()


# Global instance for compatibility with old code
api = MCPAgentsAPI()

# Add main block to run the application with uvicorn when executed directly
if __name__ == "__main__":
    import uvicorn
    import sys

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Get port from environment or use default
    port = int(os.environ.get("PORT", 8000))

    # Start uvicorn server
    logging.info(f"Starting MAO API server on port {port}")
    uvicorn.run(
        "src.mao.api.api:api", host="0.0.0.0", port=port, reload=True, log_level="info"
    )
