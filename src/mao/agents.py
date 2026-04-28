"""Agent classes with Agent, Supervisor, and create_agent factory."""

import logging
import os
import json
import uuid
import asyncio
import time
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field, create_model

from langchain.agents import create_agent as lc_create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.agents.middleware.types import AgentMiddleware, AgentState, ModelRequest, ModelResponse, OmitFromOutput
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, trim_messages
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool, tool

from langgraph.runtime import Runtime
from typing_extensions import Annotated, NotRequired

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from mao.mcp import MCPClient
from mao.observability import build_trace_metadata, langsmith_trace, merge_trace_tags
from mao.skills import SkillRegistry
from mao.checkpoint import get_checkpointer
from mao.storage import ExperienceTree, KnowledgeTree

load_dotenv()

logger = logging.getLogger(__name__)


class ParallelDelegationTask(BaseModel):
    agent_name: str = Field(description="Tool/agent name to delegate to")
    query: str = Field(description="Task or question for that agent")


def _normalize_terms(values: list[str] | None) -> list[str]:
    if not values:
        return []
    return [value.strip().lower() for value in values if isinstance(value, str) and value.strip()]


def evaluate_member_policy(member: dict[str, Any], query: str) -> tuple[bool, str | None]:
    params = member.get("params") or {}
    if not params.get("allow_supervisor_delegation", True):
        return False, "supervisor delegation disabled"

    lowered_query = query.lower()
    blocked = _normalize_terms(params.get("blocked_keywords"))
    for term in blocked:
        if term in lowered_query:
            return False, f"blocked keyword '{term}'"

    required = _normalize_terms(params.get("routing_keywords"))
    if required and not any(term in lowered_query for term in required):
        return False, "query does not match routing keywords"
    return True, None


def _approximate_token_count(messages: list[BaseMessage]) -> int:
    total_chars = 0
    for message in messages:
        total_chars += len(_ensure_str(message.content))
    return max(1, total_chars // 4)


class RuntimeAgentState(AgentState[Any]):
    response_schema: NotRequired[Annotated[dict[str, Any] | None, OmitFromOutput]]


class RetrievalLearningMiddleware(AgentMiddleware[RuntimeAgentState, None, Any]):
    state_schema = RuntimeAgentState

    def __init__(self, agent: "Agent") -> None:
        super().__init__()
        self.agent = agent

    async def awrap_model_call(
        self,
        request: ModelRequest[None],
        handler,
    ) -> ModelResponse[Any]:
        state = request.state
        messages = list(state.get("messages", []))
        user_input = ""
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                user_input = _ensure_str(message.content)
                break

        context_str = _ensure_str(await self.agent._retrieve_context(user_input))
        system_content = self.agent._build_system_prompt()
        if context_str:
            system_content = f"{system_content}\n{context_str}"

        try:
            trimmed_messages = trim_messages(
                messages,
                max_tokens=3000,
                strategy="last",
                token_counter=_approximate_token_count,
                include_system=True,
                start_on="human",
            )
        except Exception:
            logger.warning("Message trimming failed, using last 20 messages")
            trimmed_messages = messages[-20:]

        response_schema = state.get("response_schema")
        return await handler(
            request.override(
                messages=trimmed_messages,
                system_message=SystemMessage(content=system_content),
                response_format=response_schema or request.response_format,
            )
        )

    async def aafter_agent(self, state: RuntimeAgentState, runtime: Runtime[None]) -> None:
        user_input = ""
        for message in reversed(state.get("messages", [])):
            if isinstance(message, HumanMessage):
                user_input = _ensure_str(message.content)
                break

        learn_content: Any = ""
        if "structured_response" in state:
            learn_content = json.dumps(state["structured_response"], default=str)
        else:
            for message in reversed(state.get("messages", [])):
                if isinstance(message, AIMessage):
                    learn_content = _ensure_str(message.content)
                    break

        if user_input and learn_content:
            await self.agent._learn_experience(user_input, learn_content)


def _parse_hitl_tools() -> dict[str, dict[str, Any]]:
    raw = os.environ.get("MAO_HITL_TOOLS", "").strip()
    if not raw:
        return {}
    result: dict[str, dict[str, Any]] = {}
    for tool_name in [item.strip() for item in raw.split(",") if item.strip()]:
        result[tool_name] = {"allowed_decisions": ["approve", "edit", "reject"]}
    return result


def _build_invoke_config(
    *,
    thread_id: str,
    run_name: str,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> RunnableConfig:
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    config["tags"] = merge_trace_tags(tags)
    config["metadata"] = build_trace_metadata(thread_id=thread_id, metadata=metadata)
    config["run_name"] = run_name
    return config


def _create_model(
    provider: str,
    model_name: str,
    temperature: float = 0.0,
    stream: bool = False,
) -> BaseChatModel:
    kwargs: dict[str, Any] = {"temperature": temperature, "streaming": stream}
    if provider.lower() == "ollama":
        ollama_host = os.environ.get("OLLAMA_HOST")
        if ollama_host:
            kwargs["base_url"] = ollama_host
    return init_chat_model(model_name, model_provider=provider, **kwargs)


async def load_mcp_tools(
    mcp_client: MCPClient | list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if mcp_client is None:
        return []

    if isinstance(mcp_client, MCPClient):
        logger.debug("Loading tools from MCPClient...")
        try:
            tools = await mcp_client.get_tools()
            logger.debug("Loaded %d tools from MCP servers", len(tools))
            return tools
        except Exception as e:
            logger.error("Error loading MCP tools: %s", e)
            return []
    elif isinstance(mcp_client, list):
        return mcp_client
    else:
        logger.warning("Unexpected mcp_client type: %s", type(mcp_client))
        return []


async def _process_llm_response(response: Any) -> tuple[BaseMessage, str]:
    if isinstance(response, AIMessage):
        return response, str(response.content)
    content = str(response)
    return AIMessage(content=content), content


def _ensure_str(val: Any) -> str:
    if isinstance(val, str):
        return val
    if isinstance(val, list):
        return "\n".join(str(v) for v in val)
    return str(val)


def _dict_to_tool(d: dict[str, Any]) -> BaseTool:
    from pydantic import Field, create_model

    params = d.get("parameters", {})
    properties = params.get("properties", {})
    required = params.get("required", [])

    fields = {}
    for prop_name, prop_schema in properties.items():
        field_type = str
        prop_type = prop_schema.get("type", "string")
        if prop_type == "integer":
            field_type = int
        elif prop_type == "number":
            field_type = float
        elif prop_type == "boolean":
            field_type = bool
        elif prop_type == "array":
            field_type = list

        default = ... if prop_name in required else None
        desc = prop_schema.get("description", "")
        fields[prop_name] = (field_type, Field(default=default, description=desc))

    if not fields:
        fields["input"] = (str, Field(description="Input for the tool"))

    input_model = create_model(f"{d['name']}_Input", **fields)

    return StructuredTool.from_function(
        func=lambda **kwargs: f"Tool '{d['name']}' called with: {kwargs}",
        name=d["name"],
        description=d.get("description", ""),
        args_schema=input_model,
    )


def _dicts_to_tools(tools: list[Any]) -> list[Any]:
    result = []
    for t in tools:
        if callable(t) or isinstance(t, BaseTool):
            result.append(t)
        elif isinstance(t, dict) and "name" in t:
            try:
                result.append(_dict_to_tool(t))
            except Exception as e:
                logger.warning("Failed to convert dict tool '%s': %s", t.get("name"), e)
    return result


class Agent:
    def __init__(
        self,
        llm_instance: BaseChatModel,
        agent_name: str,
        tools: MCPClient | list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
        skills: list[str] | None = None,
        skill_paths: list[str] | None = None,
        stream: bool = False,
    ):
        self.llm = llm_instance
        self.name = agent_name
        self.configured_tools = tools
        self.loaded_tools: list[Any] = []
        self.system_prompt = system_prompt or "You are a helpful assistant."
        self.enabled_skills = skills or []
        self.skill_paths = skill_paths or []
        self.knowledge_tree: KnowledgeTree | None = None
        self.experience_tree: ExperienceTree | None = None
        self.skill_registry: SkillRegistry | None = None
        self.memory = get_checkpointer()
        self.stream = stream
        self.agent_runnable = None

    async def _load_mcp_tools(self) -> list[dict[str, Any]]:
        return await load_mcp_tools(self.configured_tools)

    async def _retrieve_context(self, query: str, k: int = 3) -> str:
        context_parts = []
        if self.knowledge_tree:
            knowledge_hits = await self.knowledge_tree.search_async(query, k=k)
            if knowledge_hits:
                context_parts.append(
                    "\nRelevant Knowledge:\n"
                    + "\n".join([h["page_content"] for h in knowledge_hits])
                )

        if self.experience_tree:
            experience_hits = await self.experience_tree.search_async(query, k=k)
            if experience_hits:
                context_parts.append(
                    "\nRelevant Experience:\n"
                    + "\n".join([h["page_content"] for h in experience_hits])
                )

        return "".join(context_parts).strip()

    async def _learn_experience(
        self, user_input: str, model_output: Any, tags: list[str] | None = None
    ) -> None:
        if not self.experience_tree:
            return

        model_output_str = _ensure_str(model_output)

        knowledge_id = None
        if self.knowledge_tree and user_input:
            knowledge_hits = await self.knowledge_tree.search_async(user_input, k=1)
            if knowledge_hits:
                knowledge_id = knowledge_hits[0].get("id")

        exp_text = f"User: {user_input}\nAgent: {model_output_str}"

        await self.experience_tree.learn_from_experience_async(
            exp_text, related_knowledge_id=knowledge_id, tags=tags
        )

    def _build_rag_tools(self) -> list[BaseTool]:
        tools: list[BaseTool] = []
        if self.knowledge_tree:
            kt = self.knowledge_tree

            @tool
            async def retrieve_knowledge(query: str) -> str:
                """Search the agent's knowledge base for relevant information.

                Args:
                    query: Search query to find relevant knowledge
                """
                hits = await kt.search_async(query, k=3)
                if not hits:
                    return "No relevant knowledge found."
                return "\n".join(h["page_content"] for h in hits)

            tools.append(retrieve_knowledge)

        if self.experience_tree:
            et = self.experience_tree
            kt = self.knowledge_tree

            @tool
            async def retrieve_experience(query: str) -> str:
                """Search the agent's past experiences for relevant context.

                Args:
                    query: Search query to find relevant past experiences
                """
                hits = await et.search_async(query, k=3)
                if not hits:
                    return "No relevant experience found."
                return "\n".join(h["page_content"] for h in hits)

            tools.append(retrieve_experience)

            @tool
            async def save_experience(summary: str) -> str:
                """Save a summary of what was learned or accomplished for future reference.

                Args:
                    summary: Brief summary of the interaction outcome or lesson learned
                """
                knowledge_id = None
                if kt:
                    hits = await kt.search_async(summary, k=1)
                    if hits:
                        knowledge_id = hits[0].get("id")
                await et.learn_from_experience_async(
                    summary, related_knowledge_id=knowledge_id
                )
                return "Experience saved."

            tools.append(save_experience)

        return tools

    def _build_skill_tools(self) -> list[BaseTool]:
        if not self.skill_registry:
            return []

        registry = self.skill_registry
        tools: list[BaseTool] = []

        @tool
        def list_skills() -> str:
            """List the discoverable skills available to this agent."""
            skills = registry.list_skills()
            if not skills:
                return "No discoverable skills found."
            return "\n".join(
                f"- {skill.name}: {skill.description} ({skill.path})"
                for skill in skills
            )

        tools.append(list_skills)

        @tool
        def load_skill(skill_name: str) -> str:
            """Load a skill's full SKILL.md contents when you need exact workflow details."""
            raw = registry.read_skill(skill_name)
            if not raw:
                return f"Skill '{skill_name}' not found."
            return raw

        tools.append(load_skill)

        if self.experience_tree:
            et = self.experience_tree

            @tool
            async def recommend_skills(task: str) -> str:
                """Recommend relevant skills for a task using skill metadata and prior skill insights."""
                matches = await registry.recommend_skills(task, experience_tree=et)
                if not matches:
                    return "No relevant skills found."
                lines = []
                for match in matches:
                    reasons = ", ".join(match["reasons"]) if match["reasons"] else "metadata match"
                    lines.append(
                        f"- {match['name']}: {match['description']} [{reasons}]"
                    )
                return "\n".join(lines)

            tools.append(recommend_skills)

            @tool
            async def save_skill_insight(skill_name: str, insight: str) -> str:
                """Persist a reusable lesson, caveat, or workflow note for a skill."""
                saved = await registry.save_skill_insight(
                    skill_name=skill_name,
                    insight=insight,
                    experience_tree=et,
                )
                if not saved:
                    return f"Skill '{skill_name}' not found or no experience store available."
                return "Skill insight saved."

            tools.append(save_skill_insight)

        return tools

    def _build_system_prompt(self) -> str:
        sections = [self.system_prompt.strip()]
        if self.skill_registry:
            guidance = self.skill_registry.render_guidance()
            if guidance:
                sections.append(guidance)
            selected_skills = self.skill_registry.render_selected_skills(self.enabled_skills)
            if selected_skills:
                sections.append(
                    "## Enabled Skills\n"
                    "The following local skills are preloaded. Follow them when relevant.\n\n"
                    f"{selected_skills}"
                )
        return "\n\n".join(section for section in sections if section).strip()

    async def init_agent(self):
        try:
            safe_name = self.name.replace("-", "_").replace(" ", "_")
            self.knowledge_tree = await KnowledgeTree.create(
                collection_name=f"knowledge_{safe_name}"
            )
            self.experience_tree = await ExperienceTree.create(
                collection_name=f"experience_{safe_name}"
            )
            self.skill_registry = SkillRegistry(skill_paths=self.skill_paths)

            self.loaded_tools = await self._load_mcp_tools()
            rag_tools = self._build_rag_tools()
            skill_tools = self._build_skill_tools()
            self.loaded_tools.extend(rag_tools)
            self.loaded_tools.extend(skill_tools)
            logger.info(
                "Agent '%s' loaded %d tools (%d RAG, %d skill tools)",
                self.name,
                len(self.loaded_tools),
                len(rag_tools),
                len(skill_tools),
            )
            all_tools = _dicts_to_tools(self.loaded_tools)
            middleware: list[Any] = [RetrievalLearningMiddleware(self)]
            interrupt_on = _parse_hitl_tools()
            if interrupt_on:
                middleware.append(HumanInTheLoopMiddleware(interrupt_on=interrupt_on))

            self.agent_runnable = lc_create_agent(
                model=self.llm,
                tools=all_tools,
                system_prompt=self._build_system_prompt(),
                middleware=middleware,
                checkpointer=self.memory,
                name=self.name,
                state_schema=RuntimeAgentState,
            )
            logger.info("Agent '%s' compiled with %d tools.", self.name, len(all_tools))
            return self.agent_runnable
        except Exception as e:
            logger.error("Failed to initialize agent '%s': %s", self.name, e, exc_info=True)
            raise

    def get_compiled_app(self):
        if not self.agent_runnable:
            raise RuntimeError("Agent not initialized. Call init_agent() first.")
        return self.agent_runnable


class Supervisor:
    def __init__(
        self,
        agents: list[Any],
        supervisor_provider: str,
        supervisor_model_name: str,
        supervisor_system_prompt: str,
        supervisor_tools: MCPClient | list[dict[str, Any]] | None = None,
        add_handoff_back_messages: bool = True,
        parallel_tool_calls: bool = True,
        **supervisor_kwargs: Any,
    ):
        self.agents = agents
        self.supervisor_provider = supervisor_provider
        self.supervisor_model_name = supervisor_model_name
        self.llm: BaseChatModel | None = None
        self.prompt = supervisor_system_prompt
        self.supervisor_tools = supervisor_tools
        self.supervisor_kwargs = {
            "add_handoff_back_messages": add_handoff_back_messages,
            "parallel_tool_calls": parallel_tool_calls,
            **supervisor_kwargs,
        }
        self.parallel_tool_calls = parallel_tool_calls
        self.memory = get_checkpointer()
        self.app = None
        self.supervisor_agent = None
        self.metrics: dict[str, Any] = {
            "delegations_total": 0,
            "delegation_failures": 0,
            "parallel_delegations_total": 0,
            "parallel_tasks_total": 0,
            "per_agent": {},
            "recent_delegations": [],
        }

    async def _load_supervisor_mcp_tools(self) -> list[dict[str, Any]]:
        return await load_mcp_tools(self.supervisor_tools)

    def _build_agent_tools(self) -> list[BaseTool]:
        tools: list[BaseTool] = []
        agent_tool_map: dict[str, Any] = {}

        async def invoke_worker(
            *,
            worker: Any,
            worker_name: str,
            worker_agent_id: str | None,
            worker_role: str | None,
            query: str,
            config: RunnableConfig | None,
            delegated_via: str,
        ) -> str:
            parent_thread_id = None
            if config:
                parent_thread_id = (
                    config.get("configurable", {}) or {}
                ).get("thread_id")
            child_thread_id = (
                f"{parent_thread_id}:{worker_name}"
                if parent_thread_id
                else f"{worker_name}:{uuid.uuid4().hex}"
            )
            started_at = time.perf_counter()
            self.metrics["delegations_total"] += 1
            per_agent = self.metrics["per_agent"].setdefault(
                worker_name,
                {
                    "agent_id": worker_agent_id,
                    "delegations": 0,
                    "failures": 0,
                    "last_latency_ms": None,
                },
            )
            per_agent["delegations"] += 1
            try:
                response = await worker.ainvoke(
                    {"messages": [{"role": "user", "content": query}]},
                    config={"configurable": {"thread_id": child_thread_id}},
                )
                if isinstance(response, dict) and response.get("messages"):
                    last_message = response["messages"][-1]
                    if hasattr(last_message, "content"):
                        content = str(last_message.content)
                    elif isinstance(last_message, dict):
                        content = str(last_message.get("content", ""))
                    else:
                        content = str(last_message)
                elif hasattr(response, "content"):
                    content = str(response.content)
                else:
                    content = str(response)
                status = "ok"
                return content
            except Exception as exc:
                self.metrics["delegation_failures"] += 1
                per_agent["failures"] += 1
                status = "error"
                content = (
                    f"Delegation to '{worker_name}' failed. "
                    f"Error: {exc.__class__.__name__}: {exc}"
                )
                logger.exception("Delegation to worker '%s' failed", worker_name)
                return content
            finally:
                latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
                per_agent["last_latency_ms"] = latency_ms
                self.metrics["recent_delegations"].append(
                    {
                        "agent": worker_name,
                        "agent_id": worker_agent_id,
                        "role": worker_role,
                        "thread_id": child_thread_id,
                        "delegated_via": delegated_via,
                        "status": status,
                        "latency_ms": latency_ms,
                    }
                )
                self.metrics["recent_delegations"] = self.metrics["recent_delegations"][-20:]

        for agent in self.agents:
            worker = agent.get("app") if isinstance(agent, dict) else agent
            agent_name = (
                agent.get("name")
                if isinstance(agent, dict)
                else getattr(agent, "name", None)
            ) or f"agent_{len(tools) + 1}"
            agent_role = agent.get("role") if isinstance(agent, dict) else None
            agent_id = agent.get("agent_id") if isinstance(agent, dict) else None
            safe_tool_name = (
                agent_name.lower().replace(" ", "_").replace("-", "_")
            )
            input_model = create_model(
                f"{safe_tool_name}_Input",
                query=(str, Field(description=f"Question or task for {agent_name}")),
            )

            async def call_agent(
                query: str,
                config: RunnableConfig | None = None,
                *,
                _agent=worker,
                _agent_name=agent_name,
                _agent_id=agent_id,
                _agent_role=agent_role,
                _agent_params=(agent.get("params") if isinstance(agent, dict) else None),
            ) -> str:
                allowed, reason = evaluate_member_policy(
                    {
                        "agent_id": _agent_id,
                        "name": _agent_name,
                        "role": _agent_role,
                        "params": _agent_params,
                    },
                    query,
                )
                if not allowed:
                    return (
                        f"Delegation to '{_agent_name}' denied by policy"
                        f" ({reason})."
                    )
                return await invoke_worker(
                    worker=_agent,
                    worker_name=_agent_name,
                    worker_agent_id=_agent_id,
                    worker_role=_agent_role,
                    query=query,
                    config=config,
                    delegated_via="single",
                )

            tool_instance = StructuredTool.from_function(
                coroutine=call_agent,
                name=safe_tool_name,
                description=(
                    f"Delegate work to agent '{agent_name}'"
                    + (f" with role '{agent_role}'." if agent_role else ".")
                    + " Use this when the task matches that agent's specialization."
                ),
                args_schema=input_model,
            )
            tools.append(tool_instance)
            agent_tool_map[safe_tool_name] = {
                "worker": worker,
                "agent_name": agent_name,
                "agent_id": agent_id,
                "role": agent_role,
            }

        if self.parallel_tool_calls and len(agent_tool_map) > 1:

            @tool
            async def parallel_delegate(tasks: list[ParallelDelegationTask]) -> str:
                """Delegate independent subtasks to multiple agents in parallel and return their results."""
                if not tasks:
                    return "No parallel tasks provided."
                self.metrics["parallel_delegations_total"] += 1
                self.metrics["parallel_tasks_total"] += len(tasks)
                coroutines = []
                labels = []
                for task in tasks:
                    target = agent_tool_map.get(task.agent_name)
                    if not target:
                        coroutines.append(asyncio.sleep(0, result=f"Unknown agent '{task.agent_name}'"))
                        labels.append(task.agent_name)
                        continue
                    labels.append(target["agent_name"])
                    coroutines.append(
                        invoke_worker(
                            worker=target["worker"],
                            worker_name=target["agent_name"],
                            worker_agent_id=target["agent_id"],
                            worker_role=target.get("role"),
                            query=task.query,
                            config=None,
                            delegated_via="parallel",
                        )
                    )
                results = await asyncio.gather(*coroutines)
                return "\n\n".join(
                    f"[{labels[index]}]\n{result}" for index, result in enumerate(results)
                )

            tools.append(parallel_delegate)
        return tools

    def _build_supervisor_prompt(self, agent_tools: list[BaseTool]) -> str:
        agent_list = "\n".join(
            f"- {tool.name}: {tool.description}" for tool in agent_tools
        )
        base_prompt = self.prompt.strip()
        return (
            f"{base_prompt}\n\n"
            "Available delegate agents:\n"
            f"{agent_list}\n\n"
            "Delegate to the most relevant agent tool when specialized work is needed. "
            "If multiple independent subtasks can run concurrently, use parallel_delegate. "
            "After any tool call, return a direct final answer to the user."
        )

    def get_metrics(self) -> dict[str, Any]:
        return {
            "delegations_total": self.metrics["delegations_total"],
            "delegation_failures": self.metrics["delegation_failures"],
            "parallel_delegations_total": self.metrics["parallel_delegations_total"],
            "parallel_tasks_total": self.metrics["parallel_tasks_total"],
            "per_agent": dict(self.metrics["per_agent"]),
            "recent_delegations": list(self.metrics["recent_delegations"]),
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
    )
    async def _create_supervisor_workflow(self) -> None:
        try:
            external_tools = _dicts_to_tools(await self._load_supervisor_mcp_tools())
            agent_tools = self._build_agent_tools()
            all_tools = agent_tools + external_tools
            logger.info(
                "Supervisor loaded %d agent tools and %d external tools",
                len(agent_tools),
                len(external_tools),
            )

            hitl_tools = _parse_hitl_tools()
            self.supervisor_agent = lc_create_agent(
                model=self.llm,
                tools=all_tools,
                system_prompt=self._build_supervisor_prompt(agent_tools),
                checkpointer=self.memory,
                middleware=(
                    [HumanInTheLoopMiddleware(interrupt_on=hitl_tools)]
                    if hitl_tools
                    else []
                ),
                state_schema=RuntimeAgentState,
                name="global_supervisor",
            )
            self.app = self.supervisor_agent
        except (ConnectionError, TimeoutError, OSError):
            raise
        except Exception as e:
            logger.error("Supervisor workflow initialization failed: %s", e, exc_info=True)
            raise

    async def init_supervisor(self):
        self.llm = _create_model(
            provider=self.supervisor_provider,
            model_name=self.supervisor_model_name,
            temperature=0.0,
        )
        await self._create_supervisor_workflow()
        logger.info("Supervisor initialized and compiled successfully.")
        return self.app

    async def invoke(self, messages: list[dict], thread_id: str | None = None) -> Any:
        if self.app is None:
            raise RuntimeError(
                "Supervisor must be initialized. Call init_supervisor() first."
            )
        config_dict = _build_invoke_config(
            thread_id=thread_id or uuid.uuid4().hex,
            run_name="supervisor_invoke",
            tags=["mao", "supervisor"],
            metadata={
                "supervisor_provider": self.supervisor_provider,
                "supervisor_model_name": self.supervisor_model_name,
            },
        )
        with langsmith_trace(
            run_name="supervisor_invoke",
            tags=config_dict.get("tags"),
            metadata=config_dict.get("metadata"),
        ):
            return await self.app.ainvoke({"messages": messages}, config=config_dict)


async def create_agent(
    provider: str,
    model_name: str,
    agent_name: str | None = None,
    system_prompt: str | None = None,
    tools: MCPClient | list[dict[str, Any]] | None = None,
    skills: list[str] | None = None,
    skill_paths: list[str] | None = None,
    temperature: float = 0.0,
    stream: bool = False,
) -> Any:
    if not agent_name:
        sanitized_model_name = model_name.replace(".", "_").replace("/", "_")
        agent_name = f"{provider}_{sanitized_model_name}_agent"

    llm_instance = _create_model(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        stream=stream,
    )

    agent_instance = Agent(
        llm_instance=llm_instance,
        agent_name=agent_name,
        tools=tools,
        system_prompt=system_prompt,
        skills=skills,
        skill_paths=skill_paths,
        stream=stream,
    )

    compiled_app = await agent_instance.init_agent()
    return compiled_app
