"""
Refactored agent classes with a generic Agent, a Supervisor, and a create_agent factory.
Includes best-practice memory, checkpointing, and state management for production chatbots.
"""

from typing import List, Optional, Any, Union, Callable, Dict, Tuple, TypeVar
import asyncio
import os
import logging
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from dotenv import load_dotenv

# LangChain Core
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import SystemMessage, AIMessage, BaseMessage, trim_messages
from langchain_core.runnables import RunnableConfig  # For config in invoke/ainvoke

# LLM Clients
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import OllamaLLM

# LangGraph
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import ToolNode, create_react_agent
from .storage import KnowledgeTree, ExperienceTree
from .mcp import MCPClient

load_dotenv()

T = TypeVar("T")


def get_default_callbacks() -> List[BaseCallbackHandler]:
    """Returns default callbacks, either LangSmith tracer if configured or simple logging callbacks."""
    tracer = None
    try:
        from langchain.callbacks.tracers import LangChainTracer

        if os.environ.get("LANGSMITH_TRACING") == "true" and os.environ.get(
            "LANGSMITH_API_KEY"
        ):
            tracer = LangChainTracer(
                project_name=os.environ.get("LANGSMITH_PROJECT", "mcp-agents")
            )
    except ImportError:
        pass
    if tracer:
        return [tracer]
    else:

        class LoggingCallbackHandler(BaseCallbackHandler):
            def on_chain_end(self, outputs, **kwargs):
                logging.info(f"Chain finished: {outputs}")

            def on_chain_error(self, error, **kwargs):
                logging.error(f"Chain error: {error}")

            def on_llm_end(self, response, **kwargs):
                logging.info(f"LLM finished: {response}")

            def on_llm_error(self, error, **kwargs):
                logging.error(f"LLM error: {error}")

        return [LoggingCallbackHandler()]


DEFAULT_CALLBACKS = get_default_callbacks()


def _create_llm_client(
    provider: str,
    model_name: str,
    temperature: float = 0.0,
    callbacks: Optional[List[BaseCallbackHandler]] = None,
    stream: bool = False,
    **kwargs: Any,
) -> BaseChatModel:
    """Creates an LLM client based on provider, with appropriate configuration."""
    provider_lower = provider.lower()
    actual_callbacks = callbacks or DEFAULT_CALLBACKS
    llm_specific_kwargs = kwargs.copy()

    if provider_lower == "openai":
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            callbacks=actual_callbacks,
            streaming=stream,
            **llm_specific_kwargs,
        )
    elif provider_lower == "anthropic":
        # Add Claude 3 tools beta header if needed
        if (
            "claude-3" in model_name.lower()
            and "anthropic-beta" not in llm_specific_kwargs.get("default_headers", {})
        ):
            llm_specific_kwargs.setdefault("default_headers", {})
            llm_specific_kwargs["default_headers"][
                "anthropic-beta"
            ] = "tools-2024-04-04"
        return ChatAnthropic(
            model=model_name,
            temperature=temperature,
            callbacks=actual_callbacks,
            streaming=stream,
            **llm_specific_kwargs,
        )
    elif provider_lower == "ollama":
        if OllamaLLM is None:
            raise ImportError(
                "OllamaLLM is not available. Please install langchain_ollama."
            )
        ollama_final_kwargs = {
            "model": model_name,
            "temperature": temperature,
            "callbacks": actual_callbacks,
            **llm_specific_kwargs,
        }
        ollama_host = os.environ.get("OLLAMA_HOST")
        if ollama_host and "base_url" not in ollama_final_kwargs:
            ollama_final_kwargs["base_url"] = ollama_host
        return OllamaLLM(**ollama_final_kwargs)  # type: ignore
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


async def load_mcp_tools(
    mcp_client: Union[MCPClient, List[Dict[str, Any]], None],
) -> List[Dict[str, Any]]:
    """
    Generic helper to load tools from an MCPClient.
    Used by both Agent and Supervisor classes.

    Args:
        mcp_client: Either an MCPClient instance, a list of tool definitions, or None

    Returns:
        List of tool definitions
    """
    if mcp_client is None:
        return []

    if isinstance(mcp_client, MCPClient):
        logging.debug("Entering MCPClient context to load tools...")
        async with mcp_client.session():
            try:
                # Verwende die neue load_tools Methode für alle aktiven Server
                all_tools = []
                active_servers = mcp_client.list_active_servers()

                if not active_servers:
                    logging.warning("No active servers found in MCPClient")
                    return []

                for server_name in active_servers:
                    try:
                        server_tools = mcp_client.load_tools(server_name)
                        logging.debug(
                            f"Loaded {len(server_tools)} tools from server '{server_name}'"
                        )
                        all_tools.extend(server_tools)
                    except Exception as e:
                        logging.error(
                            f"Failed to load tools from server '{server_name}': {e}"
                        )

                logging.debug(
                    f"Successfully loaded total of {len(all_tools)} tools from all active servers"
                )
                return all_tools
            except Exception as e:
                logging.error(f"Error loading MCP tools: {e}")
                return []
    elif isinstance(mcp_client, list):
        return mcp_client
    else:
        logging.warning(f"Unexpected mcp_client type: {type(mcp_client)}")
        return []


def _determine_tokenizer_for_trim(llm: BaseChatModel) -> Union[BaseChatModel, str]:
    """Helper to determine the appropriate tokenizer for message trimming."""
    if isinstance(llm, OllamaLLM):
        return llm.model
    elif not (
        hasattr(llm, "get_num_tokens_from_messages") or hasattr(llm, "get_num_tokens")
    ):
        if hasattr(llm, "model_name"):
            return llm.model_name
        elif hasattr(llm, "model"):
            return llm.model
        else:
            logging.warning(
                "Defaulting tokenizer for trim_messages to 'gpt-3.5-turbo'."
            )
            return "gpt-3.5-turbo"
    return llm


async def _process_llm_response(
    response: Any,
    stream: bool,
    token_callback: Optional[Callable[[str], None]] = None,
    streamed_content: str = "",
) -> Tuple[BaseMessage, str]:
    """Helper to process and standardize LLM responses into the proper format."""
    if isinstance(response, AIMessage):
        message = response
        content = str(message.content or streamed_content)
    elif isinstance(response, str):
        message = AIMessage(content=response)
        content = response
    else:
        message = AIMessage(content=str(response))
        content = str(response)

    # Ensure streamed content is properly set if we were streaming
    if stream and token_callback and streamed_content and not message.content:
        message.content = streamed_content

    return message, content


class Agent:
    def __init__(
        self,
        llm_instance: BaseChatModel,
        agent_name: str,
        tools: Optional[Union[MCPClient, List[Dict[str, Any]]]] = None,
        system_prompt: Optional[str] = None,
        knowledge_tree: Optional[KnowledgeTree] = None,
        experience_tree: Optional[ExperienceTree] = None,
        memory_saver: Optional[MemorySaver] = None,
        callbacks: Optional[List[BaseCallbackHandler]] = None,
        stream: bool = False,
        token_callback: Optional[Callable[[str], None]] = None,
        max_tokens_trimmed: int = 3000,
        use_react_agent: bool = False,
    ):
        self.llm = llm_instance
        self.name = agent_name
        self.configured_tools = tools
        self.loaded_tools: List[Dict[str, Any]] = []
        self.system_prompt = system_prompt or "You are a helpful assistant."
        self.knowledge_tree = knowledge_tree
        self.experience_tree = experience_tree
        self.memory = memory_saver or MemorySaver()
        self.callbacks = callbacks or DEFAULT_CALLBACKS
        self.stream = stream
        self.token_callback = token_callback
        self.max_tokens_trimmed = max_tokens_trimmed
        self.agent_runnable = None
        self.use_react_agent = use_react_agent

    async def _load_mcp_tools(self) -> List[Dict[str, Any]]:
        """Loads tools if self.configured_tools is an MCPClient instance."""
        return await load_mcp_tools(self.configured_tools)

    async def _retrieve_context(self, query: str, k: int = 3) -> str:
        """Retrieves relevant context from knowledge and experience databases."""
        context_parts = []
        if self.knowledge_tree:
            # Verwende die asynchrone Methode, wenn verfügbar
            try:
                knowledge_hits = await self.knowledge_tree.search_async(query, k=k)
            except AttributeError:
                # Fallback für ältere Versionen oder nicht initialisierte Instanzen
                knowledge_hits = await asyncio.to_thread(
                    self.knowledge_tree.search, query, k=k
                )

            if knowledge_hits:
                context_parts.append(
                    "\nRelevant Knowledge:\n"
                    + "\n".join([h["page_content"] for h in knowledge_hits])
                )

        if self.experience_tree:
            # Verwende die asynchrone Methode, wenn verfügbar
            try:
                experience_hits = await self.experience_tree.search_async(query, k=k)
            except AttributeError:
                # Fallback für ältere Versionen oder nicht initialisierte Instanzen
                experience_hits = await asyncio.to_thread(
                    self.experience_tree.search, query, k=k
                )

            if experience_hits:
                context_parts.append(
                    "\nRelevant Experience:\n"
                    + "\n".join([h["page_content"] for h in experience_hits])
                )

        return "".join(context_parts).strip()

    async def _learn_experience(
        self, user_input: str, model_output: str, tags: Optional[List[str]] = None
    ) -> None:
        """Stores interactions as experience for future reference."""
        if not self.experience_tree:
            logging.debug("Experience tree not available, skipping learning.")
            return

        knowledge_id = None
        if self.knowledge_tree and user_input:
            try:
                knowledge_hits = await self.knowledge_tree.search_async(user_input, k=1)
            except AttributeError:
                knowledge_hits = await asyncio.to_thread(
                    self.knowledge_tree.search, user_input, k=1
                )

            if knowledge_hits:
                knowledge_id = knowledge_hits[0].get("id")

        exp_text = f"User: {user_input}\nAgent: {model_output}"

        try:
            # Versuche asynchrone Methode
            await self.experience_tree.learn_from_experience_async(
                exp_text, related_knowledge_id=knowledge_id, tags=tags
            )
        except AttributeError:
            # Fallback auf synchrone Methode
            await asyncio.to_thread(
                self.experience_tree.learn_from_experience,
                exp_text,
                related_knowledge_id=knowledge_id,
                tags=tags,
            )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
    )
    async def _call_model_node(
        self, state: MessagesState, config: Optional[RunnableConfig] = None
    ) -> Dict[str, List[BaseMessage]]:
        """Core method to process messages, call the LLM, and update state."""
        user_input = state["messages"][-1].content if state["messages"] else ""
        context = await self._retrieve_context(user_input)

        # Determine appropriate tokenizer for trimming messages
        tokenizer_for_trim = _determine_tokenizer_for_trim(self.llm)

        # Create trimmed message history for the LLM
        trimmed_messages_list = trim_messages(
            state["messages"],
            max_tokens=self.max_tokens_trimmed,
            strategy="last",
            token_counter=tokenizer_for_trim,
            include_system=True,
            start_on="human",
        )

        # Prepare input for the LLM
        system_content = self.system_prompt + (f"\n{context}" if context else "")
        messages_for_llm: List[BaseMessage] = [
            SystemMessage(content=system_content)
        ] + trimmed_messages_list

        try:
            # Setup LLM call parameters
            llm_call_kwargs = {}
            if self.loaded_tools and hasattr(self.llm, "bind_tools"):
                llm_call_kwargs["tools"] = self.loaded_tools

            streamed_content = ""
            if self.stream and self.token_callback:
                # Handle streaming mode
                async for chunk in self.llm.astream(
                    messages_for_llm, **llm_call_kwargs
                ):
                    token = chunk.content if hasattr(chunk, "content") else str(chunk)
                    self.token_callback(token)
                    streamed_content += token

                # After streaming, get the full response for state tracking
                invoked_response = await self.llm.ainvoke(
                    messages_for_llm, **llm_call_kwargs
                )
            else:
                # Non-streaming mode
                invoked_response = await self.llm.ainvoke(
                    messages_for_llm, **llm_call_kwargs
                )

            # Process the response to standard format
            response_message, learn_content = await _process_llm_response(
                invoked_response, self.stream, self.token_callback, streamed_content
            )

            # Store the interaction as experience
            await self._learn_experience(user_input, learn_content)

            return {"messages": [response_message]}
        except Exception as e:
            logging.error(f"Agent '{self.name}' model call failed: {e}", exc_info=True)
            raise

    async def init_agent(self):
        """Initialize the agent, loading tools and setting up the workflow."""
        try:
            # Initialisiere die Storage-Komponenten asynchron, wenn sie noch nicht initialisiert wurden
            if self.knowledge_tree is None:
                self.knowledge_tree = await KnowledgeTree.create()
            elif (
                hasattr(self.knowledge_tree, "async_init")
                and self.knowledge_tree.embed is None
            ):
                await self.knowledge_tree.async_init()

            if self.experience_tree is None:
                self.experience_tree = await ExperienceTree.create()
            elif (
                hasattr(self.experience_tree, "async_init")
                and self.experience_tree.embed is None
            ):
                await self.experience_tree.async_init()

            self.loaded_tools = await self._load_mcp_tools()
            logging.info(f"Agent '{self.name}' loaded {len(self.loaded_tools)} tools")

            if self.use_react_agent and self.loaded_tools:
                # Use LangGraph's prebuilt ReAct agent for better tool usage
                self.agent_runnable = create_react_agent(
                    self.llm, self.loaded_tools, prompt=self.system_prompt
                )
                # Set name for debugging
                self.agent_runnable.name = self.name
                # Set checkpointer
                self.agent_runnable._checkpointer = self.memory
                logging.info(
                    f"Agent '{self.name}' created as ReAct agent with {len(self.loaded_tools)} tools."
                )
                return self.agent_runnable

            # Default graph-based agent implementation
            workflow = StateGraph(MessagesState)
            workflow.add_node("model_node", self._call_model_node)

            if self.loaded_tools and hasattr(self.llm, "bind_tools"):
                # Add tool handling if tools are available
                tool_node = ToolNode(self.loaded_tools)
                workflow.add_node("tool_node", tool_node)

                def should_continue(state: MessagesState) -> str:
                    last_message = state["messages"][-1]
                    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                        return "tool_node"
                    return END

                workflow.add_conditional_edges("model_node", should_continue)
                workflow.add_edge("tool_node", "model_node")
                workflow.set_entry_point("model_node")
            else:
                # Simple workflow without tools
                workflow.add_edge(START, "model_node")
                workflow.add_edge("model_node", END)

            self.agent_runnable = workflow.compile(
                checkpointer=self.memory, name=self.name
            )
            logging.info(f"Agent '{self.name}' compiled successfully.")
            return self.agent_runnable
        except Exception as e:
            logging.error(
                f"Failed to initialize agent '{self.name}': {e}", exc_info=True
            )
            raise

    def get_compiled_app(self):
        """Returns the compiled agent runnable, raising an error if not initialized."""
        if not self.agent_runnable:
            raise RuntimeError("Agent not initialized. Call init_agent() first.")
        return self.agent_runnable


class Supervisor:
    def __init__(
        self,
        agents: List[Any],
        supervisor_provider: str,
        supervisor_model_name: str,
        supervisor_system_prompt: str,
        supervisor_tools: Optional[Union[MCPClient, List[Dict[str, Any]]]] = None,
        callbacks: Optional[List[BaseCallbackHandler]] = None,
        llm_specific_kwargs: Optional[Dict[str, Any]] = None,
        add_handoff_back_messages: bool = True,
        parallel_tool_calls: bool = True,
        **supervisor_kwargs: Any,
    ):
        self.agents = agents
        self.supervisor_provider = supervisor_provider
        self.supervisor_model_name = supervisor_model_name
        self.llm: Optional[BaseChatModel] = None
        self.prompt = supervisor_system_prompt
        self.supervisor_tools = supervisor_tools
        # Create new dictionary with clean parameters
        supervisor_kwargs_dict = {
            "add_handoff_back_messages": add_handoff_back_messages
        }
        supervisor_kwargs_dict["parallel_tool_calls"] = parallel_tool_calls
        supervisor_kwargs_dict.update(supervisor_kwargs)
        self.supervisor_kwargs = supervisor_kwargs_dict
        self.memory = MemorySaver()
        self.app = None
        self.callbacks = callbacks or DEFAULT_CALLBACKS
        self.llm_specific_kwargs = llm_specific_kwargs or {}

    async def _load_supervisor_mcp_tools(self) -> List[Dict[str, Any]]:
        """Loads tools for the supervisor from MCPClient."""
        return await load_mcp_tools(self.supervisor_tools)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
    )
    async def _create_supervisor_workflow(self) -> None:
        """Creates the supervisor workflow with retry capability."""
        try:
            # Load tools asynchronously
            tools = await self._load_supervisor_mcp_tools()
            logging.info(f"Supervisor loaded {len(tools)} tools")

            workflow = create_supervisor(
                self.agents,
                model=self.llm,
                prompt=self.prompt,
                tools=tools,
                **self.supervisor_kwargs,
            )
            self.app = workflow.compile(
                checkpointer=self.memory, name="global_supervisor"
            )
        except Exception as e:
            logging.error(
                f"Supervisor workflow initialization failed: {e}", exc_info=True
            )
            raise

    async def init_supervisor(self):
        """Initializes the supervisor's LLM and workflow."""
        # Initialize LLM
        self.llm = _create_llm_client(
            provider=self.supervisor_provider,
            model_name=self.supervisor_model_name,
            temperature=0.0,
            callbacks=self.callbacks,
            stream=False,
            **(self.llm_specific_kwargs),
        )

        # Create and compile the supervisor workflow
        await self._create_supervisor_workflow()
        logging.info("Supervisor initialized and compiled successfully.")
        return self.app

    async def invoke(
        self, messages: List[dict], thread_id: Optional[str] = None
    ) -> Any:
        """Invokes the supervisor asynchronously with a list of messages."""
        if self.app is None:
            raise RuntimeError(
                "Supervisor must be initialized. Call init_supervisor() first."
            )

        config_dict = {"configurable": {"thread_id": thread_id}} if thread_id else {}
        input_state = {"messages": messages}
        return await self.app.ainvoke(input_state, config=config_dict)


async def create_agent(
    provider: str,
    model_name: str,
    agent_name: Optional[str] = None,
    temperature: float = 0.0,
    tools: Optional[Union[MCPClient, List[Dict[str, Any]]]] = None,
    system_prompt: Optional[str] = None,
    knowledge_tree: Optional[KnowledgeTree] = None,
    experience_tree: Optional[ExperienceTree] = None,
    callbacks: Optional[List[BaseCallbackHandler]] = None,
    stream: bool = False,
    token_callback: Optional[Callable[[str], None]] = None,
    max_tokens_trimmed: int = 3000,
    llm_specific_kwargs: Optional[Dict[str, Any]] = None,
    use_react_agent: bool = True,
) -> Any:
    """Factory function to create and initialize an agent with the specified configuration."""
    if not agent_name:
        sanitized_model_name = model_name.replace(".", "_").replace("/", "_")
        agent_name = f"{provider}_{sanitized_model_name}_agent"

    # Create the LLM instance
    llm_instance = _create_llm_client(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        callbacks=callbacks,
        stream=stream,
        **(llm_specific_kwargs or {}),
    )

    # Initialize storage components asynchron
    kt_instance = knowledge_tree
    et_instance = experience_tree

    # Wenn keine Instanzen übergeben wurden, erstelle neue asynchron
    if kt_instance is None:
        kt_instance = await KnowledgeTree.create()
    elif hasattr(kt_instance, "async_init") and kt_instance.embed is None:
        await kt_instance.async_init()

    if et_instance is None:
        et_instance = await ExperienceTree.create()
    elif hasattr(et_instance, "async_init") and et_instance.embed is None:
        await et_instance.async_init()

    memory = MemorySaver()

    # Create and initialize the agent
    agent_instance = Agent(
        llm_instance=llm_instance,
        agent_name=agent_name,
        tools=tools,
        system_prompt=system_prompt,
        knowledge_tree=kt_instance,
        experience_tree=et_instance,
        memory_saver=memory,
        callbacks=callbacks,
        stream=stream,
        token_callback=token_callback,
        max_tokens_trimmed=max_tokens_trimmed,
        use_react_agent=use_react_agent,
    )

    compiled_app = await agent_instance.init_agent()
    return compiled_app


async def main_example():
    """Example usage of the agent system."""
    await create_agent(provider="openai", model_name="gpt-3.5-turbo")
    logging.info("OpenAI Agent App created.")
    # ... other examples ...


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    # asyncio.run(main_example())
    pass
