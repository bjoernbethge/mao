"""
Basic toolkit for LangChain agents and workflows.
Includes HTTP tools (GET, POST, PUT, PATCH, DELETE), Office/PDF/Directory loader, REPL, and Wikipedia.
"""

from typing import List

from langchain_community.agent_toolkits.openapi.toolkit import RequestsToolkit
from langchain_community.utilities.requests import TextRequestsWrapper
from langchain_community.tools import PythonREPLTool, WikipediaAPIWrapper
from langchain_community.document_loaders import (
    DirectoryLoader,
    PDFPlumberLoader,
    UnstructuredFileLoader,
)

# HTTP Tools (GET, POST, PUT, PATCH, DELETE)
requests_toolkit = RequestsToolkit(
    requests_wrapper=TextRequestsWrapper(headers={}),
    allow_dangerous_requests=True,  # Achtung: Nur aktivieren, wenn du die Risiken kennst!
)
http_tools: List = requests_toolkit.get_tools()

# Weitere Tools
repl_tool = PythonREPLTool()
wikipedia_tool = WikipediaAPIWrapper()

# Document Loader Beispiele (für Office, PDF, Directory)
directory_loader = DirectoryLoader("data/")
pdf_loader = PDFPlumberLoader("file.pdf")
office_loader = UnstructuredFileLoader("file.docx")

# Export für Agenten/Chains
__all__ = [
    "http_tools",
    "repl_tool",
    "wikipedia_tool",
    "directory_loader",
    "pdf_loader",
    "office_loader",
]
