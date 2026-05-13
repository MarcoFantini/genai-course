"""
day3_multiagent/mcp_tool_server.py

Server MCP minimale — espone SOLO i tool ITSM su HTTP, niente agent.

Punto didattico:
    Questo processo è il "MCP Server". È completamente separato dall'agent
    su porta 8001. L'agent non sa nulla di questo server: sa solo che
    GET /mcp/tools restituisce un contratto {name, description, inputSchema}
    e POST /mcp/tools/{name}/call esegue il tool.

    Questo è esattamente il vantaggio di MCP: il server può essere scritto
    in qualsiasi linguaggio, girare su qualsiasi host, essere sostituito
    senza toccare l'agent.

Avvio:
    cd /path/to/genai-course
    source .venv/bin/activate
    uvicorn day3_multiagent.mcp_tool_server:app --port 8002 --reload

Endpoints esposti:
    GET  /mcp/tools              — discovery: lista tutti i tool con inputSchema
    GET  /mcp/tools/{name}       — schema completo di un singolo tool
    POST /mcp/tools/{name}/call  — esecuzione diretta di un tool (body: {"args":{...}})
    GET  /health                 — liveness check
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent.resolve()
_PROJECT_ROOT = _HERE.parent
for _p in [str(_HERE), str(_PROJECT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mcp_lab import build_adapter  # noqa: E402

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="MCP Tool Server",
    description=(
        "Server MCP minimale — espone SOLO i tool ITSM, niente agent.\n\n"
        "**Punto didattico**: questo è il lato *server* del protocollo MCP.\n"
        "L'agent su porta 8001 usa `MCPHTTPAdapter` per chiamare questi endpoint,\n"
        "esattamente come farebbe con un server MCP remoto reale.\n\n"
        "Il server non sa nulla dell'agent. L'agent non sa nulla del server.\n"
        "Comunicano solo tramite il contratto MCP: `name + description + inputSchema`.\n\n"
        "Avvio: `uvicorn day3_multiagent.mcp_tool_server:app --port 8002`"
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class CallToolRequest(BaseModel):
    args: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> Dict[str, str]:
    """Liveness check — usato dalla console HTML per verificare che il server sia attivo."""
    adapter = build_adapter()
    tool_names = [t["name"] for t in adapter.list_tools()]
    return {
        "status": "ok",
        "server": "mcp_tool_server",
        "port": "8002",
        "tools": ", ".join(tool_names),
    }


@app.get(
    "/mcp/tools",
    summary="Discovery — lista tutti i tool esposti dal server",
    description=(
        "Il client MCP chiama questo endpoint per sapere quali tool sono disponibili.\n"
        "Ogni tool include `name`, `description` e `inputSchema` (JSON Schema).\n\n"
        "Nessun LLM coinvolto — è pura discovery di contratto."
    ),
)
async def list_tools() -> List[Dict[str, Any]]:
    return build_adapter().list_tools()


@app.get(
    "/mcp/tools/{name}",
    summary="Schema completo di un tool",
    description="Restituisce il contratto MCP completo del tool: name, description, inputSchema.",
)
async def get_tool(name: str) -> Dict[str, Any]:
    spec = build_adapter().get_tool(name)
    if not spec:
        raise HTTPException(status_code=404, detail=f"Tool '{name}' non trovato.")
    return spec


@app.post(
    "/mcp/tools/{name}/call",
    summary="Esecuzione diretta di un tool",
    description=(
        "Esegue il tool con gli argomenti forniti nel body `{\"args\": {...}}`.\n\n"
        "Questo è il cuore del protocollo MCP: il client invia `name + args`,\n"
        "il server esegue e restituisce il risultato. Niente LLM, niente agent."
    ),
)
async def call_tool(name: str, request: CallToolRequest) -> Dict[str, Any]:
    adapter = build_adapter()
    result = adapter.call_tool(name, **request.args)
    return {
        "tool": name,
        "args": request.args,
        "mcp_result": result,
        "server": "mcp_tool_server:8002",
    }
