from __future__ import annotations

import asyncio
import json
import sys
from asyncio import AbstractEventLoop
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

_HERE = Path(__file__).parent.resolve()
_PROJECT_ROOT = _HERE.parent
for _p in [str(_HERE), str(_PROJECT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mcp_lab import (  # noqa: E402
    EXAMPLE_PROMPTS,
    MCPAdapter,
    MCPHTTPAdapter,
    _trace_callback,
    build_adapter,
    run_agent,
    run_agent_fast,
)

app = FastAPI(
    title="Day3 MCP Lab API",
    description=(
        "Espone i tool MCP del laboratorio Giorno 3 — pomeriggio.\n\n"
        "Endpoint principali:\n"
        "- GET /mcp/tools        — discovery: lista i tool esposti dal server\n"
        "- GET /mcp/tools/{name} — schema completo di un tool (inputSchema JSON)\n"
        "- POST /mcp/tools/{name}/call — chiamata diretta, bypassa l'LLM\n"
        "- GET /mcp/agent/stream — agent ReAct con SSE streaming delle trace\n"
        "- POST /mcp/agent       — agent sincrono (no streaming)\n"
        "- GET /mcp/examples     — prompt di esempio per il lab"
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_executor = ThreadPoolExecutor(max_workers=4)


# ---------------------------------------------------------------------------
# Modelli request/response
# ---------------------------------------------------------------------------

class CallToolRequest(BaseModel):
    args: Dict[str, Any] = {}


class AgentRequest(BaseModel):
    query: str
    fast: bool = False


# ---------------------------------------------------------------------------
# Helpers SSE
# ---------------------------------------------------------------------------

def _sse_event(data: Any, event: str = "message") -> str:
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


# ---------------------------------------------------------------------------
# Streaming agent
# ---------------------------------------------------------------------------

async def _stream_agent(
    query: str, fast: bool, remote_url: Optional[str] = None
) -> AsyncGenerator[str, None]:
    loop: AbstractEventLoop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    SENTINEL = object()

    def on_event(event_dict: dict) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, event_dict)

    token = _trace_callback.set(on_event)
    ctx = copy_context()
    _trace_callback.reset(token)

    # Se remote_url è fornito, i tool vengono scoperti e chiamati su quel server.
    # L'agent non cambia nulla — riceve un adapter con la stessa interfaccia.
    adapter = MCPHTTPAdapter(remote_url) if remote_url else build_adapter()
    tool_source = remote_url if remote_url else "local (in-process)"

    def run_sync() -> Any:
        try:
            if fast:
                return run_agent_fast(query, adapter=adapter)
            return run_agent(query, adapter=adapter)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, SENTINEL)

    future = loop.run_in_executor(_executor, ctx.run, run_sync)

    while True:
        try:
            item = await asyncio.wait_for(queue.get(), timeout=180.0)
        except asyncio.TimeoutError:
            yield _sse_event({"detail": "Timeout: nessun evento dall'agente"}, event="error")
            return

        if item is SENTINEL:
            break

        yield _sse_event(item, event="trace")

    try:
        answer, state = await future
    except Exception as exc:
        yield _sse_event({"detail": str(exc)}, event="error")
        return

    yield _sse_event(
        {
            "answer": answer,
            "tool_calls": state.get("tool_calls", []),
            "citations": state.get("citations", []),
            "tokens_in": state.get("tokens_in", 0),
            "tokens_out": state.get("tokens_out", 0),
            "fast_mode": state.get("fast_mode", False),
            "total_traces": len(state.get("trace", [])),
            "tool_source": tool_source,
        },
        event="done",
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/mcp/examples")
async def get_examples() -> List[Dict[str, Any]]:
    """Restituisce i prompt di esempio del laboratorio MCP."""
    return EXAMPLE_PROMPTS


@app.get("/mcp/tools")
async def list_tools() -> List[Dict[str, Any]]:
    """
    Discovery: lista tutti i tool esposti dal server MCP.

    Punto didattico: il client riceve name + description + inputSchema
    senza sapere nulla dell'implementazione.
    """
    adapter = build_adapter()
    return adapter.list_tools()


@app.get("/mcp/tools/{name}")
async def get_tool(name: str) -> Dict[str, Any]:
    """
    Schema completo di un singolo tool MCP.

    Mostra il contratto che l'LLM vede quando deve decidere se chiamarlo.
    """
    adapter = build_adapter()
    spec = adapter.get_tool(name)
    if not spec:
        raise HTTPException(status_code=404, detail=f"Tool '{name}' non trovato.")
    return spec


@app.post("/mcp/tools/{name}/call")
async def call_tool(name: str, request: CallToolRequest) -> Dict[str, Any]:
    """
    Chiamata diretta a un tool MCP, bypassa completamente l'LLM.

    Punto didattico: dimostra che i tool MCP sono invocabili da qualsiasi
    client — non solo dall'LLM. Stessa firma di adapter.call_tool().
    """
    loop = asyncio.get_running_loop()
    adapter = build_adapter()

    def _call() -> Dict[str, Any]:
        return adapter.call_tool(name, **request.args)

    try:
        result = await loop.run_in_executor(_executor, _call)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "tool": name,
        "args": request.args,
        "mcp_result": result,
    }


@app.post("/mcp/agent")
async def run_agent_sync(request: AgentRequest) -> Dict[str, Any]:
    """Esecuzione sincrona dell'agent MCP (no streaming)."""
    loop = asyncio.get_running_loop()
    adapter = build_adapter()

    def _run() -> Any:
        if request.fast:
            return run_agent_fast(request.query, adapter=adapter)
        return run_agent(request.query, adapter=adapter)

    try:
        answer, state = await loop.run_in_executor(_executor, _run)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "answer": answer,
        "tool_calls": state.get("tool_calls", []),
        "citations": state.get("citations", []),
        "tokens_in": state.get("tokens_in", 0),
        "tokens_out": state.get("tokens_out", 0),
        "fast_mode": state.get("fast_mode", False),
        "trace": state.get("trace", []),
    }


@app.get("/mcp/agent/stream")
async def run_agent_stream(
    query: str = Query(...),
    fast: bool = Query(False),
    remote_url: Optional[str] = Query(
        None,
        description=(
            "Se fornito, i tool vengono scoperti e chiamati su questo server HTTP remoto "
            "tramite MCPHTTPAdapter. L'agent non cambia nulla."
        ),
    ),
) -> StreamingResponse:
    """
    Esecuzione dell'agent con SSE streaming delle trace.

    Ogni chiamata tool e risposta LLM viene trasmessa come evento SSE
    in real-time, prima della risposta finale (evento 'done').

    Se `remote_url` è fornito (es. http://localhost:8002), i tool vengono
    chiamati su quel server invece che in-process — questa è la dimostrazione
    pratica di MCP: stesso agent, tool source intercambiabile.
    """
    return StreamingResponse(
        _stream_agent(query, fast, remote_url),
        media_type="text/event-stream",
    )


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_console() -> HTMLResponse:
    """
    Serve mcp_console.html direttamente da uvicorn.

    Apri http://localhost:8001 invece di aprire il file locale:
    questo evita restrizioni file:// su EventSource (SSE).
    """
    html_path = _HERE / "mcp_console.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="mcp_console.html non trovato.")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
