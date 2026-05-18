from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shlex
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict, Literal
from datetime import datetime, timedelta
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_google_vertexai import ChatVertexAI
from contextvars import ContextVar
from pydantic import BaseModel, Field, create_model
from typing_extensions import Annotated

try:
    from langgraph.graph.message import add_messages
except Exception:
    # LangGraph non è obbligatorio per questo lab MCP.
    # Il fallback permette l'import del file anche senza LangGraph installato.
    def add_messages(x: Any) -> Any:  # type: ignore
        return x


"""
day3_multiagent/mcp_lab.py

Esercitazione Giorno 3 — Pomeriggio:
- esporre tool del dominio ITSM tramite un'interfaccia MCP-like;
- usare un MCPAdapter simulato con la stessa interfaccia concettuale del client reale;
- integrare i tool MCP nel KnowledgeAgent del Giorno 3;
- inspector "fatto in casa": list-tools, call, schema;
- modalità --fast deterministica per demo robuste anche con quota LLM esaurita;
- salvataggio JSON completo della run per debug, audit didattico e cost tracking;
- bonus: server FastMCP reale opzionale, abilitato se la libreria mcp è installata.

Architettura:

           Host (questo script)
                  |
                  v
         +-----------------+
         |  MCP Client     |       wrappa i tool MCP come BaseTool LangChain
         +-----------------+
                  |
        +---------+---------+
        v                   v
     MCPAdapter         FastMCP server         (un solo lato attivo)
   (in-process)          (subprocess)
        |                   |
        v                   v
    tool del dominio   tool del dominio
   (search_kb,...)    (search_kb,...)

L'adapter simulato ha la stessa interfaccia concettuale del client MCP:
    list_tools()              -> list[dict]
    get_tool(name)            -> dict | None
    call_tool(name, **kwargs) -> dict

Il punto didattico:
    l'agente non deve sapere se search_kb è una funzione Python locale,
    un tool esposto via server MCP stdio, o un tool remoto. Sa solo:
    c'è un tool con name, description e inputSchema.

Prima di usare la RAG come tool:
    python day1_morning_rag/main.py setup-data
    python day1_morning_rag/main.py ingest

Comandi:
    python day3_multiagent/mcp_lab.py examples
    python day3_multiagent/mcp_lab.py list-tools
    python day3_multiagent/mcp_lab.py describe search_kb
    python day3_multiagent/mcp_lab.py call search_kb '{"query":"policy P1","top_k":2}'
    python day3_multiagent/mcp_lab.py agent "Mostrami INC-1002 e cerca la policy P1"
    python day3_multiagent/mcp_lab.py agent "Mostrami INC-1002 e cerca la policy P1" --fast
    python day3_multiagent/mcp_lab.py agent "Mostrami INC-1002 e cerca la policy P1" --verbose
    python day3_multiagent/mcp_lab.py cost-from-run runs/day3_mcp/agent_YYYYMMDD_HHMMSS_xxxxxxxx.json
    python day3_multiagent/mcp_lab.py serve            # avvia FastMCP reale, se installato
    python day3_multiagent/mcp_lab.py inspector        # discovery del server reale via subprocess

Variabili .env:
    GOOGLE_API_KEY=...
    GEMINI_MODEL=gemini-2.5-flash
    MIN_SECONDS_BETWEEN_MODEL_CALLS=15
    MAX_AGENT_ITERATIONS=5
    MAX_TOTAL_SECONDS=120
    MCP_BACKEND=adapter            # "adapter" (default) | "fastmcp"
    MCP_SERVER_CMD="python day3_multiagent/mcp_lab.py serve"
    PRICE_INPUT_PER_1M=0.10
    PRICE_OUTPUT_PER_1M=0.40

Nota didattica:
    MCP standardizza il contratto del tool. LangChain, Gemini, Claude o un
    client custom possono usare lo stesso tool perché ricevono un name,
    una description e uno schema argomenti.
"""


# ---------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------

load_dotenv()

BASE_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = BASE_DIR.parent if BASE_DIR.name == "day3_multiagent" else BASE_DIR

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

MIN_SECONDS_BETWEEN_MODEL_CALLS = float(
    os.getenv("MIN_SECONDS_BETWEEN_MODEL_CALLS", "15")
)

MAX_AGENT_ITERATIONS = int(os.getenv("MAX_AGENT_ITERATIONS", "5"))
MAX_TOTAL_SECONDS = float(os.getenv("MAX_TOTAL_SECONDS", "120"))

MCP_BACKEND = os.getenv("MCP_BACKEND", "adapter").lower().strip()

MCP_SERVER_CMD = os.getenv(
    "MCP_SERVER_CMD",
    f"{sys.executable} {Path(__file__).resolve()} serve",
)

PRICE_INPUT_PER_1M = float(os.getenv("PRICE_INPUT_PER_1M", "0.075"))
PRICE_OUTPUT_PER_1M = float(os.getenv("PRICE_OUTPUT_PER_1M", "0.30"))

MAX_HANDOFFS = int(os.getenv("MAX_HANDOFFS", "8"))
MAX_TOTAL_SECONDS = float(os.getenv("MAX_TOTAL_SECONDS", "120"))

PRICE_INPUT_PER_1M = float(os.getenv("PRICE_INPUT_PER_1M", "0.075"))
PRICE_OUTPUT_PER_1M = float(os.getenv("PRICE_OUTPUT_PER_1M", "0.30"))


# ---------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------

load_dotenv()

BASE_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = BASE_DIR.parent if BASE_DIR.name == "day2_agents" else BASE_DIR

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MIN_SECONDS_BETWEEN_MODEL_CALLS = float(
    os.getenv("MIN_SECONDS_BETWEEN_MODEL_CALLS", "15")
)

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "hclsw-gcp-wrkld-auto")
GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-east1")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "./service_account.json")
TOP_K = int(os.getenv("TOP_K", "3"))

ALLOW_FALLBACK_KB = (
    os.getenv("ALLOW_FALLBACK_KB", "true")
    .lower()
    .strip()
    in {"1", "true", "yes", "y"}
)

MAX_TOOL_CALLS = int(os.getenv("MAX_TOOL_CALLS", "8"))
MAX_TOTAL_SECONDS = float(os.getenv("MAX_TOTAL_SECONDS", "90"))

# ContextVar per lo streaming SSE delle trace MCP.
# Se impostato, append_trace chiama il callback in real-time verso mcp_api.py.
_trace_callback: ContextVar[Optional[Callable[[dict], None]]] = ContextVar(
    "_trace_callback", default=None
)


def get_vertex_llm() -> ChatVertexAI:
    """
    Crea un'istanza di ChatVertexAI autenticata tramite service account.

    ChatVertexAI supporta .bind_tools() esattamente come ChatGoogleGenerativeAI,
    quindi tool calling, LangGraph e il loop ReAct funzionano senza modifiche.

    L'autenticazione avviene tramite google-auth:
    - legge il file JSON dal path in GOOGLE_APPLICATION_CREDENTIALS;
    - lo passa come `credentials` a ChatVertexAI;
    - non serve GOOGLE_API_KEY.
    """
    from google.oauth2 import service_account as sa

    sa_path = Path(GOOGLE_APPLICATION_CREDENTIALS)
    if not sa_path.is_absolute():
        sa_path = BASE_DIR / sa_path

    if not sa_path.exists():
        raise FileNotFoundError(
            f"Service account non trovato: {sa_path}. "
            "Imposta GOOGLE_APPLICATION_CREDENTIALS nel .env."
        )

    credentials = sa.Credentials.from_service_account_file(
        str(sa_path),
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )

    return ChatVertexAI(
        model=GEMINI_MODEL,
        project=GOOGLE_CLOUD_PROJECT,
        location=GOOGLE_CLOUD_LOCATION,
        credentials=credentials,
        temperature=0,
    )


# ---------------------------------------------------------------------
# Import del dominio: i tool MCP wrappano queste funzioni
# ---------------------------------------------------------------------

DAY2_IMPORT_ERROR: Optional[Exception] = None

try:
    from day2_agents.itsm_agent import (
        compute_sla as domain_compute_sla,
        lookup_record as domain_lookup_record,
        search_kb as domain_search_kb,
    )
except Exception as exc:
    DAY2_IMPORT_ERROR = exc

    def domain_search_kb(query: str, top_k: int = 3) -> Dict[str, Any]:  # type: ignore
        return {
            "success": False,
            "error": (
                "day2_agents.itsm_agent non importabile. "
                f"Errore: {DAY2_IMPORT_ERROR}"
            ),
            "results": [],
        }

    def domain_lookup_record(record_id: str) -> Dict[str, Any]:  # type: ignore
        return {
            "success": False,
            "error": (
                "day2_agents.itsm_agent non importabile. "
                f"Errore: {DAY2_IMPORT_ERROR}"
            ),
            "results": [],
        }

    def domain_compute_sla(ticket: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
        return {
            "success": False,
            "error": (
                "day2_agents.itsm_agent non importabile. "
                f"Errore: {DAY2_IMPORT_ERROR}"
            ),
            "results": [],
        }


# ---------------------------------------------------------------------
# Modelli Pydantic e stato
# ---------------------------------------------------------------------

class MCPToolSpec(BaseModel):
    """Specifica di un tool MCP esposto dal server."""

    name: str
    description: str
    inputSchema: Dict[str, Any] = Field(default_factory=dict)


class MCPToolResult(BaseModel):
    """Risultato di una tool call MCP, struttura compatibile con il protocollo reale."""

    content: List[Dict[str, Any]] = Field(default_factory=list)
    isError: bool = False


class AgentState(TypedDict):
    """
    Stato del KnowledgeAgent MCP.

    Questo lab non è un secondo supervisor multi-agent: non mantiene routing,
    handoff o azioni critiche. Mantiene invece ciò che serve per MCP:
    messaggi, tool call, citazioni, token e trace.
    """

    messages: Annotated[list, add_messages]
    task_id: str

    tool_calls: List[dict]
    citations: List[dict]
    tokens_in: int
    tokens_out: int
    trace: List[dict]

    fast_mode: bool


def model_to_dict(model: BaseModel) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def to_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, default=str)


def print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2, default=str))


def extract_usage(message: Any) -> Dict[str, Any]:
    """Estrae usage da un messaggio LLM in modo robusto."""
    usage = getattr(message, "usage_metadata", None) or {}
    if not usage and hasattr(message, "response_metadata"):
        usage = (message.response_metadata or {}).get("usage_metadata") or {}
    return dict(usage) if usage else {}


def extract_text_content(content: Any) -> str:
    """
    Normalizza il contenuto restituito dai provider LLM.

    Alcuni provider/modelli restituiscono una stringa; altri restituiscono
    una lista di blocchi, per esempio:
        [{"type": "text", "text": "...", "extras": {...}}]

    Per il terminale e per il JSON didattico vogliamo mostrare solo il testo.
    """
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: List[str] = []

        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    parts.append(item["content"])
                else:
                    parts.append(json.dumps(item, ensure_ascii=False, default=str))
            else:
                parts.append(str(item))

        return "\n".join(part for part in parts if part).strip()

    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content["text"]
        if isinstance(content.get("content"), str):
            return content["content"]
        return json.dumps(content, ensure_ascii=False, default=str)

    return str(content)


def is_quota_error(exc: Exception) -> bool:
    """Riconosce errori di quota/rate limit del provider LLM."""
    text = str(exc).lower()
    return (
        "429" in text
        or "resource_exhausted" in text
        or "quota" in text
        or "rate limit" in text
    )


def rate_limit_sleep(last_call_ts: float) -> float:
    now = time.time()
    wait = MIN_SECONDS_BETWEEN_MODEL_CALLS - (now - last_call_ts)
    if wait > 0:
        time.sleep(wait)
    return time.time()


# def get_vertex_llm() -> ChatGoogleGenerativeAI:
#     if not os.getenv("GOOGLE_API_KEY"):
#         raise RuntimeError("GOOGLE_API_KEY non configurata")
#     return ChatGoogleGenerativeAI(
#         model=GEMINI_MODEL,
#         temperature=0,
#         timeout=25,
#     )


def append_trace(
    trace: List[dict],
    event: str,
    **payload: Any,
) -> None:
    event_dict = {
        "step": len(trace) + 1,
        "event": event,
        "timestamp": time.time(),
        **payload,
    }
    trace.append(event_dict)

    cb = _trace_callback.get()
    if cb is not None:
        cb(event_dict)


def serialize_message(message: Any) -> Dict[str, Any]:
    """Serializza messaggi LangChain in modo leggibile nel JSON di run."""
    return {
        "type": message.__class__.__name__,
        "name": getattr(message, "name", None),
        "content": extract_text_content(getattr(message, "content", str(message))),
        "tool_calls": getattr(message, "tool_calls", None),
        "usage_metadata": getattr(message, "usage_metadata", None),
        "response_metadata": getattr(message, "response_metadata", None),
    }


def state_to_serializable(state: Dict[str, Any]) -> Dict[str, Any]:
    """Converte lo stato in una struttura JSON-friendly."""
    data = dict(state)
    data["messages"] = [
        serialize_message(m) for m in state.get("messages", [])
    ]
    return data


def save_run_json(
    *,
    mode: str,
    query: str,
    answer: str,
    state: Dict[str, Any],
    trace_dir: Optional[str] = None,
) -> Path:
    """
    Salva il JSON completo della run.

    Default:
        runs/day3_mcp/

    Il terminale resta leggibile; il file JSON contiene tutto il necessario
    per debug, replay didattico, audit e calcolo costo.
    """
    output_dir = Path(trace_dir).expanduser().resolve() if trace_dir else (
        PROJECT_ROOT / "runs" / "day3_mcp"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    task_id = str(state.get("task_id", "no-task"))[:8]
    path = output_dir / f"{mode}_{timestamp}_{task_id}.json"

    tokens_in = int(state.get("tokens_in", 0) or 0)
    tokens_out = int(state.get("tokens_out", 0) or 0)

    payload = {
        "run": {
            "mode": mode,
            "query": query,
            "answer": answer,
            "saved_at": timestamp,
        },
        "summary": {
            "task_id": state.get("task_id"),
            "fast_mode": state.get("fast_mode", False),
            "counts": {
                "messages": len(state.get("messages", [])),
                "tool_calls": len(state.get("tool_calls", [])),
                "citations": len(state.get("citations", [])),
                "trace": len(state.get("trace", [])),
            },
            "token_usage": {
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "tokens_total": tokens_in + tokens_out,
            },
        },
        "state": state_to_serializable(state),
    }

    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    return path


def estimate_cost_from_run_json(path: str) -> Dict[str, Any]:
    """Calcola il costo stimato a partire da un JSON generato da save_run_json."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))

    summary = payload.get("summary", {})
    state = payload.get("state", {})
    token_usage = summary.get("token_usage", {})

    tokens_in = int(
        token_usage.get("tokens_in")
        or state.get("tokens_in")
        or 0
    )
    tokens_out = int(
        token_usage.get("tokens_out")
        or state.get("tokens_out")
        or 0
    )

    cost_in = tokens_in * PRICE_INPUT_PER_1M / 1_000_000
    cost_out = tokens_out * PRICE_OUTPUT_PER_1M / 1_000_000
    cost_total = cost_in + cost_out

    return {
        "file": path,
        "tokens": {
            "input": tokens_in,
            "output": tokens_out,
            "total": tokens_in + tokens_out,
        },
        "cost_usd": {
            "input": round(cost_in, 8),
            "output": round(cost_out, 8),
            "total": round(cost_total, 8),
            "for_10k_queries_day_per_month": round(cost_total * 10_000 * 30, 2),
        },
        "run_counts": {
            "tool_calls": len(state.get("tool_calls", [])),
            "citations": len(state.get("citations", [])),
            "trace": len(state.get("trace", [])),
        },
        "prices_per_1M": {
            "input": PRICE_INPUT_PER_1M,
            "output": PRICE_OUTPUT_PER_1M,
        },
    }


# ---------------------------------------------------------------------
# Tool del dominio — wrapper sottili sulle funzioni del Giorno 2
# ---------------------------------------------------------------------

def _handler_search_kb(query: str, top_k: int = 3) -> Dict[str, Any]:
    return domain_search_kb(query, top_k)


def _handler_lookup_record(record_id: str) -> Dict[str, Any]:
    return domain_lookup_record(record_id)


def _handler_compute_sla(
    id: str,
    priority: str,
    elapsed_hours: float,
    owner: str = "",
    service: str = "unknown",
    workaround_available: bool = False,
) -> Dict[str, Any]:
    # opened_at is derived from elapsed_hours so it is never exposed in the
    # MCP inputSchema — the caller only needs to pass how many hours have
    # passed; the handler reconstructs the absolute timestamp internally.
    opened_at = datetime.now() - timedelta(hours=elapsed_hours)
    return domain_compute_sla({
        "id": id,
        "key": id,
        "record_type": "incident",
        "summary": f"Ticket {id}",
        "description": f"Ticket {id}",
        "priority": priority,
        "status": "Open",
        "service": service,
        "environment": "production",
        "component": "unknown",
        "elapsed_hours": elapsed_hours,
        "owner": owner,
        "assignee": None,
        "reporter": "unknown",
        "affected_users": 0,
        "business_impact": "",
        "workaround_available": workaround_available,
        "labels": [],
        "linked_records": [],
        "comments": [],
        "opened_at": opened_at,
    })


DOMAIN_TOOLS: List[Dict[str, Any]] = [
    {
        "spec": MCPToolSpec(
            name="search_kb",
            description=(
                "Cerca nella knowledge base ITSM: policy, SLA, escalation, procedure, "
                "on-call, runbook e knowledge article. Usa per domande su regole operative."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Domanda dell'utente in linguaggio naturale.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Numero massimo di hit da restituire.",
                        "default": 3,
                        "minimum": 1,
                        "maximum": 10,
                    },
                },
                "required": ["query"],
            },
        ),
        "handler": _handler_search_kb,
    },
    {
        "spec": MCPToolSpec(
            name="lookup_record",
            description=(
                "Recupera un record operativo ITSM/Jira-like dato un ID, per esempio INC-1002. "
                "Restituisce priorità, stato, owner, servizio, ambiente e tempo trascorso."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "record_id": {
                        "type": "string",
                        "description": "Identificativo del record, es. INC-1002.",
                    }
                },
                "required": ["record_id"],
            },
        ),
        "handler": _handler_lookup_record,
    },
    {
        "spec": MCPToolSpec(
            name="compute_sla",
            description=(
                "Calcola stato SLA: ok, near_breach o violated, e produce una "
                "raccomandazione operativa a partire dai dati del ticket."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "ID ticket, es. INC-1002.",
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["P1", "P2", "P3", "P4"],
                        "description": "Priorità ITSM del ticket.",
                    },
                    "elapsed_hours": {
                        "type": "number",
                        "minimum": 0,
                        "description": "Ore trascorse dall'apertura.",
                    },
                    "owner": {
                        "type": "string",
                        "default": "",
                        "description": "Team owner responsabile.",
                    },
                    "service": {
                        "type": "string",
                        "default": "unknown",
                        "description": "Servizio impattato.",
                    },
                    "workaround_available": {
                        "type": "boolean",
                        "default": False,
                        "description": "True se è disponibile un workaround.",
                    },
                },
                "required": ["id", "priority", "elapsed_hours"],
            },
        ),
        "handler": _handler_compute_sla,
    },
]


# ---------------------------------------------------------------------
# MCP Adapter simulato — stessa interfaccia concettuale di un client MCP
# ---------------------------------------------------------------------

class MCPAdapter:
    """
    Adapter MCP simulato, in-process.

    Espone:
        list_tools() -> list[dict]
        get_tool(name) -> dict | None
        call_tool(name, **kwargs) -> dict

    Differenza rispetto a un server MCP reale:
    - non c'è JSON-RPC;
    - non c'è subprocess;
    - non c'è transport stdio/SSE/HTTP.

    Differenza vista dall'agent:
    - nessuna, se il contratto name/description/inputSchema è preservato.
    """

    def __init__(self, name: str = "hcl-itsm-adapter"):
        self.name = name
        self._tools: Dict[str, Dict[str, Any]] = {}

    def register(self, spec: MCPToolSpec, handler: Callable[..., Dict[str, Any]]) -> None:
        self._tools[spec.name] = {
            "spec": spec,
            "handler": handler,
        }

    def list_tools(self) -> List[Dict[str, Any]]:
        return [model_to_dict(entry["spec"]) for entry in self._tools.values()]

    def get_tool(self, name: str) -> Optional[Dict[str, Any]]:
        entry = self._tools.get(name)
        if not entry:
            return None
        return model_to_dict(entry["spec"])

    def call_tool(self, name: str, **kwargs: Any) -> Dict[str, Any]:
        entry = self._tools.get(name)
        if not entry:
            return model_to_dict(
                MCPToolResult(
                    content=[{"type": "text", "text": f"Tool '{name}' non registrato."}],
                    isError=True,
                )
            )

        try:
            result = entry["handler"](**kwargs)
            return model_to_dict(
                MCPToolResult(
                    content=[{"type": "text", "text": to_json(result)}],
                    isError=not bool(result.get("success", True)),
                )
            )
        except Exception as exc:
            return model_to_dict(
                MCPToolResult(
                    content=[{"type": "text", "text": f"Errore tool {name}: {exc}"}],
                    isError=True,
                )
            )


def build_adapter() -> MCPAdapter:
    """Costruisce un adapter pronto all'uso con i tool del dominio."""
    adapter = MCPAdapter()
    for entry in DOMAIN_TOOLS:
        adapter.register(entry["spec"], entry["handler"])
    return adapter


# ---------------------------------------------------------------------
# MCPHTTPAdapter — stesso contratto, tool su server HTTP remoto
# ---------------------------------------------------------------------

MCP_REMOTE_URL = os.getenv("MCP_REMOTE_URL", "http://localhost:8002")


class MCPHTTPAdapter:
    """
    Adapter MCP che parla con un server HTTP remoto.

    Espone la STESSA interfaccia di MCPAdapter:
        list_tools() -> list[dict]
        get_tool(name) -> dict | None
        call_tool(name, **kwargs) -> dict

    Punto didattico centrale:
        run_agent() riceve un oggetto con .list_tools()/.call_tool()
        e NON sa se i tool sono in-process (MCPAdapter) o su un server
        separato (MCPHTTPAdapter). Il codice dell'agent non cambia di una riga.

    Flusso reale con questo adapter:
        1. list_tools()  → GET  http://host:port/mcp/tools
        2. call_tool()   → POST http://host:port/mcp/tools/{name}/call
        3. get_tool()    → GET  http://host:port/mcp/tools/{name}

    In un sistema MCP vero il transport sarebbe JSON-RPC su stdio o HTTP;
    qui usiamo HTTP/REST per rendere il flusso ispezionabile nel browser.
    """

    def __init__(self, base_url: str = "http://localhost:8002"):
        self.base_url = base_url.rstrip("/")
        self.name = f"http-adapter({self.base_url})"

    def _get(self, path: str) -> Any:
        import urllib.request as _ur
        with _ur.urlopen(f"{self.base_url}{path}", timeout=15) as r:
            return json.loads(r.read().decode())

    def _post(self, path: str, payload: Dict[str, Any]) -> Any:
        import urllib.request as _ur
        body = json.dumps(payload, ensure_ascii=False, default=str).encode()
        req = _ur.Request(
            f"{self.base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with _ur.urlopen(req, timeout=15) as r:
            return json.loads(r.read().decode())

    def list_tools(self) -> List[Dict[str, Any]]:
        return self._get("/mcp/tools")

    def get_tool(self, name: str) -> Optional[Dict[str, Any]]:
        try:
            return self._get(f"/mcp/tools/{name}")
        except Exception:
            return None

    def call_tool(self, name: str, **kwargs: Any) -> Dict[str, Any]:
        try:
            resp = self._post(f"/mcp/tools/{name}/call", {"args": kwargs})
            # mcp_tool_server wrappa il risultato in {"tool":..., "mcp_result":...}
            return resp.get("mcp_result", resp)
        except Exception as exc:
            return model_to_dict(
                MCPToolResult(
                    content=[{"type": "text", "text": f"Errore HTTP {name}@{self.base_url}: {exc}"}],
                    isError=True,
                )
            )


# ---------------------------------------------------------------------
# FastMCP server reale — opzionale
# ---------------------------------------------------------------------

#def build_fastmcp_server() -> Any:
def build_fastmcp_server(
    host: str = "127.0.0.1",
    port: int = 8000,
) -> Any:    
    """
    Costruisce un server FastMCP reale.

    Nota importante:
    i nomi dei tool sono allineati all'adapter:
        search_kb
        lookup_record
        compute_sla

    Così inspector e adapter mostrano lo stesso contratto.
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except Exception as exc:
        raise RuntimeError(
            "FastMCP non è installato. Per il bonus avanzato installa: pip install mcp. "
            f"Dettaglio: {exc}"
        )

    #mcp = FastMCP("hcl-itsm-server")
    mcp = FastMCP(
        "hcl-itsm-server",
        host=host,
        port=port,
        streamable_http_path="/mcp",
    )

    @mcp.tool()
    def search_kb(query: str, top_k: int = 3) -> str:
        """Cerca nella knowledge base ITSM: policy, SLA, escalation e procedure."""
        return to_json(_handler_search_kb(query, top_k))

    @mcp.tool()
    def lookup_record(record_id: str) -> str:
        """Recupera un record operativo ITSM/Jira-like, per esempio INC-1002."""
        return to_json(_handler_lookup_record(record_id))

    @mcp.tool()
    def compute_sla(
        id: str,
        priority: str,
        elapsed_hours: float,
        owner: str = "",
        service: str = "unknown",
        workaround_available: bool = False,
    ) -> str:
        """Calcola stato SLA e raccomandazione operativa."""
        return to_json(_handler_compute_sla(
            id=id,
            priority=priority,
            elapsed_hours=elapsed_hours,
            owner=owner,
            service=service,
            workaround_available=workaround_available,
        ))

    return mcp


def run_fastmcp_server() -> None:
    """Avvia il server FastMCP su stdio."""
    server = build_fastmcp_server()
    server.run()

def run_fastmcp_http_server(
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    """
    Avvia il server FastMCP come server HTTP persistente.

    Questa modalità è diversa da stdio:
    - stdio: il client di solito avvia il server come subprocess;
    - HTTP: il server resta acceso su una porta e client separati possono collegarsi.

    Endpoint MCP atteso:
        http://127.0.0.1:8000/mcp
    """
    server = build_fastmcp_server(
        host=host,
        port=port,
    )

    server.run(
        transport="streamable-http",
    )


# ---------------------------------------------------------------------
# Client MCP reale via subprocess stdio — usato dall'inspector
# ---------------------------------------------------------------------

async def fastmcp_discovery(command: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Apre una sessione MCP su stdio verso il subprocess `command` e fa discovery.

    Restituisce:
        (tools, server_info)

    Solleva RuntimeError se la libreria mcp non è installata o se il subprocess
    non risponde correttamente.
    """
    try:
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client
    except Exception as exc:
        raise RuntimeError(
            "Pacchetto mcp non installato. Per il bonus avanzato: pip install mcp. "
            f"Dettaglio: {exc}"
        )

    argv = shlex.split(command)
    params = StdioServerParameters(command=argv[0], args=argv[1:])

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            tools_resp = await session.list_tools()
            tools = [
                {
                    "name": t.name,
                    "description": t.description,
                    "inputSchema": getattr(t, "inputSchema", None) or {},
                }
                for t in tools_resp.tools
            ]
            return tools, {"server": getattr(init, "serverInfo", None)}


async def http_mcp_call_tool(
    url: str,
    tool_name: str,
    args: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Client MCP reale via Streamable HTTP.

    Non avvia il server.
    Si collega a un server MCP già acceso, per esempio:
        http://127.0.0.1:8000/mcp

    Flusso:
        client HTTP -> initialize -> list_tools -> call_tool
    """
    try:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client
    except Exception as exc:
        raise RuntimeError(
            "Pacchetto mcp non installato o client HTTP non disponibile. "
            "Installa/aggiorna con: pip install -U 'mcp[cli]'. "
            f"Dettaglio: {exc}"
        )

    async with streamable_http_client(url) as (
        read_stream,
        write_stream,
        _,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            init = await session.initialize()

            tools_resp = await session.list_tools()
            available_tools = [tool.name for tool in tools_resp.tools]

            if tool_name not in available_tools:
                return {
                    "url": url,
                    "server_info": str(getattr(init, "serverInfo", None)),
                    "available_tools": available_tools,
                    "tool": tool_name,
                    "args": args,
                    "error": f"Tool '{tool_name}' non esposto dal server MCP.",
                }

            result = await session.call_tool(tool_name, args)

            content = []
            for item in getattr(result, "content", []) or []:
                content.append({
                    "type": getattr(item, "type", None),
                    "text": getattr(item, "text", None),
                })

            return {
                "url": url,
                "server_info": str(getattr(init, "serverInfo", None)),
                "available_tools": available_tools,
                "tool": tool_name,
                "args": args,
                "mcp_result": {
                    "content": content,
                    "isError": getattr(result, "isError", False),
                },
            }

async def fastmcp_call_tool(
    command: str,
    tool_name: str,
    args: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Apre una sessione MCP reale su stdio verso il subprocess `command`
    e chiama un tool esposto dal server FastMCP.

    Questa è una chiamata MCP reale:
        Host/Client MCP -> subprocess stdio -> Server FastMCP -> tool
    """
    try:
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client
    except Exception as exc:
        raise RuntimeError(
            "Pacchetto mcp non installato. Installa con: pip install mcp. "
            f"Dettaglio: {exc}"
        )

    argv = shlex.split(command)
    params = StdioServerParameters(command=argv[0], args=argv[1:])

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()

            # Discovery reale: utile per verificare che il tool esista.
            tools_resp = await session.list_tools()
            available_tools = [tool.name for tool in tools_resp.tools]

            if tool_name not in available_tools:
                return {
                    "server_info": str(getattr(init, "serverInfo", None)),
                    "available_tools": available_tools,
                    "tool": tool_name,
                    "args": args,
                    "error": f"Tool '{tool_name}' non esposto dal server MCP.",
                }

            # Chiamata MCP reale al tool.
            result = await session.call_tool(tool_name, args)

            content = []
            for item in getattr(result, "content", []) or []:
                content.append({
                    "type": getattr(item, "type", None),
                    "text": getattr(item, "text", None),
                })

            return {
                "server_info": str(getattr(init, "serverInfo", None)),
                "available_tools": available_tools,
                "tool": tool_name,
                "args": args,
                "mcp_result": {
                    "content": content,
                    "isError": getattr(result, "isError", False),
                },
            }

# ---------------------------------------------------------------------
# Wrapping dei tool MCP come BaseTool LangChain
# ---------------------------------------------------------------------

def pydantic_model_from_json_schema(tool_name: str, schema: Dict[str, Any]) -> type[BaseModel]:
    """
    Conversione minimale JSON Schema -> Pydantic model.

    Sufficiente per il laboratorio:
    - string
    - integer
    - number
    - boolean

    Obiettivo didattico:
    LangChain deve vedere uno schema argomenti preciso, non un generico **kwargs.
    """
    properties = schema.get("properties", {}) or {}
    required = set(schema.get("required", []) or [])

    fields: Dict[str, Any] = {}

    type_map = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
    }

    for field_name, spec in properties.items():
        json_type = spec.get("type", "string")
        py_type = type_map.get(json_type, Any)

        description = spec.get("description", "")
        default = ... if field_name in required else spec.get("default", None)

        fields[field_name] = (
            py_type,
            Field(default, description=description),
        )

    if not fields:
        fields["_empty"] = (Optional[str], Field(None, description="Tool senza argomenti."))

    safe_name = re.sub(r"\W+", "_", tool_name).strip("_") or "Tool"
    return create_model(f"{safe_name}_Args", **fields)


def make_langchain_tools(adapter: MCPAdapter) -> List[StructuredTool]:
    """
    Crea StructuredTool LangChain a partire dai tool MCP esposti dall'adapter.

    Mapping:
        MCP name        -> LangChain tool name
        MCP description -> LangChain tool description
        MCP inputSchema -> Pydantic args_schema

    Questo preserva il punto centrale del lab: lo schema MCP diventa lo schema
    che il modello vede quando deve decidere una tool call.
    """
    tools: List[StructuredTool] = []

    for spec in adapter.list_tools():
        name = spec["name"]
        description = spec["description"]
        input_schema = spec.get("inputSchema", {}) or {}
        args_schema = pydantic_model_from_json_schema(name, input_schema)

        def _make_runner(_name: str = name, _adapter: MCPAdapter = adapter):
            def _runner(**kwargs: Any) -> str:
                result = _adapter.call_tool(_name, **kwargs)
                return to_json(result)

            return _runner

        tools.append(
            StructuredTool.from_function(
                func=_make_runner(),
                name=name,
                description=description,
                args_schema=args_schema,
            )
        )

    return tools


# ---------------------------------------------------------------------
# Agent KnowledgeAgent-like, alimentato dai tool MCP
# ---------------------------------------------------------------------

KNOWLEDGE_PROMPT = """
Sei il KnowledgeAgent del laboratorio MCP.

Compito:
- rispondere alla richiesta dell'utente usando SOLO i tool esposti dal server MCP collegato.

Regole:
1. Usa search_kb per policy, SLA, regole operative, on-call e procedure.
2. Usa lookup_record per recuperare ticket per id.
3. Usa compute_sla per valutare violazione SLA dopo avere recuperato i dati del ticket.
4. Per ogni informazione documentale, indica la fonte se disponibile.
5. Se un tool fallisce, dichiaralo chiaramente: non inventare ticket, policy o SLA.
6. Termina con una sintesi in italiano di 5-8 righe, con citazioni se disponibili.
""".strip()


TICKET_ID_RE = re.compile(r"\bINC-\d+\b", flags=re.IGNORECASE)


def extract_ticket_id(text: str) -> Optional[str]:
    """Estrae il primo ID ticket ITSM dal testo, se presente."""
    match = TICKET_ID_RE.search(text or "")
    return match.group(0).upper() if match else None


def should_search_policy(text: str) -> bool:
    lowered = text.lower()
    keywords = [
        "policy",
        "sla",
        "escalation",
        "p1",
        "p2",
        "procedura",
        "regola",
        "on-call",
        "runbook",
        "knowledge",
    ]
    return any(k in lowered for k in keywords)


def extract_domain_results(mcp_result: Dict[str, Any]) -> List[dict]:
    """
    Estrae results dal payload MCP-like:
        {
          "content": [
            {"type": "text", "text": "{... JSON dominio ...}"}
          ],
          "isError": false
        }
    """
    results: List[dict] = []

    for chunk in mcp_result.get("content", []) or []:
        if not isinstance(chunk, dict):
            continue

        text = chunk.get("text", "")
        if not text:
            continue

        try:
            inner = json.loads(text)
        except Exception:
            continue

        for item in inner.get("results", []) or []:
            if isinstance(item, dict):
                results.append(item)

    return results


def extract_first_domain_result(mcp_result: Dict[str, Any]) -> Optional[dict]:
    results = extract_domain_results(mcp_result)
    return results[0] if results else None


def extract_citations_from_mcp_result(mcp_result: Dict[str, Any]) -> List[dict]:
    """Estrazione best-effort delle citazioni da un risultato MCP."""
    citations: List[dict] = []

    for hit in extract_domain_results(mcp_result):
        if hit.get("source"):
            citations.append({
                "source": hit["source"],
                "snippet": (hit.get("snippet") or "")[:240],
            })

    return citations


def run_agent_fast(query: str, adapter: Optional[MCPAdapter] = None) -> Tuple[str, AgentState]:
    """
    Esecuzione deterministica del lab MCP.

    Non usa LLM. Serve per:
    - demo con quota LLM esaurita;
    - mostrare list_tools/call_tool in modo esplicito;
    - dimostrare che MCP è indipendente dal modello.
    """
    if adapter is None:
        adapter = build_adapter()

    state: AgentState = {
        "messages": [HumanMessage(content=query)],
        "task_id": str(uuid.uuid4()),
        "tool_calls": [],
        "citations": [],
        "tokens_in": 0,
        "tokens_out": 0,
        "trace": [],
        "fast_mode": True,
    }

    lines: List[str] = [
        "Esecuzione MCP in modalità fast/deterministica.",
        "",
    ]

    ticket_id = extract_ticket_id(query)

    ticket: Optional[dict] = None
    sla: Optional[dict] = None

    append_trace(
        state["trace"],
        "fast_start",
        query=query,
        tools=[tool["name"] for tool in adapter.list_tools()],
    )

    if ticket_id:
        args = {"record_id": ticket_id}
        result = adapter.call_tool("lookup_record", **args)

        state["tool_calls"].append({
            "name": "lookup_record",
            "args": args,
        })

        append_trace(
            state["trace"],
            "fast_tool_call",
            tool="lookup_record",
            args=args,
            result=result,
        )

        ticket = extract_first_domain_result(result)

        if ticket:
            lines.append(f"Record recuperato: {ticket_id}.")
            lines.append(f"- Priorità: {ticket.get('priority', 'n/d')}")
            lines.append(f"- Stato: {ticket.get('status', 'n/d')}")
            lines.append(f"- Owner: {ticket.get('owner', 'n/d')}")
            lines.append(f"- Servizio: {ticket.get('service', 'n/d')}")
        else:
            lines.append(f"Record {ticket_id} non trovato. Non invento dati.")

    if ticket:
        args = {
            "id": ticket.get("id") or ticket.get("key"),
            "priority": ticket.get("priority"),
            "elapsed_hours": ticket.get("elapsed_hours", 0),
            "owner": ticket.get("owner", ""),
            "service": ticket.get("service", "unknown"),
            "workaround_available": ticket.get("workaround_available", False),
        }

        result = adapter.call_tool("compute_sla", **args)

        state["tool_calls"].append({
            "name": "compute_sla",
            "args": args,
        })

        append_trace(
            state["trace"],
            "fast_tool_call",
            tool="compute_sla",
            args=args,
            result=result,
        )

        sla = extract_first_domain_result(result)

        if sla:
            lines.extend([
                "",
                "Valutazione SLA:",
                f"- Stato SLA: {sla.get('status', 'n/d')}",
                f"- Soglia: {sla.get('threshold_hours', 'n/d')}h",
                f"- Tempo trascorso: {sla.get('elapsed_hours', 'n/d')}h",
                f"- Ore residue: {sla.get('remaining_hours', 'n/d')}h",
                f"- Raccomandazione: {sla.get('recommendation', 'n/d')}",
            ])

            reason = sla.get("reason")
            if reason:
                lines.append(f"- Motivazione: {reason}")

    if should_search_policy(query):
        args = {"query": query, "top_k": 3}
        result = adapter.call_tool("search_kb", **args)

        state["tool_calls"].append({
            "name": "search_kb",
            "args": args,
        })

        append_trace(
            state["trace"],
            "fast_tool_call",
            tool="search_kb",
            args=args,
            result=result,
        )

        citations = extract_citations_from_mcp_result(result)
        state["citations"].extend(citations)

        if citations:
            lines.extend([
                "",
                "Evidenze documentali:",
            ])

            for citation in citations:
                source = citation.get("source", "fonte sconosciuta")
                snippet = citation.get("snippet", "")
                lines.append(f"- {source}: {snippet}")

    if not ticket_id and not should_search_policy(query):
        lines.append(
            "La richiesta non contiene un ID ticket né parole chiave documentali. "
            "In modalità fast non chiamo tool inutili."
        )

    lines.extend([
        "",
        "Nota: questa modalità non usa il modello LLM; serve come fallback didattico e operativo.",
    ])

    answer = "\n".join(lines)

    state["messages"].append(AIMessage(content=answer, name="mcp_agent_fast"))

    append_trace(
        state["trace"],
        "fast_final_answer",
        text=answer,
    )

    return answer, state


def run_agent(query: str, adapter: Optional[MCPAdapter] = None) -> Tuple[str, AgentState]:
    """
    Esegue un KnowledgeAgent ReAct alimentato dai tool MCP.

    LOOP REACT:
        1. LLM legge query + tool disponibili.
        2. LLM decide se rispondere o chiamare uno o più tool.
        3. Ogni tool call passa attraverso adapter.call_tool().
        4. Il risultato torna come ToolMessage.
        5. Si ripete finché l'LLM smette di chiamare tool.

    Se il provider LLM non è disponibile o va in quota, viene attivato il
    fallback deterministico run_agent_fast().
    """
    if adapter is None:
        adapter = build_adapter()

    tools = make_langchain_tools(adapter)
    tool_map = {t.name: t for t in tools}

    state: AgentState = {
        "messages": [
            SystemMessage(content=KNOWLEDGE_PROMPT),
            HumanMessage(content=query),
        ],
        "task_id": str(uuid.uuid4()),
        "tool_calls": [],
        "citations": [],
        "tokens_in": 0,
        "tokens_out": 0,
        "trace": [],
        "fast_mode": False,
    }

    try:
        llm = get_vertex_llm().bind_tools(tools)
    except Exception as exc:
        append_trace(
            state["trace"],
            "llm_initialization_error",
            error=str(exc),
            quota_like_error=is_quota_error(exc),
        )

        answer, fast_state = run_agent_fast(query, adapter=adapter)
        state["fast_mode"] = True
        state["tool_calls"].extend(fast_state["tool_calls"])
        state["citations"].extend(fast_state["citations"])
        state["trace"].extend(fast_state["trace"])
        state["messages"].append(AIMessage(content=answer, name="mcp_agent_fast"))
        return answer, state

    last_call = 0.0
    start = time.time()

    append_trace(
        state["trace"],
        "agent_start",
        query=query,
        tools=[tool.name for tool in tools],
    )

    for step in range(1, MAX_AGENT_ITERATIONS + 1):
        if time.time() - start > MAX_TOTAL_SECONDS:
            append_trace(
                state["trace"],
                "stop",
                error=f"Timeout {MAX_TOTAL_SECONDS}s superato",
            )
            return "Timeout agente raggiunto", state

        try:
            last_call = rate_limit_sleep(last_call)
            response = llm.invoke(state["messages"])
        except Exception as exc:
            append_trace(
                state["trace"],
                "llm_error",
                step_index=step,
                error=str(exc),
                quota_like_error=is_quota_error(exc),
            )

            answer, fast_state = run_agent_fast(query, adapter=adapter)
            state["fast_mode"] = True
            state["tool_calls"].extend(fast_state["tool_calls"])
            state["citations"].extend(fast_state["citations"])
            state["trace"].extend(fast_state["trace"])
            state["messages"].append(AIMessage(content=answer, name="mcp_agent_fast"))
            return answer, state

        state["messages"].append(response)

        usage = extract_usage(response)
        state["tokens_in"] += int(usage.get("input_tokens", 0) or 0)
        state["tokens_out"] += int(usage.get("output_tokens", 0) or 0)

        tool_calls = getattr(response, "tool_calls", None) or []
        text = extract_text_content(response.content)

        append_trace(
            state["trace"],
            "llm_response",
            step_index=step,
            text=text if text else "[tool-call only]",
            tool_calls=[
                {
                    "name": tc.get("name"),
                    "args": tc.get("args", {}),
                }
                for tc in tool_calls
            ],
            usage=usage,
        )

        if not tool_calls:
            return text or "Nessuna risposta testuale", state

        for tc in tool_calls:
            name = tc.get("name", "")
            args = tc.get("args", {}) or {}
            state["tool_calls"].append({"name": name, "args": args})

            if name not in tool_map:
                raw = to_json({
                    "content": [{"type": "text", "text": f"Tool '{name}' non disponibile."}],
                    "isError": True,
                })
            else:
                try:
                    raw = tool_map[name].invoke(args)
                except Exception as exc:
                    raw = to_json({
                        "content": [{"type": "text", "text": f"Errore tool {name}: {exc}"}],
                        "isError": True,
                    })

            try:
                parsed = json.loads(raw)
            except Exception:
                parsed = {"content": [{"type": "text", "text": raw}], "isError": False}

            state["citations"].extend(extract_citations_from_mcp_result(parsed))

            append_trace(
                state["trace"],
                "tool_result",
                tool=name,
                args=args,
                result=parsed,
            )

            state["messages"].append(
                ToolMessage(
                    content=raw,
                    name=name,
                    tool_call_id=tc.get("id", f"mcp-{step}-{name}"),
                )
            )

    append_trace(
        state["trace"],
        "stop",
        error=f"MAX_AGENT_ITERATIONS ({MAX_AGENT_ITERATIONS}) raggiunto",
    )
    return "Max iterations raggiunte", state


# ---------------------------------------------------------------------
# Selezione del backend MCP
# ---------------------------------------------------------------------

def build_mcp_client(remote_url: Optional[str] = None) -> Any:
    """
    Factory che restituisce un adapter compatibile con MCPAdapter.

    Casi d'uso:
        remote_url fornito          → MCPHTTPAdapter (tool su server HTTP)
        MCP_BACKEND=http nel .env   → MCPHTTPAdapter su MCP_REMOTE_URL
        MCP_BACKEND=adapter (def.)  → MCPAdapter in-process
        MCP_BACKEND=fastmcp         → MCPAdapter in-process (fallback lab)

    Il punto: run_agent() non sa quale adapter sta usando.
    """
    if remote_url:
        return MCPHTTPAdapter(remote_url)
    if MCP_BACKEND == "http":
        return MCPHTTPAdapter(MCP_REMOTE_URL)
    if MCP_BACKEND == "fastmcp":
        return build_adapter()
    return build_adapter()


# ---------------------------------------------------------------------
# Prompt di esempio
# ---------------------------------------------------------------------

EXAMPLE_PROMPTS: List[Dict[str, str]] = [
    {
        "title": "1. Discovery: cosa espone il server?",
        "peculiarity": "Mostra come list_tools() funziona dal lato client.",
        "prompt": "python day3_multiagent/mcp_lab.py list-tools",
    },
    {
        "title": "2. Schema di un tool MCP",
        "peculiarity": "Mostra name, description e inputSchema del tool.",
        "prompt": "python day3_multiagent/mcp_lab.py describe search_kb",
    },
    {
        "title": "3. Chiamata diretta a un tool MCP",
        "peculiarity": "Bypassa l'LLM e chiama direttamente adapter.call_tool().",
        "prompt": (
            "python day3_multiagent/mcp_lab.py call search_kb "
            "'{\"query\":\"P1 escalation\",\"top_k\":2}'"
        ),
    },
    {
        "title": "4. Agent + MCP: investigazione",
        "peculiarity": "L'agent decide quali tool MCP chiamare e li compone.",
        "prompt": (
            "python day3_multiagent/mcp_lab.py agent "
            "\"Mostrami INC-1002 e cita la policy di escalation P1.\""
        ),
    },
    {
        "title": "5. Agent + MCP senza LLM",
        "peculiarity": "Modalità fallback: nessuna chiamata LLM, solo tool MCP deterministici.",
        "prompt": (
            "python day3_multiagent/mcp_lab.py agent "
            "\"Mostrami INC-1002 e cita la policy di escalation P1.\" --fast"
        ),
    },
    {
        "title": "6. Cost tracking da run salvata",
        "peculiarity": "Calcola costo stimato a partire dal JSON completo della run.",
        "prompt": (
            "python day3_multiagent/mcp_lab.py cost-from-run "
            "runs/day3_mcp/agent_YYYYMMDD_HHMMSS_xxxxxxxx.json"
        ),
    },
    {
        "title": "7. Inspector FastMCP reale",
        "peculiarity": "Apre subprocess con FastMCP, initialize + tools/list. Richiede pip install mcp.",
        "prompt": "python day3_multiagent/mcp_lab.py inspector",
    },
]


# ---------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------

def cmd_examples(_: argparse.Namespace) -> None:
    for item in EXAMPLE_PROMPTS:
        print("=" * 100)
        print(item["title"])
        print(f"Peculiarità: {item['peculiarity']}")
        print(f"Prompt: {item['prompt']}")


def cmd_list_tools(_: argparse.Namespace) -> None:
    adapter = build_mcp_client()
    print_json(adapter.list_tools())


def cmd_describe(args: argparse.Namespace) -> None:
    adapter = build_mcp_client()
    spec = adapter.get_tool(args.name)

    if not spec:
        print_json({"error": f"Tool '{args.name}' non trovato."})
        sys.exit(2)

    print_json(spec)


def cmd_call(args: argparse.Namespace) -> None:
    adapter = build_mcp_client()

    try:
        params = json.loads(args.args_json or "{}")
    except json.JSONDecodeError as exc:
        print_json({"error": f"JSON argomenti non valido: {exc}"})
        sys.exit(2)

    if not isinstance(params, dict):
        print_json({"error": "Gli argomenti devono essere un oggetto JSON."})
        sys.exit(2)

    result = adapter.call_tool(args.name, **params)

    print_json({
        "tool": args.name,
        "args": params,
        "mcp_result": result,
    })


def print_compact_tool_calls(tool_calls: List[dict]) -> None:
    if not tool_calls:
        print("Nessuna tool call registrata.")
        return

    for index, tc in enumerate(tool_calls, start=1):
        print(f"{index}. {tc.get('name')}({to_json(tc.get('args') or {})})")


def cmd_agent(args: argparse.Namespace) -> None:
    adapter = build_mcp_client()

    if args.fast:
        answer, state = run_agent_fast(args.query, adapter=adapter)
    else:
        answer, state = run_agent(args.query, adapter=adapter)

    run_path = save_run_json(
        mode="agent_fast" if args.fast else "agent",
        query=args.query,
        answer=answer,
        state=state,
        trace_dir=args.trace_dir,
    )

    print("\nANSWER")
    print("=" * 100)
    print(answer)

    print("\nSUMMARY")
    print("=" * 100)
    print_json({
        "task_id": state.get("task_id"),
        "fast_mode": state.get("fast_mode", False),
        "tool_calls": len(state.get("tool_calls", [])),
        "citations": len(state.get("citations", [])),
        "trace": len(state.get("trace", [])),
        "tokens_in": state.get("tokens_in", 0),
        "tokens_out": state.get("tokens_out", 0),
        "trace_file": str(run_path),
    })

    print("\nTOOL CALLS COMPACT")
    print("=" * 100)
    print_compact_tool_calls(state.get("tool_calls", []))

    if args.verbose:
        print("\nTOOL CALLS")
        print("=" * 100)
        print_json(state.get("tool_calls", []))

        print("\nCITATIONS")
        print("=" * 100)
        print_json(state.get("citations", []))

        print("\nTRACE")
        print("=" * 100)
        print_json(state.get("trace", []))


def cmd_cost_from_run(args: argparse.Namespace) -> None:
    print_json(estimate_cost_from_run_json(args.path))


def cmd_serve(_: argparse.Namespace) -> None:
    """Avvia il server FastMCP reale su stdio."""
    run_fastmcp_server()


def cmd_serve_http(args: argparse.Namespace) -> None:
    """
    Avvia il server FastMCP reale su Streamable HTTP.

    Questa è la modalità giusta per demo con:
        Terminale 1: server acceso
        Terminale 2: client che si collega al server già acceso
    """
    print(
        f"Avvio MCP server HTTP su http://{args.host}:{args.port}/mcp"
    )
    run_fastmcp_http_server(
        host=args.host,
        port=args.port,
    )

def cmd_inspector(args: argparse.Namespace) -> None:
    """
    Inspector minimale: apre subprocess `MCP_SERVER_CMD` come server stdio
    e chiama initialize + list_tools.
    """
    command = args.command or MCP_SERVER_CMD
    print(f"Inspector → comando server: {command}")

    try:
        tools, info = asyncio.run(fastmcp_discovery(command))
    except RuntimeError as exc:
        print_json({"error": str(exc)})
        sys.exit(2)

    print_json({"server_info": info, "tools": tools})

def cmd_http_call(args: argparse.Namespace) -> None:
    """
    Client MCP HTTP reale.

    Si collega a un server MCP già acceso.
    Non avvia il server.
    """
    try:
        params = json.loads(args.args_json or "{}")
    except json.JSONDecodeError as exc:
        print_json({"error": f"JSON argomenti non valido: {exc}"})
        sys.exit(2)

    if not isinstance(params, dict):
        print_json({"error": "Gli argomenti devono essere un oggetto JSON."})
        sys.exit(2)

    try:
        result = asyncio.run(
            http_mcp_call_tool(
                url=args.url,
                tool_name=args.name,
                args=params,
            )
        )
    except RuntimeError as exc:
        print_json({"error": str(exc)})
        sys.exit(2)
    except Exception as exc:
        print_json({
            "error": (
                "Chiamata HTTP MCP fallita. "
                "Verifica che il server sia acceso e che l'URL sia corretto."
            ),
            "detail": str(exc),
            "url": args.url,
        })
        sys.exit(2)

    print_json(result)

def cmd_inspector_call(args: argparse.Namespace) -> None:
    """
    Inspector avanzato: apre subprocess `MCP_SERVER_CMD` come server stdio
    e chiama realmente un tool via MCP session.call_tool().
    """
    command = args.command or MCP_SERVER_CMD
    print(f"Inspector-call → comando server: {command}")

    try:
        params = json.loads(args.args_json or "{}")
    except json.JSONDecodeError as exc:
        print_json({"error": f"JSON argomenti non valido: {exc}"})
        sys.exit(2)

    if not isinstance(params, dict):
        print_json({"error": "Gli argomenti devono essere un oggetto JSON."})
        sys.exit(2)

    try:
        result = asyncio.run(
            fastmcp_call_tool(
                command=command,
                tool_name=args.name,
                args=params,
            )
        )
    except RuntimeError as exc:
        print_json({"error": str(exc)})
        sys.exit(2)

    print_json(result)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Esercitazione Giorno 3 — pomeriggio: laboratorio MCP. "
            "Espone tool ITSM tramite adapter simulato o server FastMCP reale, "
            "integra il client nell'agent LangChain e salva trace/costi."
        )
    )

    subparsers = parser.add_subparsers(required=True)

    examples_parser = subparsers.add_parser("examples")
    examples_parser.set_defaults(func=cmd_examples)

    list_parser = subparsers.add_parser("list-tools")
    list_parser.set_defaults(func=cmd_list_tools)

    describe_parser = subparsers.add_parser("describe")
    describe_parser.add_argument("name", type=str)
    describe_parser.set_defaults(func=cmd_describe)

    call_parser = subparsers.add_parser("call")
    call_parser.add_argument("name", type=str)
    call_parser.add_argument("args_json", type=str, default="{}", nargs="?")
    call_parser.set_defaults(func=cmd_call)

    agent_parser = subparsers.add_parser("agent")
    agent_parser.add_argument("query", type=str)
    agent_parser.add_argument(
        "--fast",
        action="store_true",
        help="Esegue il lab MCP senza LLM, usando chiamate deterministiche ai tool.",
    )
    agent_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Stampa tool calls, citazioni e trace completi a terminale.",
    )
    agent_parser.add_argument(
        "--trace-dir",
        type=str,
        default=None,
        help="Cartella dove salvare il JSON completo della run. Default: runs/day3_mcp/",
    )
    agent_parser.set_defaults(func=cmd_agent)

    cost_run_parser = subparsers.add_parser("cost-from-run")
    cost_run_parser.add_argument("path", type=str)
    cost_run_parser.set_defaults(func=cmd_cost_from_run)

    serve_parser = subparsers.add_parser("serve")
    serve_parser.set_defaults(func=cmd_serve)

    serve_http_parser = subparsers.add_parser("serve-http")
    serve_http_parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host su cui esporre il server HTTP MCP. Default: 127.0.0.1.",
    )
    serve_http_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Porta HTTP del server MCP. Default: 8000.",
    )
    serve_http_parser.set_defaults(func=cmd_serve_http)

    http_call_parser = subparsers.add_parser("http-call")
    http_call_parser.add_argument("name", type=str)
    http_call_parser.add_argument("args_json", type=str, default="{}", nargs="?")
    http_call_parser.add_argument(
        "--url",
        type=str,
        default="http://127.0.0.1:8000/mcp",
        help="URL del server MCP HTTP. Default: http://127.0.0.1:8000/mcp.",
    )
    http_call_parser.set_defaults(func=cmd_http_call)

    inspector_parser = subparsers.add_parser("inspector")
    inspector_parser.add_argument(
        "--command",
        type=str,
        default=None,
        help="Comando del server MCP da avviare. Default: MCP_SERVER_CMD.",
    )
    inspector_parser.set_defaults(func=cmd_inspector)

    inspector_call_parser = subparsers.add_parser("inspector-call")
    inspector_call_parser.add_argument("name", type=str)
    inspector_call_parser.add_argument("args_json", type=str, default="{}", nargs="?")
    inspector_call_parser.add_argument(
        "--command",
        type=str,
        default=None,
        help="Comando del server MCP da avviare. Default: MCP_SERVER_CMD.",
    )
    inspector_call_parser.set_defaults(func=cmd_inspector_call)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()