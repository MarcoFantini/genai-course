from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict, Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_vertexai import ChatVertexAI
from contextvars import ContextVar
from pydantic import BaseModel, Field
from typing_extensions import Annotated

try:
    from langgraph.graph.message import add_messages
except Exception:
    # Fallback per permettere l'import anche senza LangGraph installato.
    def add_messages(x: Any) -> Any:  # type: ignore
        return x


"""
day3_multiagent/supervisor.py

Esercitazione Giorno 3 — Mattino:
- decomposizione del caso d'uso ITSM in 3 agenti specializzati
- supervisor LangGraph che fa routing condizionale tra gli agenti
- stato condiviso tra agenti con TypedDict + reducer
- principio del minimo privilegio: ogni agent ha SOLO i tool che gli servono
- logging strutturato di ogni handoff (chi delega a chi, perché, con quali token)
- calcolo costo di una conversazione multi-agent

Architettura:

        START
          |
          v
       supervisor
       |    |    |
      v     v     v
  triage  know-   action
  Agent   ledge   Agent
            Agent
       \    |    /
        v   v   v
       supervisor (rievaluation loop)
          |
          v   (quando supervisor decide "end")
         END

Agenti:
    TriageAgent     classifica priorità, dominio e prossimo step
    KnowledgeAgent  cerca policy / SLA / procedure (RAG del Giorno 1)
    ActionAgent     esegue azioni (idempotenti) sul mondo esterno simulato

Prima di usare il KnowledgeAgent (RAG reale):
    python day1_morning_rag/main.py setup-data
    python day1_morning_rag/main.py ingest

Modalità:
    manual  -> loop "supervisor + agenti" scritto a mano, utile per capire cosa
               succede sotto LangGraph
    graph   -> versione LangGraph con stato condiviso, checkpoint in memoria e
               routing condizionale sul campo state['next']

Comandi:
    python day3_multiagent/supervisor.py examples
    python day3_multiagent/supervisor.py manual "Mostrami INC-1002 e proponi azione"
    (opzionale) python day3_multiagent/supervisor.py manual "Mostrami INC-1002 e proponi azione" --verbose
    python day3_multiagent/supervisor.py graph "Mostrami INC-1002 e proponi azione"
    python day3_multiagent/supervisor.py graph "Mostrami INC-1002 e proponi azione" --fast
    (stima) python day3_multiagent/supervisor.py cost-estimate --rounds 5
    (costo reale - sostituire JSON) python day3_multiagent/supervisor.py cost-from-run runs/day3_multiagent/graph_20260512_235914_f8fcaca5.json

Variabili .env:
    GOOGLE_API_KEY=...
    GEMINI_MODEL=gemini-2.5-flash
    MIN_SECONDS_BETWEEN_MODEL_CALLS=15
    MAX_HANDOFFS=8
    MAX_TOTAL_SECONDS=120
    PRICE_INPUT_PER_1M=0.10 # gemini-2.5-flash-lite
    PRICE_OUTPUT_PER_1M=0.40 # gemini-2.5-flash-lite

Nota didattica:
    Il supervisor non risolve direttamente: legge lo stato, decide chi prende
    il turno e inserisce il nome del prossimo nodo in state["next"]. Il grafo
    legge state["next"] e instrada il flusso. È un router con memoria.
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

# ContextVar per lo streaming SSE delle trace.
# Se impostato, append_trace chiama il callback ogni volta che genera un evento,
# permettendo ad api.py di ricevere gli eventi in real-time senza modificare
# nessuna firma di funzione esistente.
_trace_callback: ContextVar[Optional[Callable[[dict], None]]] = ContextVar(
    "_trace_callback", default=None
)

# ContextVar per il Human-in-the-Loop (HITL).
# Se impostato, supervisor_node chiama il callback quando ci sono azioni
# pending_approval e attende la decisione umana prima di produrre la risposta.
# Il callback riceve un dict {task_id, actions} e ritorna True (approvato)
# o False (rifiutato). In CLI il callback non è impostato: default sicuro = False.
_hitl_callback: ContextVar[Optional[Callable[[dict], bool]]] = ContextVar(
    "_hitl_callback", default=None
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
# Import dei tool del Giorno 2
# ---------------------------------------------------------------------
# Riusiamo i tool di day2_agents.itsm_agent: in produzione esisterebbe un
# package "domain.tools" o equivalente. In questo lab li importiamo per
# evitare duplicazione e per mostrare che gli agent specializzati possono
# riusare lo stesso tool registry — basta selezionare un sottoinsieme diverso.

DAY2_IMPORT_ERROR: Optional[Exception] = None

try:
    from day2_agents.itsm_agent import (
        SAMPLE_RECORDS,
        build_pending_action,
        compute_sla,
        execute_critical_action,
        lookup_record,
        search_kb,
    )

except Exception as exc:
    DAY2_IMPORT_ERROR = exc
    SAMPLE_RECORDS = {}

    def search_kb(query: str, top_k: int = 3) -> Dict[str, Any]:  # type: ignore
        return {
            "success": False,
            "error": (
                "day2_agents.itsm_agent non importabile. "
                f"Errore: {DAY2_IMPORT_ERROR}"
            ),
            "results": [],
        }

    def lookup_record(record_id: str) -> Dict[str, Any]:  # type: ignore
        return {
            "success": False,
            "error": (
                "day2_agents.itsm_agent non importabile. "
                f"Errore: {DAY2_IMPORT_ERROR}"
            ),
            "results": [],
        }

    def compute_sla(ticket: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
        return {
            "success": False,
            "error": (
                "day2_agents.itsm_agent non importabile. "
                f"Errore: {DAY2_IMPORT_ERROR}"
            ),
            "results": [],
        }

    def build_pending_action(  # type: ignore
        ticket: Dict[str, Any],
        sla: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        return None

    def execute_critical_action(  # type: ignore
        action: Dict[str, Any],
        approved: bool,
    ) -> Dict[str, Any]:
        return {
            "success": False,
            "error": "day2_agents.itsm_agent non importabile.",
            "results": [],
        }


# ---------------------------------------------------------------------
# Modelli Pydantic
# ---------------------------------------------------------------------

NEXT_NODE = Literal[
    "triage",
    "knowledge",
    "action",
    "end",
]


class TriageDecision(BaseModel):
    """Output strutturato del TriageAgent."""

    domain: Literal["itsm", "hr", "finance", "other"] = Field(
        description="Dominio operativo principale della richiesta."
    )
    intent: Literal[
        "question",
        "investigation",
        "action_request",
        "complaint",
    ] = Field(description="Intento dell'utente.")
    priority_guess: Literal["P1", "P2", "P3", "P4"] = Field(
        description="Priorità stimata in base al testo. Verrà confermata dai record."
    )
    needs_knowledge: bool = Field(
        description="True se serve documentazione (policy, SLA, procedure)."
    )
    needs_action: bool = Field(
        description="True se la richiesta implica un'azione operativa (escalation, notifica, change)."
    )
    reasoning: str = Field(
        description="Spiegazione breve della classificazione (max 2 frasi)."
    )


class SupervisorDecision(BaseModel):
    """Output strutturato del Supervisor."""

    next: NEXT_NODE = Field(
        description=(
            "Prossimo nodo da invocare. 'end' quando la risposta finale è pronta."
        )
    )
    reason: str = Field(
        description="Perché il supervisor sceglie questo nodo (max 2 frasi)."
    )


class HandoffEvent(BaseModel):
    """Una riga di log per ogni handoff supervisor -> agent o agent -> supervisor."""

    step: int
    from_agent: str
    to_agent: str
    decision_reason: Optional[str] = None
    tokens_in: int = 0
    tokens_out: int = 0
    timestamp: float


class TraceEvent(BaseModel):
    """Evento generico di trace per debug / replay."""

    step: int
    event: str
    timestamp: float
    agent: Optional[str] = None
    payload: Optional[Any] = None
    error: Optional[str] = None

class AgentState(TypedDict):
    """
    Stato condiviso del grafo multi-agent.

    Convenzione importante:
    - messages, citations, actions, handoffs e traces usano reducer append-only.
    - quindi i nodi devono restituire SOLO i nuovi elementi prodotti nel turno,
      non una copia completa dello stato precedente.
    """

    messages: Annotated[list, add_messages]
    task_id: str

    triage: Optional[dict]
    ticket: Optional[dict]
    sla: Optional[dict]

    citations: Annotated[List[dict], lambda old, new: list(old) + list(new)]
    actions: Annotated[List[dict], lambda old, new: list(old) + list(new)]
    handoffs: Annotated[List[dict], lambda old, new: list(old) + list(new)]
    traces: Annotated[List[dict], lambda old, new: list(old) + list(new)]

    next: NEXT_NODE

    tokens_in: int
    tokens_out: int

    # Contatori difensivi: evitano loop quando un agent non riesce a produrre
    # lo stato necessario, per esempio ticket mancante o action non generabile.
    knowledge_attempts: int
    action_attempts: int

    # Modalità didattica veloce: riduce drasticamente le chiamate LLM.
    # Utile con quote free-tier basse o durante demo in aula.
    fast_mode: bool

    # Human-in-the-Loop: True = approvato, False = rifiutato, None = non ancora chiesta.
    human_decision: Optional[bool]


def model_to_dict(model: BaseModel) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def to_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, default=str)


def append_trace(
    traces: List[dict],
    *,
    step: int,
    event: str,
    agent: Optional[str] = None,
    payload: Optional[Any] = None,
    error: Optional[str] = None,
) -> None:
    event_dict = model_to_dict(
        TraceEvent(
            step=step,
            event=event,
            agent=agent,
            payload=payload,
            error=error,
            timestamp=time.time(),
        )
    )
    traces.append(event_dict)

    cb = _trace_callback.get()
    if cb is not None:
        cb(event_dict)


def append_handoff(
    handoffs: List[dict],
    *,
    step: int,
    from_agent: str,
    to_agent: str,
    decision_reason: Optional[str] = None,
    usage_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    usage = usage_metadata or {}
    handoffs.append(
        model_to_dict(
            HandoffEvent(
                step=step,
                from_agent=from_agent,
                to_agent=to_agent,
                decision_reason=decision_reason,
                tokens_in=int(usage.get("input_tokens", 0) or 0),
                tokens_out=int(usage.get("output_tokens", 0) or 0),
                timestamp=time.time(),
            )
        )
    )


def extract_usage(message: Any) -> Dict[str, Any]:
    """Estrae usage da un messaggio LLM in modo robusto."""
    usage = getattr(message, "usage_metadata", None) or {}
    if not usage and hasattr(message, "response_metadata"):
        usage = (message.response_metadata or {}).get("usage_metadata") or {}
    return dict(usage) if usage else {}

TICKET_ID_RE = re.compile(r"\bINC-\d+\b", flags=re.IGNORECASE)


def extract_ticket_id(text: str) -> Optional[str]:
    """Estrae il primo ID ticket ITSM dal testo, se presente."""
    match = TICKET_ID_RE.search(text or "")
    return match.group(0).upper() if match else None


def get_latest_user_text(state: AgentState) -> str:
    """Recupera l'ultimo messaggio utente dallo stato."""
    user_msg = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None,
    )
    return str(user_msg.content) if user_msg else ""


def next_trace_step(state: AgentState, local_traces: List[dict]) -> int:
    """
    Calcola lo step progressivo per un trace event.

    local_traces contiene solo i delta prodotti dal nodo corrente.
    """
    return len(state.get("traces") or []) + len(local_traces) + 1


def next_handoff_step(state: AgentState, local_handoffs: List[dict]) -> int:
    """
    Calcola lo step progressivo per un handoff event.

    local_handoffs contiene solo i delta prodotti dal nodo corrente.
    """
    return len(state.get("handoffs") or []) + len(local_handoffs) + 1


def serialize_message(message: Any) -> Dict[str, Any]:
    """Serializza messaggi LangChain in modo leggibile dentro il trace JSON."""
    return {
        "type": message.__class__.__name__,
        "name": getattr(message, "name", None),
        "content": getattr(message, "content", str(message)),
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
    Salva lo stato completo della run in JSON.

    Di default scrive in:
        runs/day3_multiagent/

    Il terminale resta leggibile, mentre il JSON completo rimane disponibile
    per debug, audit, replay didattico e confronto tra run.
    """
    output_dir = Path(trace_dir).expanduser().resolve() if trace_dir else (
        PROJECT_ROOT / "runs" / "day3_multiagent"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    task_id = str(state.get("task_id", "no-task"))[:8]
    path = output_dir / f"{mode}_{timestamp}_{task_id}.json"

    payload = {
        "run": {
            "mode": mode,
            "query": query,
            "answer": answer,
            "saved_at": timestamp,
        },
        "summary": {
            "task_id": state.get("task_id"),
            "next": state.get("next"),
            "knowledge_attempts": state.get("knowledge_attempts", 0),
            "action_attempts": state.get("action_attempts", 0),
            "counts": {
                "messages": len(state.get("messages", [])),
                "handoffs": len(state.get("handoffs", [])),
                "traces": len(state.get("traces", [])),
                "citations": len(state.get("citations", [])),
                "actions": len(state.get("actions", [])),
            },
            "token_usage": {
                "tokens_in": state.get("tokens_in", 0),
                "tokens_out": state.get("tokens_out", 0),
                "tokens_total": (
                    int(state.get("tokens_in", 0) or 0)
                    + int(state.get("tokens_out", 0) or 0)
                ),
            },
        },
        "state": state_to_serializable(state),
    }

    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    return path


def print_compact_handoffs(handoffs: List[dict]) -> None:
    """Stampa una vista compatta e leggibile degli handoff."""
    if not handoffs:
        print("Nessun handoff registrato.")
        return

    for h in handoffs:
        print(
            f"{h.get('step')}. "
            f"{h.get('from_agent')} -> {h.get('to_agent')} | "
            f"{h.get('decision_reason')}"
        )

def fallback_triage_decision(user_text: str) -> Dict[str, Any]:
    """
    Triage deterministico.

    Serve in modalità --fast o come fallback quando il provider LLM è in quota.
    Non pretende di essere semanticamente perfetto: è una regola robusta per
    i casi didattici ITSM del laboratorio.
    """
    lowered = user_text.lower()
    has_ticket = extract_ticket_id(user_text) is not None

    action_keywords = [
        "proponi azione",
        "apri",
        "esegui",
        "escala",
        "escalation",
        "chiudi",
        "notifica",
    ]

    knowledge_keywords = [
        "mostrami",
        "controlla",
        "verifica",
        "sla",
        "policy",
        "procedura",
        "quando",
    ]

    needs_action = any(k in lowered for k in action_keywords)
    needs_knowledge = has_ticket or any(k in lowered for k in knowledge_keywords)

    if has_ticket:
        intent = "investigation"
    elif needs_action:
        intent = "action_request"
    else:
        intent = "question"

    priority_guess = "P1" if "p1" in lowered or has_ticket else "P3"

    return {
        "domain": "itsm",
        "intent": intent,
        "priority_guess": priority_guess,
        "needs_knowledge": needs_knowledge,
        "needs_action": needs_action,
        "reasoning": (
            "Classificazione deterministica basata su ID ticket e parole chiave "
            "operative."
        ),
    }


def build_deterministic_final_answer(state: AgentState) -> str:
    """Costruisce una risposta finale senza chiamare il modello."""
    ticket = state.get("ticket") or {}
    sla = state.get("sla") or {}
    actions = state.get("actions") or []

    if not ticket:
        return (
            "Non sono riuscito a recuperare il ticket richiesto. "
            "Non propongo azioni operative senza un record valido."
        )

    ticket_id = ticket.get("id") or ticket.get("key") or "ticket sconosciuto"
    priority = ticket.get("priority", "n/d")
    status = ticket.get("status", "n/d")
    service = ticket.get("service", "n/d")
    owner = ticket.get("owner", "n/d")
    summary = ticket.get("summary", "n/d")
    impact = ticket.get("business_impact", "")

    lines = [
        f"Ho recuperato il ticket {ticket_id}.",
        "",
        "Sintesi operativa:",
        f"- Summary: {summary}",
        f"- Priorità: {priority}",
        f"- Stato: {status}",
        f"- Servizio: {service}",
        f"- Owner: {owner}",
    ]

    if impact:
        lines.append(f"- Impatto business: {impact}")

    if sla:
        lines.extend([
            "",
            "Valutazione SLA:",
            f"- Stato SLA: {sla.get('status', 'n/d')}",
            f"- Tempo trascorso: {sla.get('elapsed_hours', 'n/d')}h",
            f"- Soglia: {sla.get('threshold_hours', 'n/d')}h",
            f"- Ore residue: {sla.get('remaining_hours', 'n/d')}h",
            f"- Raccomandazione: {sla.get('recommendation', 'n/d')}",
        ])

        reason = sla.get("reason")
        if reason:
            lines.append(f"- Motivazione: {reason}")

    if actions:
        lines.extend([
            "",
            "Azione proposta:",
        ])

        for action in actions:
            lines.append(
                f"- {action.get('action_type', 'azione')} "
                f"per {action.get('ticket_id', ticket_id)} "
                f"verso {action.get('owner', owner)} "
                f"[stato: {action.get('status', 'n/d')}]"
            )

        lines.extend([
            "",
            "Conferma umana richiesta: sì.",
            "L'azione è stata preparata come pending action, ma non è stata eseguita definitivamente.",
        ])
    else:
        lines.extend([
            "",
            "Nessuna azione operativa è stata preparata.",
        ])

    return "\n".join(lines)

# ---------------------------------------------------------------------
# Tool LangChain — divisi PER AGENT (principio del minimo privilegio)
# ---------------------------------------------------------------------

@tool
def kb_search_tool(query: str, top_k: int = 3) -> str:
    """
    Cerca policy, procedure, SLA, escalation e knowledge article nella knowledge base ITSM.

    Tool del KnowledgeAgent. Restituisce sempre JSON con success, results, error.
    Ogni hit contiene source e snippet, utilizzabili come citazione.

    Nota didattica:
    LangChain usa questa docstring come descrizione del tool. È volutamente
    esplicita per aiutare il modello a decidere quando chiamarlo.
    """
    return to_json(search_kb(query, top_k))


@tool
def kb_lookup_record_tool(record_id: str) -> str:
    """
    Recupera un record operativo ITSM/Jira-like dato un id (es. INC-1002).

    Tool del KnowledgeAgent: l'agent ha bisogno di leggere i record per
    contestualizzare la documentazione, ma NON è autorizzato a modificarli.
    """
    return to_json(lookup_record(record_id))


@tool
def kb_compute_sla_tool(
    id: str,
    priority: str,
    elapsed_hours: float,
    owner: str = "",
    service: str = "unknown",
    workaround_available: bool = False,
) -> str:
    """
    Calcola stato SLA e raccomandazione operativa per un ticket.

    Tool del KnowledgeAgent: calcolo deterministico, non muta stato esterno.
    Output: violated / near_breach / ok + recommendation.
    """
    ticket = {
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
        "opened_at": datetime.now() - timedelta(hours=elapsed_hours),
        "owner": owner,
        "assignee": None,
        "reporter": "unknown",
        "affected_users": 0,
        "business_impact": "",
        "workaround_available": workaround_available,
        "labels": [],
        "linked_records": [],
        "comments": [],
    }
    return to_json(compute_sla(ticket))


@tool
def action_propose_escalation_tool(
    ticket_id: str,
    priority: str,
    owner: str,
    reason: str,
    idempotency_key: str,
) -> str:
    """
    Prepara (NON esegue) una escalation formale verso il team owner.

    Tool dell'ActionAgent. Restituisce una pending_action che dovrà essere
    approvata da un operatore umano prima dell'esecuzione effettiva.

    L'argomento idempotency_key serve a evitare doppie escalation in caso
    di retry: se la chiamata viene ripetuta con la stessa key, l'azione
    non viene duplicata.
    """
    action = {
        "action_type": "open_formal_escalation",
        "ticket_id": ticket_id,
        "priority": priority,
        "owner": owner,
        "critical": True,
        "reason": reason,
        "idempotency_key": idempotency_key,
    }
    return to_json({
        "success": True,
        "results": [action],
        "error": None,
        "meta": {"requires_human_approval": True},
    })


@tool
def action_execute_tool(
    ticket_id: str,
    action_type: str,
    owner: str,
    approved: bool,
    idempotency_key: str,
) -> str:
    """
    Esegue (in modo simulato) un'azione critica precedentemente proposta.

    Tool dell'ActionAgent. L'esecuzione avviene SOLO se approved=True.

    In una pipeline reale questa chiamata produrrebbe side-effect:
    creazione ticket nel sistema downstream, invio mail, post nei canali
    di alerting. In questo lab si limita a registrare l'evento.
    """
    pending = {
        "action_type": action_type,
        "ticket_id": ticket_id,
        "priority": "",
        "owner": owner,
        "critical": True,
        "reason": "Eseguito tramite ActionAgent",
        "idempotency_key": idempotency_key,
    }
    return to_json(execute_critical_action(pending, approved))


# Tool registry PER AGENTE: minimo privilegio
KNOWLEDGE_TOOLS = [kb_search_tool, kb_lookup_record_tool, kb_compute_sla_tool]
ACTION_TOOLS = [action_propose_escalation_tool, action_execute_tool]

TOOL_MAP = {t.name: t for t in (*KNOWLEDGE_TOOLS, *ACTION_TOOLS)}


# ---------------------------------------------------------------------
# Prompt per gli agenti
# ---------------------------------------------------------------------

TRIAGE_PROMPT = """
Sei il TriageAgent di un sistema ITSM enterprise.

Compito:
- leggere la richiesta dell'utente;
- classificare dominio, intent, priorità stimata;
- decidere se serve documentazione (knowledge) o un'azione (action).

Regole:
1. Non rispondere all'utente: produci SOLO l'output strutturato secondo lo schema.
2. Non chiamare tool: il tuo unico output è una decisione di triage.
3. Se la richiesta cita un record (es. INC-1002), è quasi sempre intent="investigation".
4. Se la richiesta usa verbi come "esegui", "apri escalation", "chiudi" -> needs_action=True.
5. Se la richiesta cita policy, SLA, regole operative -> needs_knowledge=True.
6. Rispondi sempre in italiano nel campo reasoning, max 2 frasi.
""".strip()


KNOWLEDGE_PROMPT = """
Sei il KnowledgeAgent di un sistema ITSM.

Compito:
- recuperare informazioni operative e documentali per supportare il supervisor.

Regole:
1. Usa SOLO i tool: kb_search_tool, kb_lookup_record_tool, kb_compute_sla_tool.
2. Non eseguire azioni con side-effect (lo fa ActionAgent).
3. Per ogni evidenza, indica la fonte (source del KB hit o id del record).
4. Restituisci una sintesi breve, 5-8 righe, con citazioni esplicite.
5. Non inventare ticket, policy o tempi SLA: se un tool fallisce, dichiaralo.
6. Termina il messaggio con la riga: "RITORNO AL SUPERVISOR".
""".strip()


ACTION_PROMPT = """
Sei l'ActionAgent di un sistema ITSM.

Compito:
- preparare ed eseguire azioni operative quando il supervisor lo richiede.

Regole:
1. Usa SOLO i tool: action_propose_escalation_tool, action_execute_tool.
2. Le azioni critiche devono SEMPRE generare prima una pending_action
   (action_propose_escalation_tool) e poi richiedere conferma umana.
3. Non eseguire (action_execute_tool con approved=True) senza che il
   supervisor abbia messaggiato esplicitamente "AUTORIZZATO".
4. Genera idempotency_key = "<ticket_id>:<action_type>".
5. Termina il messaggio con la riga: "RITORNO AL SUPERVISOR".
""".strip()


SUPERVISOR_PROMPT = """
Sei il Supervisor di un sistema multi-agent ITSM.

Hai accesso a tre agenti specializzati:
- triage: classifica una nuova richiesta (intent, dominio, priorità).
- knowledge: cerca policy, recupera record, calcola SLA.
- action: prepara o esegue azioni critiche (escalation, notifiche).

Compito:
- leggere lo stato corrente (triage, ticket, sla, actions) e decidere il
  prossimo nodo, secondo le regole sotto.

Regole di routing:
1. Se state['triage'] è None: next = 'triage'.
2. Altrimenti, se needs_knowledge=True e ticket/sla non sono ancora valorizzati:
   next = 'knowledge'.
3. Altrimenti, se needs_action=True e nessuna pending action è stata ancora
   preparata: next = 'action'.
4. Altrimenti: next = 'end'.

Quando next = 'end', produci una risposta finale per l'utente con:
- sintesi
- evidenze (citazioni)
- valutazione SLA (se presente)
- raccomandazione
- eventuale conferma umana richiesta

Rispondi sempre in italiano. Il campo reason deve essere conciso (1-2 frasi).
""".strip()


# ---------------------------------------------------------------------
# Costruzione modello base
# ---------------------------------------------------------------------

# def make_llm() -> ChatGoogleGenerativeAI:
#     if not os.getenv("GOOGLE_API_KEY"):
#         raise RuntimeError("GOOGLE_API_KEY non configurata")
#     return ChatGoogleGenerativeAI(
#         model=GEMINI_MODEL,
#         temperature=0,
#         timeout=25,
#     )


def rate_limit_sleep(last_call_ts: float) -> float:
    now = time.time()
    wait = MIN_SECONDS_BETWEEN_MODEL_CALLS - (now - last_call_ts)
    if wait > 0:
        time.sleep(wait)
    return time.time()

# ----------------------------------------------------------------------
# Helper
# ----------------------------------------------------------------------

def is_quota_error(exc: Exception) -> bool:
    """Riconosce errori di quota/rate limit del provider LLM."""
    text = str(exc).lower()
    return (
        "429" in text
        or "resource_exhausted" in text
        or "quota" in text
        or "rate limit" in text
    )


def fallback_triage_decision(user_text: str) -> Dict[str, Any]:
    """
    Triage deterministico di emergenza.

    Serve per non bloccare la demo quando il provider LLM è in quota error.
    Non sostituisce il modello in produzione, ma rende il laboratorio robusto.
    """
    lowered = user_text.lower()
    has_ticket = extract_ticket_id(user_text) is not None

    action_keywords = [
        "proponi azione",
        "apri",
        "esegui",
        "escala",
        "escalation",
        "chiudi",
        "notifica",
    ]

    knowledge_keywords = [
        "mostrami",
        "controlla",
        "verifica",
        "sla",
        "policy",
        "procedura",
        "quando",
    ]

    needs_action = any(k in lowered for k in action_keywords)
    needs_knowledge = has_ticket or any(k in lowered for k in knowledge_keywords)

    if has_ticket:
        intent = "investigation"
    elif needs_action:
        intent = "action_request"
    else:
        intent = "question"

    priority_guess = "P1" if "p1" in lowered or has_ticket else "P3"

    return {
        "domain": "itsm",
        "intent": intent,
        "priority_guess": priority_guess,
        "needs_knowledge": needs_knowledge,
        "needs_action": needs_action,
        "reasoning": (
            "Fallback deterministico: classificazione basata su ID ticket "
            "e parole chiave operative, usata perché il modello non è disponibile."
        ),
    }

def estimate_cost_from_run_json(path: str) -> Dict[str, Any]:
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

    traces = state.get("traces", [])
    handoffs = state.get("handoffs", [])

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
            "handoffs": len(handoffs),
            "traces": len(traces),
            "actions": len(state.get("actions", [])),
            "citations": len(state.get("citations", [])),
        },
        "prices_per_1M": {
            "input": PRICE_INPUT_PER_1M,
            "output": PRICE_OUTPUT_PER_1M,
        },
    }

def cmd_cost_from_run(args: argparse.Namespace) -> None:
    print_json(estimate_cost_from_run_json(args.path))

# ---------------------------------------------------------------------
# Implementazione dei nodi agente
# ---------------------------------------------------------------------

def triage_agent(state: AgentState) -> Dict[str, Any]:
    """Classifica la richiesta in modo strutturato, con fallback deterministico."""
    user_text = get_latest_user_text(state)
    traces: List[dict] = []

    if state.get("fast_mode", False):
        triage_dict = fallback_triage_decision(user_text)

        append_trace(
            traces,
            step=next_trace_step(state, traces),
            event="triage_fast",
            agent="triage",
            payload=triage_dict,
        )

        return {
            "triage": triage_dict,
            "traces": traces,
            "next": "knowledge" if triage_dict["needs_knowledge"] else (
                "action" if triage_dict["needs_action"] else "end"
            ),
        }

    try:
        llm = get_vertex_llm().with_structured_output(TriageDecision)

        decision = llm.invoke([
            SystemMessage(content=TRIAGE_PROMPT),
            HumanMessage(content=user_text),
        ])

        triage_dict = model_to_dict(decision)

        append_trace(
            traces,
            step=next_trace_step(state, traces),
            event="triage_decision",
            agent="triage",
            payload=triage_dict,
        )

    except Exception as exc:
        triage_dict = fallback_triage_decision(user_text)

        append_trace(
            traces,
            step=next_trace_step(state, traces),
            event="triage_fallback",
            agent="triage",
            payload={
                "triage": triage_dict,
                "provider_error": str(exc),
                "quota_like_error": is_quota_error(exc),
            },
            error=str(exc),
        )

    return {
        "triage": triage_dict,
        "traces": traces,
        "next": "knowledge" if triage_dict["needs_knowledge"] else (
            "action" if triage_dict["needs_action"] else "end"
        ),
    }

def build_deterministic_final_answer(state: AgentState) -> str:
    """Costruisce una risposta finale senza chiamare il modello."""
    ticket = state.get("ticket") or {}
    sla = state.get("sla") or {}
    actions = state.get("actions") or []

    if not ticket:
        return (
            "Non sono riuscito a recuperare il ticket richiesto. "
            "Non eseguo né propongo azioni operative senza un record valido."
        )

    ticket_id = ticket.get("id") or ticket.get("key") or "ticket sconosciuto"
    priority = ticket.get("priority", "n/d")
    status = ticket.get("status", "n/d")
    service = ticket.get("service", "n/d")
    owner = ticket.get("owner", "n/d")

    lines = [
        f"Ho recuperato il ticket {ticket_id}.",
        "",
        "Sintesi operativa:",
        f"- Priorità: {priority}",
        f"- Stato: {status}",
        f"- Servizio: {service}",
        f"- Owner: {owner}",
    ]

    if sla:
        lines.extend([
            "",
            "Valutazione SLA:",
            f"- Stato SLA: {sla.get('status', 'n/d')}",
            f"- Tempo trascorso: {sla.get('elapsed_hours', 'n/d')}h",
            f"- Soglia: {sla.get('threshold_hours', 'n/d')}h",
            f"- Raccomandazione: {sla.get('recommendation', 'n/d')}",
        ])

    if actions:
        lines.extend([
            "",
            "Azione proposta:",
        ])
        for action in actions:
            lines.append(
                f"- {action.get('action_type', 'azione')} "
                f"per {action.get('ticket_id', ticket_id)} "
                f"verso {action.get('owner', owner)} "
                f"[stato: {action.get('status', 'n/d')}]"
            )

        lines.extend([
            "",
            "Conferma umana richiesta: sì. "
            "L'azione è stata solo preparata, non eseguita definitivamente.",
        ])
    else:
        lines.extend([
            "",
            "Nessuna azione operativa è stata preparata.",
        ])

    return "\n".join(lines)


def knowledge_agent(state: AgentState) -> Dict[str, Any]:
    """
    Recupera evidenze documentali e operative.

    Con la soluzione B, questo nodo restituisce solo:
    - nuovi messaggi;
    - nuove citazioni;
    - nuovi trace event;
    - eventuali aggiornamenti puntuali di ticket/sla;
    - token cumulativi aggiornati.
    """
    if state.get("fast_mode", False):
        user_text = get_latest_user_text(state)
        ticket_id = extract_ticket_id(user_text)

        traces: List[dict] = []
        citations: List[dict] = []

        ticket = state.get("ticket")
        sla = state.get("sla")

        if not ticket_id:
            text = (
                "KnowledgeAgent fast: nessun ID ticket trovato nella richiesta. "
                "Non posso recuperare record operativo."
            )

            append_trace(
                traces,
                step=next_trace_step(state, traces),
                event="knowledge_fast_no_ticket",
                agent="knowledge",
                payload={"user_text": user_text},
            )

            return {
                "messages": [AIMessage(content=text, name="knowledge")],
                "ticket": ticket,
                "sla": sla,
                "citations": citations,
                "traces": traces,
                "knowledge_attempts": int(state.get("knowledge_attempts", 0) or 0) + 1,
            }

        lookup_result = lookup_record(ticket_id)

        append_trace(
            traces,
            step=next_trace_step(state, traces),
            event="knowledge_fast_lookup",
            agent="knowledge",
            payload={
                "ticket_id": ticket_id,
                "result": lookup_result,
            },
        )

        if lookup_result.get("success") and lookup_result.get("results"):
            ticket = lookup_result["results"][0]
        else:
            text = (
                f"KnowledgeAgent fast: ticket {ticket_id} non trovato. "
                "Non invento dati e restituisco il controllo al supervisor."
            )

            return {
                "messages": [AIMessage(content=text, name="knowledge")],
                "ticket": ticket,
                "sla": sla,
                "citations": citations,
                "traces": traces,
                "knowledge_attempts": int(state.get("knowledge_attempts", 0) or 0) + 1,
            }

        sla_result = compute_sla(ticket)

        append_trace(
            traces,
            step=next_trace_step(state, traces),
            event="knowledge_fast_compute_sla",
            agent="knowledge",
            payload={
                "ticket_id": ticket_id,
                "result": sla_result,
            },
        )

        if sla_result.get("success") and sla_result.get("results"):
            sla = sla_result["results"][0]

        text = (
            f"KnowledgeAgent fast: recuperato {ticket_id} e calcolato SLA. "
            "RITORNO AL SUPERVISOR."
        )

        return {
            "messages": [AIMessage(content=text, name="knowledge")],
            "ticket": ticket,
            "sla": sla,
            "citations": citations,
            "traces": traces,
            "knowledge_attempts": int(state.get("knowledge_attempts", 0) or 0) + 1,
        }

    llm = get_vertex_llm().bind_tools(KNOWLEDGE_TOOLS)

    user_text = get_latest_user_text(state)
    ticket_id = extract_ticket_id(user_text)

    messages: List[Any] = [
        SystemMessage(content=KNOWLEDGE_PROMPT),
        HumanMessage(
            content=(
                "Triage corrente:\n"
                f"{to_json(state.get('triage') or {})}\n\n"
                "Richiesta utente:\n"
                f"{user_text}\n\n"
                "Istruzione operativa:\n"
                "- Se la richiesta contiene un ID ticket, per esempio INC-1002, "
                "devi prima chiamare kb_lookup_record_tool con quell'ID.\n"
                "- Se il record viene recuperato, calcola poi lo SLA con "
                "kb_compute_sla_tool usando i dati del record.\n"
                "- Se il record non esiste, dichiaralo senza inventare dati."
            )
        ),
    ]

    traces: List[dict] = []
    citations: List[dict] = []

    ticket = state.get("ticket")
    sla = state.get("sla")

    tokens_in = int(state.get("tokens_in", 0) or 0)
    tokens_out = int(state.get("tokens_out", 0) or 0)
    last_call = 0.0

    for inner_step in range(1, 5):
        last_call = rate_limit_sleep(last_call)
        response = llm.invoke(messages)
        messages.append(response)

        usage = extract_usage(response)
        tokens_in += int(usage.get("input_tokens", 0) or 0)
        tokens_out += int(usage.get("output_tokens", 0) or 0)

        tool_calls = getattr(response, "tool_calls", None) or []

        if not tool_calls:
            text = response.content if isinstance(response.content, str) else str(response.content)

            append_trace(
                traces,
                step=next_trace_step(state, traces),
                event="knowledge_response",
                agent="knowledge",
                payload={"text": text, "usage": usage},
            )

            return {
                "messages": [AIMessage(content=text, name="knowledge")],
                "ticket": ticket,
                "sla": sla,
                "citations": citations,
                "traces": traces,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "knowledge_attempts": int(state.get("knowledge_attempts", 0) or 0) + 1,
            }

        for tc in tool_calls:
            name = tc.get("name", "")
            args = tc.get("args", {}) or {}

            try:
                raw = TOOL_MAP[name].invoke(args)
            except Exception as exc:
                raw = to_json({"success": False, "error": str(exc), "results": []})

            try:
                parsed = json.loads(raw)
            except Exception:
                parsed = {"success": True, "results": [raw]}

            if name == "kb_lookup_record_tool" and parsed.get("success"):
                results = parsed.get("results") or []
                if results:
                    ticket = results[0]

            if name == "kb_compute_sla_tool" and parsed.get("success"):
                results = parsed.get("results") or []
                if results:
                    sla = results[0]

            if name == "kb_search_tool" and parsed.get("success"):
                for hit in parsed.get("results", []):
                    citations.append({
                        "source": hit.get("source"),
                        "snippet": (hit.get("snippet") or "")[:240],
                    })

            append_trace(
                traces,
                step=next_trace_step(state, traces),
                event="knowledge_tool_call",
                agent="knowledge",
                payload={"tool": name, "args": args, "result": parsed},
            )

            messages.append(
                ToolMessage(
                    content=raw,
                    name=name,
                    tool_call_id=tc.get("id", f"k-{inner_step}-{name}"),
                )
            )

    append_trace(
        traces,
        step=next_trace_step(state, traces),
        event="knowledge_cap_reached",
        agent="knowledge",
        payload={
            "ticket_id_detected": ticket_id,
            "ticket_found": ticket is not None,
            "sla_found": sla is not None,
        },
    )

    return {
        "messages": [AIMessage(content="KnowledgeAgent: cap tool raggiunto.", name="knowledge")],
        "ticket": ticket,
        "sla": sla,
        "citations": citations,
        "traces": traces,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "knowledge_attempts": int(state.get("knowledge_attempts", 0) or 0) + 1,
    }


def action_agent(state: AgentState) -> Dict[str, Any]:
    """
    Prepara / esegue azioni critiche.

    Questo nodo non prova a dedurre autonomamente un ticket dal testo utente:
    opera solo su ticket/sla già presenti nello stato. La responsabilità di
    recuperare il record è del KnowledgeAgent.

    In fast_mode non usa LLM: costruisce una pending action deterministica
    se ticket e SLA sono disponibili.
    """

    if state.get("fast_mode", False):
        ticket = state.get("ticket") or {}
        sla = state.get("sla") or {}

        traces: List[dict] = []
        actions: List[dict] = []

        if not ticket:
            text = (
                "ActionAgent fast: nessun ticket disponibile nello stato. "
                "Non propongo azioni senza record operativo."
            )

            append_trace(
                traces,
                step=next_trace_step(state, traces),
                event="action_fast_no_ticket",
                agent="action",
                payload={},
            )

            return {
                "messages": [AIMessage(content=text, name="action")],
                "actions": actions,
                "traces": traces,
                "action_attempts": int(state.get("action_attempts", 0) or 0) + 1,
            }

        ticket_id = ticket.get("id") or ticket.get("key")
        priority = ticket.get("priority", "")
        owner = ticket.get("owner", "unknown")
        recommendation = sla.get("recommendation", "") if sla else ""

        should_propose = (
            recommendation == "escalate"
            or sla.get("status") in {"violated", "near_breach"}
            or bool((state.get("triage") or {}).get("needs_action"))
        )

        if should_propose:
            action = {
                "action_type": "open_formal_escalation",
                "ticket_id": ticket_id,
                "priority": priority,
                "owner": owner,
                "critical": True,
                "reason": sla.get("reason") or "Azione richiesta dall'utente o raccomandata dallo SLA.",
                "idempotency_key": f"{ticket_id}:open_formal_escalation",
                "status": "pending_approval",
            }
            actions.append(action)

            text = (
                f"ActionAgent fast: preparata escalation formale per {ticket_id}. "
                "Stato: pending_approval. RITORNO AL SUPERVISOR."
            )

            append_trace(
                traces,
                step=next_trace_step(state, traces),
                event="action_fast_proposal",
                agent="action",
                payload={"action": action},
            )

        else:
            text = (
                f"ActionAgent fast: nessuna escalation necessaria per {ticket_id}. "
                "RITORNO AL SUPERVISOR."
            )

            append_trace(
                traces,
                step=next_trace_step(state, traces),
                event="action_fast_no_action_needed",
                agent="action",
                payload={
                    "ticket_id": ticket_id,
                    "sla": sla,
                },
            )

        return {
            "messages": [AIMessage(content=text, name="action")],
            "actions": actions,
            "traces": traces,
            "action_attempts": int(state.get("action_attempts", 0) or 0) + 1,
        }

    llm = get_vertex_llm().bind_tools(ACTION_TOOLS)

    ticket = state.get("ticket") or {}
    sla = state.get("sla") or {}
    triage = state.get("triage") or {}

    proposal = build_pending_action(ticket, sla) if ticket and sla else None

    messages: List[Any] = [
        SystemMessage(content=ACTION_PROMPT),
        HumanMessage(
            content=(
                "Stato corrente per la decisione operativa:\n"
                f"triage = {to_json(triage)}\n"
                f"ticket = {to_json({k: ticket.get(k) for k in ('id','priority','owner','service')})}\n"
                f"sla    = {to_json(sla)}\n"
                f"proposta deterministica = {to_json(proposal)}\n\n"
                "Se la proposta deterministica è non nulla, conferma usando "
                "action_propose_escalation_tool con idempotency_key = "
                "f\"{ticket_id}:open_formal_escalation\".\n"
                "Se ticket o SLA sono mancanti, non inventare dati e restituisci "
                "una spiegazione breve."
            )
        ),
    ]

    traces: List[dict] = []
    actions: List[dict] = []

    tokens_in = int(state.get("tokens_in", 0) or 0)
    tokens_out = int(state.get("tokens_out", 0) or 0)
    last_call = 0.0

    for inner_step in range(1, 4):
        last_call = rate_limit_sleep(last_call)
        response = llm.invoke(messages)
        messages.append(response)

        usage = extract_usage(response)
        tokens_in += int(usage.get("input_tokens", 0) or 0)
        tokens_out += int(usage.get("output_tokens", 0) or 0)

        tool_calls = getattr(response, "tool_calls", None) or []

        if not tool_calls:
            text = response.content if isinstance(response.content, str) else str(response.content)

            append_trace(
                traces,
                step=next_trace_step(state, traces),
                event="action_response",
                agent="action",
                payload={"text": text, "usage": usage},
            )

            return {
                "messages": [AIMessage(content=text, name="action")],
                "actions": actions,
                "traces": traces,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "action_attempts": int(state.get("action_attempts", 0) or 0) + 1,
            }

        for tc in tool_calls:
            name = tc.get("name", "")
            args = tc.get("args", {}) or {}

            try:
                raw = TOOL_MAP[name].invoke(args)
            except Exception as exc:
                raw = to_json({"success": False, "error": str(exc), "results": []})

            try:
                parsed = json.loads(raw)
            except Exception:
                parsed = {"success": True, "results": [raw]}

            if name == "action_propose_escalation_tool" and parsed.get("success"):
                for a in parsed.get("results", []):
                    actions.append({**a, "status": "pending_approval"})

            if name == "action_execute_tool" and parsed.get("success"):
                for a in parsed.get("results", []):
                    actions.append({"executed": a.get("executed"), "detail": a})

            append_trace(
                traces,
                step=next_trace_step(state, traces),
                event="action_tool_call",
                agent="action",
                payload={"tool": name, "args": args, "result": parsed},
            )

            messages.append(
                ToolMessage(
                    content=raw,
                    name=name,
                    tool_call_id=tc.get("id", f"a-{inner_step}-{name}"),
                )
            )

    append_trace(
        traces,
        step=next_trace_step(state, traces),
        event="action_cap_reached",
        agent="action",
        payload={"actions_created": len(actions)},
    )

    return {
        "messages": [AIMessage(content="ActionAgent: cap tool raggiunto.", name="action")],
        "actions": actions,
        "traces": traces,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "action_attempts": int(state.get("action_attempts", 0) or 0) + 1,
    }


def supervisor_node(state: AgentState) -> Dict[str, Any]:
    """
    Decide il prossimo agente o produce la risposta finale.

    Il routing è volutamente quasi tutto deterministico: costa meno,
    è più prevedibile e rende più chiaro il comportamento del grafo.
    """
    triage = state.get("triage")
    ticket = state.get("ticket")
    sla = state.get("sla")
    actions = state.get("actions") or []

    user_text = get_latest_user_text(state)
    mentioned_ticket_id = extract_ticket_id(user_text)

    knowledge_attempts = int(state.get("knowledge_attempts", 0) or 0)
    action_attempts = int(state.get("action_attempts", 0) or 0)

    traces: List[dict] = []
    handoffs: List[dict] = []

    tokens_in = int(state.get("tokens_in", 0) or 0)
    tokens_out = int(state.get("tokens_out", 0) or 0)

    # 1. Prima classificazione.
    if triage is None:
        next_node: NEXT_NODE = "triage"
        reason = "Nessun triage in stato: prima cosa, classificare la richiesta."

        append_handoff(
            handoffs,
            step=next_handoff_step(state, handoffs),
            from_agent="supervisor",
            to_agent=next_node,
            decision_reason=reason,
        )
        append_trace(
            traces,
            step=next_trace_step(state, traces),
            event="route",
            agent="supervisor",
            payload={"next": next_node, "reason": reason},
        )

        return {"next": next_node, "handoffs": handoffs, "traces": traces}

    # 2. Se la richiesta cita un ticket, prima serve il KnowledgeAgent.
    # Questo impedisce di mandare l'ActionAgent a vuoto senza ticket/sla.
    needs_operational_lookup = bool(mentioned_ticket_id)
    needs_knowledge = bool(triage.get("needs_knowledge")) or needs_operational_lookup
    missing_operational_context = ticket is None or sla is None

    if needs_knowledge and missing_operational_context and knowledge_attempts < 2:
        next_node = "knowledge"
        reason = (
            "La richiesta cita un ticket o richiede evidenze operative: "
            "prima di qualunque azione servono lookup record e valutazione SLA."
        )

        append_handoff(
            handoffs,
            step=next_handoff_step(state, handoffs),
            from_agent="supervisor",
            to_agent=next_node,
            decision_reason=reason,
        )
        append_trace(
            traces,
            step=next_trace_step(state, traces),
            event="route",
            agent="supervisor",
            payload={
                "next": next_node,
                "reason": reason,
                "mentioned_ticket_id": mentioned_ticket_id,
                "knowledge_attempts": knowledge_attempts,
            },
        )

        return {"next": next_node, "handoffs": handoffs, "traces": traces}

    # 3. Se è richiesta un'azione, la facciamo solo se esiste un ticket.
    # Se il ticket manca anche dopo il KnowledgeAgent, si chiude con una
    # risposta finale invece di entrare in loop.
    if bool(triage.get("needs_action")) and not actions:
        if ticket is not None and action_attempts < 1:
            next_node = "action"
            reason = (
                "Triage indica azione operativa e il contesto minimo "
                "ticket/SLA è disponibile."
            )

            append_handoff(
                handoffs,
                step=next_handoff_step(state, handoffs),
                from_agent="supervisor",
                to_agent=next_node,
                decision_reason=reason,
            )
            append_trace(
                traces,
                step=next_trace_step(state, traces),
                event="route",
                agent="supervisor",
                payload={
                    "next": next_node,
                    "reason": reason,
                    "action_attempts": action_attempts,
                },
            )

            return {"next": next_node, "handoffs": handoffs, "traces": traces}

        if ticket is None:
            append_trace(
                traces,
                step=next_trace_step(state, traces),
                event="action_skipped",
                agent="supervisor",
                payload={
                    "reason": (
                        "Azione richiesta, ma nessun ticket disponibile nello stato. "
                        "Si produce risposta finale senza side-effect."
                    ),
                    "mentioned_ticket_id": mentioned_ticket_id,
                    "knowledge_attempts": knowledge_attempts,
                },
            )

    # 3.5 Human-in-the-Loop: se ci sono azioni pending_approval, aspetta la decisione umana
    # prima di produrre la risposta finale.
    pending_approval_actions = [a for a in actions if a.get("status") == "pending_approval"]
    if pending_approval_actions and state.get("human_decision") is None:
        cb = _hitl_callback.get()
        if cb is not None:
            approved = cb({
                "task_id": state.get("task_id"),
                "actions": pending_approval_actions,
            })
        else:
            # CLI / nessun callback: default sicuro = non eseguire senza conferma esplicita.
            approved = False

        append_trace(
            traces,
            step=next_trace_step(state, traces),
            event="human_decision_received",
            agent="supervisor",
            payload={"approved": approved, "actions": pending_approval_actions},
        )

        ticket_local = state.get("ticket") or {}
        ticket_id_local = ticket_local.get("id") or ticket_local.get("key") or "ticket sconosciuto"
        if approved:
            decision_text = (
                f"Operatore ha APPROVATO l'azione per {ticket_id_local}. "
                "L'escalation è stata autorizzata."
            )
        else:
            decision_text = (
                f"Operatore ha RIFIUTATO l'azione per {ticket_id_local}. "
                "Nessun side-effect eseguito. Le azioni rimangono in stato pending_approval."
            )

        base_answer = build_deterministic_final_answer(state)
        final_text = f"{base_answer}\n\n---\n{decision_text}"

        append_handoff(
            handoffs,
            step=next_handoff_step(state, handoffs),
            from_agent="supervisor",
            to_agent="end",
            decision_reason=f"HITL completato: {'approvato' if approved else 'rifiutato'}.",
        )

        return {
            "messages": [AIMessage(content=final_text, name="supervisor")],
            "human_decision": approved,
            "next": "end",
            "handoffs": handoffs,
            "traces": traces,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
        }

    # 4. Risposta finale.
    if state.get("fast_mode", False):
        final_text = build_deterministic_final_answer(state)

        append_handoff(
            handoffs,
            step=next_handoff_step(state, handoffs),
            from_agent="supervisor",
            to_agent="end",
            decision_reason="Risposta finale deterministica pronta.",
        )
        append_trace(
            traces,
            step=next_trace_step(state, traces),
            event="final_answer_fast",
            agent="supervisor",
            payload={"text": final_text},
        )

        return {
            "messages": [AIMessage(content=final_text, name="supervisor")],
            "next": "end",
            "handoffs": handoffs,
            "traces": traces,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
        }
    try:
        llm = get_vertex_llm()
        rate_limit_sleep(0.0)

        summary_msg = llm.invoke([
            SystemMessage(content=SUPERVISOR_PROMPT),
            HumanMessage(
                content=(
                    "Stato finale:\n"
                    f"triage   = {to_json(triage)}\n"
                    f"ticket   = {to_json(ticket)}\n"
                    f"sla      = {to_json(sla)}\n"
                    f"citations= {to_json(state.get('citations') or [])}\n"
                    f"actions  = {to_json(actions)}\n"
                    f"knowledge_attempts = {knowledge_attempts}\n"
                    f"action_attempts    = {action_attempts}\n\n"
                    "Produci una risposta finale per l'utente. "
                    "Se manca un ticket o una action non è stata possibile, dichiaralo chiaramente."
                )
            ),
        ])

        usage = extract_usage(summary_msg)
        tokens_in += int(usage.get("input_tokens", 0) or 0)
        tokens_out += int(usage.get("output_tokens", 0) or 0)

        final_text = summary_msg.content if isinstance(summary_msg.content, str) else str(summary_msg.content)

    except Exception as exc:
        usage = {}
        final_text = build_deterministic_final_answer(state)

        append_trace(
            traces,
            step=next_trace_step(state, traces),
            event="final_answer_fallback",
            agent="supervisor",
            payload={
                "reason": "Risposta finale deterministica per errore provider LLM.",
                "provider_error": str(exc),
                "quota_like_error": is_quota_error(exc),
            },
            error=str(exc),
        )

    append_handoff(
        handoffs,
        step=next_handoff_step(state, handoffs),
        from_agent="supervisor",
        to_agent="end",
        decision_reason="Risposta finale pronta.",
        usage_metadata=usage,
    )
    append_trace(
        traces,
        step=next_trace_step(state, traces),
        event="final_answer",
        agent="supervisor",
        payload={"text": final_text, "usage": usage},
    )

    return {
        "messages": [AIMessage(content=final_text, name="supervisor")],
        "next": "end",
        "handoffs": handoffs,
        "traces": traces,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
    }


# ---------------------------------------------------------------------
# Modalità manual: stesso flusso, scritto a mano per chiarezza didattica
# ---------------------------------------------------------------------

def run_manual(query: str, fast_mode: bool = False) -> Tuple[str, AgentState]:
    """
    Esegue il flusso supervisor + 3 agenti senza LangGraph.

    Loop:
        supervisor -> {triage | knowledge | action | end}
        finché next != 'end' e non si superano MAX_HANDOFFS / MAX_TOTAL_SECONDS.
    """
    state: AgentState = {
        "messages": [HumanMessage(content=query)],
        "task_id": str(uuid.uuid4()),
        "triage": None,
        "ticket": None,
        "sla": None,
        "citations": [],
        "actions": [],
        "handoffs": [],
        "traces": [],
        "next": "triage",
        "tokens_in": 0,
        "tokens_out": 0,
        "knowledge_attempts": 0,
        "action_attempts": 0,
        "fast_mode": fast_mode,
        "human_decision": None,
    }

    start = time.time()
    handoff_count = 0

    while True:
        
        if handoff_count >= MAX_HANDOFFS:
            append_trace(
                state["traces"],
                step=len(state["traces"]) + 1,
                event="stop",
                error=f"MAX_HANDOFFS ({MAX_HANDOFFS}) raggiunto",
            )
            return "Max handoffs raggiunti", state

        if time.time() - start > MAX_TOTAL_SECONDS:
            append_trace(state["traces"], step=len(state["traces"]) + 1,
                         event="stop", error=f"Timeout {MAX_TOTAL_SECONDS}s superato")
            return "Timeout agente raggiunto", state

        # Supervisor decide o conclude.
        update = supervisor_node(state)
        state = _merge_state(state, update)
        handoff_count += 1

        if state["next"] == "end":
            last = state["messages"][-1]
            return (
                last.content if isinstance(last.content, str) else str(last.content),
                state,
            )

        if state["next"] == "triage":
            update = triage_agent(state)
            state = _merge_state(state, update)

        elif state["next"] == "knowledge":
            update = knowledge_agent(state)
            state = _merge_state(state, update)

        elif state["next"] == "action":
            update = action_agent(state)
            state = _merge_state(state, update)

        else:
            return f"Routing sconosciuto: {state['next']}", state


def _merge_state(state: AgentState, update: Dict[str, Any]) -> AgentState:
    """
    Merge manuale dello stato.

    Questa funzione replica il comportamento dei reducer LangGraph:
    i nodi restituiscono delta, e qui vengono aggiunti allo stato.
    """
    merged = dict(state)

    for key, value in update.items():
        if key == "messages":
            merged["messages"] = list(merged.get("messages", [])) + list(value)

        elif key in {"citations", "actions", "handoffs", "traces"}:
            merged[key] = list(merged.get(key, [])) + list(value)

        else:
            merged[key] = value

    return merged  # type: ignore[return-value]


# ---------------------------------------------------------------------
# Modalità graph: lo stesso flusso, ma in LangGraph
# ---------------------------------------------------------------------

def build_graph():
    try:
        from langgraph.graph import END, START, StateGraph
        from langgraph.checkpoint.memory import MemorySaver
    except Exception as exc:
        raise RuntimeError(
            "LangGraph non è installato. Installa con: pip install langgraph. "
            f"Dettaglio: {exc}"
        )

    def route_from_supervisor(state: AgentState) -> str:
        return state.get("next", "end")

    builder = StateGraph(AgentState)

    builder.add_node("supervisor", supervisor_node)
    builder.add_node("triage", triage_agent)
    builder.add_node("knowledge", knowledge_agent)
    builder.add_node("action", action_agent)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "triage": "triage",
            "knowledge": "knowledge",
            "action": "action",
            "end": END,
        },
    )
    builder.add_edge("triage", "supervisor")
    builder.add_edge("knowledge", "supervisor")
    builder.add_edge("action", "supervisor")

    return builder.compile(checkpointer=MemorySaver())


def run_graph(
    query: str,
    thread_id: str = "demo-day3-001",
    fast_mode: bool = False,
) -> Dict[str, Any]:
    graph = build_graph()
    initial: AgentState = {
        "messages": [HumanMessage(content=query)],
        "task_id": str(uuid.uuid4()),
        "triage": None,
        "ticket": None,
        "sla": None,
        "citations": [],
        "actions": [],
        "handoffs": [],
        "traces": [],
        "next": "triage",
        "tokens_in": 0,
        "tokens_out": 0,
        "knowledge_attempts": 0,
        "action_attempts": 0,
        "fast_mode": fast_mode,
        "human_decision": None,
    }
    return graph.invoke(initial, config={"configurable": {"thread_id": thread_id}})
    


# ---------------------------------------------------------------------
# Costo di una conversazione multi-agent
# ---------------------------------------------------------------------

def estimate_cost(
    rounds: int = 5,
    agents_per_round: int = 4,
    input_tokens_per_call: int = 800,
    output_tokens_per_call: int = 200,
) -> Dict[str, Any]:
    """
    Stima il costo di una conversazione multi-agent.

    Formula didattica:
        costo = rounds × agents × (input_tok × prezzo_in + output_tok × prezzo_out)
    """
    total_in = rounds * agents_per_round * input_tokens_per_call
    total_out = rounds * agents_per_round * output_tokens_per_call

    cost_in = total_in * PRICE_INPUT_PER_1M / 1_000_000
    cost_out = total_out * PRICE_OUTPUT_PER_1M / 1_000_000
    cost_total = cost_in + cost_out

    return {
        "rounds": rounds,
        "agents_per_round": agents_per_round,
        "tokens": {
            "input": total_in,
            "output": total_out,
            "total": total_in + total_out,
        },
        "cost_usd": {
            "input": round(cost_in, 6),
            "output": round(cost_out, 6),
            "total": round(cost_total, 6),
            "for_10k_queries_day_per_month": round(cost_total * 10_000 * 30, 2),
        },
        "prices_per_1M": {
            "input": PRICE_INPUT_PER_1M,
            "output": PRICE_OUTPUT_PER_1M,
        },
    }


# ---------------------------------------------------------------------
# Prompt di esempio
# ---------------------------------------------------------------------

EXAMPLE_PROMPTS: List[Dict[str, str]] = [
    {
        "title": "1. Multi-agent classico: investigazione + escalation",
        "peculiarity": (
            "TriageAgent classifica come investigation P1, KnowledgeAgent recupera record + SLA, "
            "ActionAgent prepara la pending escalation."
        ),
        "prompt": "Mostrami INC-1002, calcola lo SLA e prepara l'escalation se necessario.",
    },
    {
        "title": "2. Knowledge puro: policy",
        "peculiarity": "Triage classifica come question: solo KnowledgeAgent risponde.",
        "prompt": "Quando un ticket P1 deve essere portato in escalation al team on-call?",
    },
    {
        "title": "3. Near breach P2",
        "peculiarity": (
            "Caso al limite: KnowledgeAgent legge il record, calcola SLA in stato near_breach."
        ),
        "prompt": "Controlla INC-1003: è vicino alla violazione SLA?",
    },
    {
        "title": "4. Errore controllato: record inesistente",
        "peculiarity": (
            "KnowledgeAgent deve dichiarare il fallimento del lookup senza inventare dati."
        ),
        "prompt": "Recupera INC-9999, calcola SLA e proponi un'azione.",
    },
    {
        "title": "5. Action richiesta esplicita",
        "peculiarity": (
            "Triage classifica come action_request: ActionAgent prepara escalation con idempotency."
        ),
        "prompt": (
            "Apri una escalation formale per INC-1002 verso team-sre con motivazione 'SLA P1'."
        ),
    },
]


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2, default=str))


def cmd_examples(_: argparse.Namespace) -> None:
    for item in EXAMPLE_PROMPTS:
        print("=" * 100)
        print(item["title"])
        print(f"Peculiarità: {item['peculiarity']}")
        print(f"Prompt: {item['prompt']}")


def cmd_manual(args: argparse.Namespace) -> None:
    answer, state = run_manual(args.query, fast_mode=args.fast)

    run_path = save_run_json(
        mode="manual",
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
        "next": state.get("next"),
        "fast_mode": state.get("fast_mode", False),
        "knowledge_attempts": state.get("knowledge_attempts", 0),
        "action_attempts": state.get("action_attempts", 0),
        "handoffs": len(state.get("handoffs", [])),
        "traces": len(state.get("traces", [])),
        "citations": len(state.get("citations", [])),
        "actions": len(state.get("actions", [])),
        "tokens_in": state.get("tokens_in", 0),
        "tokens_out": state.get("tokens_out", 0),
        "trace_file": str(run_path),
    })

    print("\nROUTE COMPACT")
    print("=" * 100)
    print_compact_handoffs(state.get("handoffs", []))

    if args.verbose:
        print("\nHANDOFFS")
        print("=" * 100)
        print_json(state.get("handoffs", []))

        print("\nTRACE")
        print("=" * 100)
        print_json(state.get("traces", []))


def cmd_graph(args: argparse.Namespace) -> None:
    result = run_graph(
        args.query,
        thread_id=args.thread_id,
        fast_mode=args.fast,
    )

    print("\nGRAPH RESULT")
    print("=" * 100)

    messages = result.get("messages", []) if isinstance(result, dict) else []
    answer = ""

    if messages:
        last = messages[-1]
        content = last.content if hasattr(last, "content") else str(last)
        answer = content if isinstance(content, str) else str(content)
        print(answer)
    else:
        print("Nessuna risposta testuale")

    run_path = save_run_json(
        mode="graph",
        query=args.query,
        answer=answer,
        state=result if isinstance(result, dict) else {"raw_result": result},
        trace_dir=args.trace_dir,
    )

    print("\nSUMMARY")
    print("=" * 100)
    print_json({
        "thread_id": args.thread_id,
        "task_id": result.get("task_id") if isinstance(result, dict) else None,
        "next": result.get("next") if isinstance(result, dict) else None,
        "fast_mode": result.get("fast_mode", False) if isinstance(result, dict) else False,
        "knowledge_attempts": result.get("knowledge_attempts", 0) if isinstance(result, dict) else 0,
        "action_attempts": result.get("action_attempts", 0) if isinstance(result, dict) else 0,
        "handoffs": len(result.get("handoffs", [])) if isinstance(result, dict) else 0,
        "traces": len(result.get("traces", [])) if isinstance(result, dict) else 0,
        "citations": len(result.get("citations", [])) if isinstance(result, dict) else 0,
        "actions": len(result.get("actions", [])) if isinstance(result, dict) else 0,
        "tokens_in": result.get("tokens_in", 0) if isinstance(result, dict) else 0,
        "tokens_out": result.get("tokens_out", 0) if isinstance(result, dict) else 0,
        "trace_file": str(run_path),
    })

    print("\nROUTE COMPACT")
    print("=" * 100)
    print_compact_handoffs(
        result.get("handoffs", []) if isinstance(result, dict) else []
    )

    if args.verbose and isinstance(result, dict):
        print("\nHANDOFFS")
        print("=" * 100)
        print_json(result.get("handoffs", []))

        print("\nTRACE")
        print("=" * 100)
        print_json(result.get("traces", []))


def cmd_cost(args: argparse.Namespace) -> None:
    print_json(estimate_cost(
        rounds=args.rounds,
        agents_per_round=args.agents,
        input_tokens_per_call=args.input_tok,
        output_tokens_per_call=args.output_tok,
    ))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Esercitazione Giorno 3 — mattino: supervisor + 3 agenti specializzati ITSM "
            "con LangGraph, handoff loggati e stima costi."
        )
    )

    subparsers = parser.add_subparsers(required=True)

    examples_parser = subparsers.add_parser("examples")
    examples_parser.set_defaults(func=cmd_examples)

    manual_parser = subparsers.add_parser("manual")
    manual_parser.add_argument("query", type=str)
    manual_parser.add_argument(
        "--fast",
        action="store_true",
        help=(
            "Usa routing e tool deterministici per ridurre chiamate LLM. "
            "Utile per demo e quote free-tier basse."
        ),
    )
    manual_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Stampa handoffs e trace completi a terminale oltre a salvarli su JSON.",
    )
    manual_parser.add_argument(
        "--trace-dir",
        type=str,
        default=None,
        help="Cartella dove salvare il JSON completo della run. Default: runs/day3_multiagent/",
    )
    manual_parser.set_defaults(func=cmd_manual)

    graph_parser = subparsers.add_parser("graph")
    graph_parser.add_argument("query", type=str)
    graph_parser.add_argument("--thread-id", type=str, default="demo-day3-001")
    graph_parser.add_argument(
        "--fast",
        action="store_true",
        help=(
            "Usa routing e tool deterministici per ridurre chiamate LLM. "
            "Utile per demo e quote free-tier basse."
        ),
    )
    graph_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Stampa handoffs e trace completi a terminale oltre a salvarli su JSON.",
    )
    graph_parser.add_argument(
        "--trace-dir",
        type=str,
        default=None,
        help="Cartella dove salvare il JSON completo della run. Default: runs/day3_multiagent/",
    )
    graph_parser.set_defaults(func=cmd_graph)

    cost_parser = subparsers.add_parser("cost-estimate")
    cost_parser.add_argument("--rounds", type=int, default=5)
    cost_parser.add_argument("--agents", type=int, default=4)
    cost_parser.add_argument("--input-tok", type=int, default=800)
    cost_parser.add_argument("--output-tok", type=int, default=200)
    cost_parser.set_defaults(func=cmd_cost)

    cost_run_parser = subparsers.add_parser("cost-from-run")
    cost_run_parser.add_argument("path", type=str)
    cost_run_parser.set_defaults(func=cmd_cost_from_run)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()