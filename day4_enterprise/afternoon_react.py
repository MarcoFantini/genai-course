from __future__ import annotations
import sys, io
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

"""
day4_enterprise/lab_pomeriggio_d4.py

╔══════════════════════════════════════════════════════════════════════════════╗
║  Esercitazione Giorno 4 — POMERIGGIO                                        ║
║  Panorama framework: confronto pratico LangGraph vs OpenAI-style ReAct       ║
╚══════════════════════════════════════════════════════════════════════════════╝

Obiettivi:
  1. Implementare lo stesso task ITSM con DUE backend diversi:
       • Backend A "langgraph"  — LangGraph stateful (conoscono già il Day 3)
       • Backend B "react"      — loop ReAct manuale senza framework
  2. Aggiungere Qdrant come vector store (sostituisce la KB dizionario)
  3. Misurare e confrontare: token, latenza, complessità del codice
  4. Applicare il "Decision Framework" delle slide per scegliere il framework giusto

Architettura:

  ┌──────────────────────────────────────────────────────┐
  │                   CompareRunner                       │
  │  esegue la stessa domanda su tutti i backend,         │
  │  misura token/latenza, produce tabella comparativa    │
  └──────────────────────────────────────────────────────┘
         │                              │
  ┌──────▼──────────┐         ┌─────────▼──────────┐
  │  Backend A       │         │   Backend B         │
  │  LangGraph       │         │   ReAct Manual      │
  │  (graph + state) │         │   (loop while)      │
  └──────┬──────────┘         └─────────┬──────────┘
         │                              │
         └─────────────┬────────────────┘
                       │ stessi tool
              ┌────────▼────────┐
              │   Tool Layer    │
              │  lookup_ticket  │
              │  search_qdrant  │
              │  compute_sla    │
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │  Qdrant (mem)   │
              │  vector DB      │
              │  in-process     │
              └─────────────────┘

Comandi:
    python day4_enterprise/lab_pomeriggio_d4.py qdrant-demo          # popola e query Qdrant
    python day4_enterprise/lab_pomeriggio_d4.py langgraph "Domanda"  # backend LangGraph
    python day4_enterprise/lab_pomeriggio_d4.py react "Domanda"      # backend ReAct
    python day4_enterprise/lab_pomeriggio_d4.py compare "Domanda"    # entrambi + tabella
    python day4_enterprise/lab_pomeriggio_d4.py langgraph "Domanda" --fast
    python day4_enterprise/lab_pomeriggio_d4.py decision-matrix      # guida alla scelta
    python day4_enterprise/lab_pomeriggio_d4.py examples

Variabili .env:
    GOOGLE_API_KEY=...
    GEMINI_MODEL=gemini-2.5-flash
    QDRANT_URL=          # lasciare vuoto per in-memory (default)
    QDRANT_API_KEY=      # opzionale per cloud.qdrant.io
    MIN_SECONDS_BETWEEN_MODEL_CALLS=15
    PRICE_INPUT_PER_1M=0.10
    PRICE_OUTPUT_PER_1M=0.40

Nota didattica:
    Il backend B (ReAct manuale) mostra esattamente cosa LangGraph automatizza:
    il loop observation → thought → action. Confrontando i due, capisci il valore
    reale di un framework vs la semplicità di codice from-scratch.

    Qdrant sostituisce il dict KB: ora la ricerca è semantica (coseno tra embeddings),
    non esatta. Con filtri JSON si può filtrare per priority/status prima del ranking.
"""

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict

from dotenv import load_dotenv

# ── import opzionali ────────────────────────────────────────────────────────────

try:
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    print("[WARN] pydantic non installato. pip install pydantic")

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
    from langchain_core.tools import tool, BaseTool
    from langchain_core.utils.function_calling import convert_to_openai_function
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("[WARN] langchain-google-genai non installato.")

try:
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    from langgraph.checkpoint.memory import MemorySaver
    from typing_extensions import Annotated
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("[INFO] langgraph non installato. Backend A disabilitato. pip install langgraph")
    def add_messages(x: Any) -> Any: return x  # type: ignore
    Annotated = None  # type: ignore

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, VectorParams, PointStruct, Filter,
        FieldCondition, MatchValue, SearchRequest,
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    print("[INFO] qdrant-client non installato. pip install qdrant-client")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


# ── config ─────────────────────────────────────────────────────────────────────

load_dotenv()

BASE_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = BASE_DIR.parent if BASE_DIR.name == "day4_enterprise" else BASE_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RUNS_DIR = BASE_DIR / "runs" / "day4_pomeriggio"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
QDRANT_URL = os.getenv("QDRANT_URL", "")  # vuoto = in-memory
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
PRICE_IN = float(os.getenv("PRICE_INPUT_PER_1M", "0.10"))
PRICE_OUT = float(os.getenv("PRICE_OUTPUT_PER_1M", "0.40"))
MIN_DELAY = float(os.getenv("MIN_SECONDS_BETWEEN_MODEL_CALLS", "15"))

_last_call_ts: float = 0.0


def _rate_limit():
    global _last_call_ts
    wait = MIN_DELAY - (time.monotonic() - _last_call_ts)
    if wait > 0:
        print(f"  [rate-limit] attendo {round(wait,1)}s...")
        time.sleep(wait)
    _last_call_ts = time.monotonic()


def _cost(ti: int, to: int) -> float:
    return ti * PRICE_IN / 1_000_000 + to * PRICE_OUT / 1_000_000


# =============================================================================
# DOMAIN DATA — stesso dominio ITSM dei giorni precedenti
# =============================================================================

TICKETS: Dict[str, Dict[str, Any]] = {
    "INC-1001": {
        "id": "INC-1001", "priority": "P2", "status": "Open",
        "title": "VPN aziendale non raggiungibile da sede Milano",
        "description": "50 utenti non si connettono alla VPN. Impatto: lavoro da remoto bloccato.",
        "sla_hours": 4, "domain": "network",
    },
    "INC-1002": {
        "id": "INC-1002", "priority": "P1", "status": "Open",
        "title": "Database Oracle produzione non risponde",
        "description": "DB principale down. Tutti i servizi applicativi bloccati. Revenue 10k€/h.",
        "sla_hours": 1, "domain": "database",
    },
    "INC-1003": {
        "id": "INC-1003", "priority": "P3", "status": "In Progress",
        "title": "Stampante ufficio 3° piano offline",
        "description": "HP offline. Workaround: usare quella al 2° piano.",
        "sla_hours": 8, "domain": "hardware",
    },
    "INC-1004": {
        "id": "INC-1004", "priority": "P2", "status": "Open",
        "title": "Sistema di autenticazione SSO intermittente",
        "description": "Utenti riportano errori di login ogni ~15 minuti. SAML assertion fallisce.",
        "sla_hours": 4, "domain": "security",
    },
}

KB_DOCS = [
    {
        "id": "KB-001",
        "title": "Policy P1 — Incident Critico",
        "content": "Incident P1: escalation immediata. SLA 1 ora. Notifica obbligatoria CTO e manager entro 15 minuti. War room attivata entro 30 minuti. Post-mortem obbligatorio entro 48 ore.",
        "category": "policy", "priority_applies": "P1",
    },
    {
        "id": "KB-002",
        "title": "Policy P2 — Incident Alto",
        "content": "Incident P2: presa in carico entro 30 minuti. SLA 4 ore. Notifica team lead entro 1 ora. Update al richiedente ogni 2 ore.",
        "category": "policy", "priority_applies": "P2",
    },
    {
        "id": "KB-003",
        "title": "Procedura VPN — Troubleshooting",
        "content": "1. Verifica servizio VPN sul server gateway. 2. Controlla certificati SSL/TLS. 3. Riavvia il gateway se necessario. 4. Apri bridge di emergenza se downtime >30 min. 5. Notifica networking team.",
        "category": "procedure", "priority_applies": "P2",
    },
    {
        "id": "KB-004",
        "title": "Procedura Database Oracle — Recovery",
        "content": "1. Controlla alert log Oracle. 2. Verifica spazio tablespace. 3. Controlla processi bloccanti con V$SESSION. 4. Esegui recover automatico se archivelog mode attivo. 5. Contatta DBA on-call.",
        "category": "procedure", "priority_applies": "P1",
    },
    {
        "id": "KB-005",
        "title": "SLA Definition — Livelli di Servizio",
        "content": "P1: 1h risoluzione, 15min risposta. P2: 4h risoluzione, 30min risposta. P3: 8h risoluzione, 2h risposta. P4: 24h risoluzione, 4h risposta. Outside SLA: penali contrattuali.",
        "category": "sla",
    },
    {
        "id": "KB-006",
        "title": "Procedura SSO — Troubleshooting",
        "content": "1. Verifica IdP Okta/Azure AD availability. 2. Controlla SAML metadata scaduti. 3. Verifica clock sync tra SP e IdP (SAML sensibile a ±5min). 4. Rollback a versione precedente se recente deploy.",
        "category": "procedure", "priority_applies": "P2",
    },
]


# =============================================================================
# QDRANT VECTOR STORE — in-memory o cloud.qdrant.io
# =============================================================================

COLLECTION_NAME = "itsm_kb"
VECTOR_DIM = 128  # embedding ridotto (toy embedding deterministico)


def _toy_embed(text: str) -> List[float]:
    """
    Embedding deterministico (toy) senza API esterna.
    In produzione: google.generativeai.embed_content() o text-embedding-3-small.

    Funziona così: crea un vettore di 128 dimensioni sommando i valori ASCII
    delle parole chiave del testo, con un hash deterministico.
    Non è semantico, ma permette di dimostrare Qdrant senza quota embedding.
    """
    import hashlib
    vec = [0.0] * VECTOR_DIM
    words = text.lower().split()
    for i, w in enumerate(words[:64]):
        h = int(hashlib.md5(w.encode()).hexdigest(), 16)
        for j in range(4):
            idx = (h >> (j * 8)) % VECTOR_DIM
            vec[idx] += 1.0 / (i + 1)
    # normalizza
    norm = sum(v * v for v in vec) ** 0.5
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


class QdrantKB:
    """
    Knowledge Base con Qdrant come backend vettoriale.

    In-memory: QdrantClient(":memory:") — nessun server necessario.
    Cloud: QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    Nota didattica:
      La ricerca vettoriale non è "trova la parola esatta" ma
      "trova il documento semanticamente più simile". Con embeddings reali
      (Google/OpenAI) cattura sinonimi, concetti correlati e parafrasi.
      Con il toy embedding di questo lab è solo un esempio dell'API Qdrant.
    """

    def __init__(self):
        self._client: Optional[Any] = None
        self._available = QDRANT_AVAILABLE

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not QDRANT_AVAILABLE:
            return None
        if QDRANT_URL:
            self._client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)
            print(f"  [Qdrant] Connesso a: {QDRANT_URL}")
        else:
            self._client = QdrantClient(":memory:")
            print("  [Qdrant] Modalità in-memory (nessun server necessario)")
        return self._client

    def setup(self) -> bool:
        """Crea la collection e indicizza i documenti KB."""
        client = self._get_client()
        if client is None:
            return False
        try:
            # Ricrea la collection
            existing = [c.name for c in client.get_collections().collections]
            if COLLECTION_NAME in existing:
                client.delete_collection(COLLECTION_NAME)

            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            )

            # Indicizza i documenti
            points = []
            for i, doc in enumerate(KB_DOCS):
                text_for_embed = f"{doc['title']} {doc['content']}"
                vec = _toy_embed(text_for_embed)
                points.append(PointStruct(
                    id=i,
                    vector=vec,
                    payload={
                        "doc_id": doc["id"],
                        "title": doc["title"],
                        "content": doc["content"],
                        "category": doc.get("category", ""),
                        "priority_applies": doc.get("priority_applies", ""),
                    },
                ))

            client.upsert(collection_name=COLLECTION_NAME, points=points)
            print(f"  [Qdrant] Indicizzati {len(points)} documenti in '{COLLECTION_NAME}'")
            return True
        except Exception as exc:
            print(f"  [Qdrant] Errore setup: {exc}")
            return False

    def search(
        self,
        query: str,
        top_k: int = 3,
        filter_priority: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Ricerca semantica nella KB.

        filter_priority: se impostato (es. "P1"), filtra solo documenti
        applicabili a quella priorità. Mostra il filtraggio nativo di Qdrant.
        """
        client = self._get_client()
        if client is None:
            return self._fallback_search(query, top_k)

        try:
            vec = _toy_embed(query)

            # Costruisci filtro Qdrant (opzionale)
            qdrant_filter = None
            if filter_priority:
                qdrant_filter = Filter(
                    should=[
                        FieldCondition(
                            key="priority_applies",
                            match=MatchValue(value=filter_priority),
                        ),
                        FieldCondition(
                            key="priority_applies",
                            match=MatchValue(value=""),  # documenti universali
                        ),
                    ]
                )

            results = client.search(
                collection_name=COLLECTION_NAME,
                query_vector=vec,
                limit=top_k,
                query_filter=qdrant_filter,
                with_payload=True,
            )

            return [
                {
                    "id": r.payload.get("doc_id", ""),
                    "title": r.payload.get("title", ""),
                    "content": r.payload.get("content", ""),
                    "score": round(r.score, 4),
                    "source": "qdrant",
                }
                for r in results
            ]
        except Exception as exc:
            print(f"  [Qdrant] Errore ricerca: {exc}")
            return self._fallback_search(query, top_k)

    def _fallback_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Ricerca keyword semplice quando Qdrant non è disponibile."""
        q_words = set(query.lower().split())
        scored = []
        for doc in KB_DOCS:
            score = sum(1 for w in q_words if w in doc["content"].lower() or w in doc["title"].lower())
            scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {"id": d["id"], "title": d["title"], "content": d["content"], "score": s, "source": "fallback"}
            for s, d in scored[:top_k]
        ]


# Singleton KB
qdrant_kb = QdrantKB()


# =============================================================================
# TOOL LAYER — stessi tool, usati da entrambi i backend
# =============================================================================

def tool_lookup_ticket(ticket_id: str) -> Dict[str, Any]:
    """Recupera i dettagli di un ticket ITSM dal sistema."""
    t = TICKETS.get(ticket_id.upper())
    if t:
        return {"success": True, "ticket": t}
    return {"success": False, "error": f"Ticket {ticket_id} non trovato", "available": list(TICKETS.keys())}


def tool_search_kb(query: str, top_k: int = 3, priority: Optional[str] = None) -> Dict[str, Any]:
    """Ricerca nella Knowledge Base ITSM (Qdrant vector search)."""
    results = qdrant_kb.search(query, top_k=top_k, filter_priority=priority)
    return {"success": True, "results": results, "count": len(results)}


def tool_compute_sla(ticket_id: str) -> Dict[str, Any]:
    """Calcola lo stato SLA di un ticket."""
    t = TICKETS.get(ticket_id.upper())
    if not t:
        return {"success": False, "error": f"Ticket {ticket_id} non trovato"}
    import random
    # Simula tempo trascorso
    elapsed_h = round(random.uniform(0.2, t["sla_hours"] * 1.5), 2)
    remaining_h = max(0.0, t["sla_hours"] - elapsed_h)
    status = "in_sla" if remaining_h > 0 else "sla_breach"
    return {
        "success": True,
        "ticket_id": ticket_id,
        "priority": t["priority"],
        "sla_hours": t["sla_hours"],
        "elapsed_hours": elapsed_h,
        "remaining_hours": round(remaining_h, 2),
        "status": status,
        "breach": status == "sla_breach",
    }


def tool_list_open_tickets(priority: Optional[str] = None) -> Dict[str, Any]:
    """Elenca i ticket aperti, filtrando opzionalmente per priorità."""
    results = [
        t for t in TICKETS.values()
        if t["status"] in ("Open", "In Progress")
        and (priority is None or t["priority"] == priority)
    ]
    return {"success": True, "tickets": results, "count": len(results)}


# Registry tool disponibili
TOOL_REGISTRY: Dict[str, Any] = {
    "lookup_ticket": {
        "fn": tool_lookup_ticket,
        "description": "Recupera i dettagli di un ticket ITSM dato il suo ID (es. INC-1002)",
        "params": {"ticket_id": "string"},
    },
    "search_kb": {
        "fn": tool_search_kb,
        "description": "Ricerca nella Knowledge Base ITSM: policy, SLA, procedure",
        "params": {"query": "string", "top_k": "int (default 3)", "priority": "string opzionale"},
    },
    "compute_sla": {
        "fn": tool_compute_sla,
        "description": "Calcola lo stato SLA del ticket: in_sla o sla_breach",
        "params": {"ticket_id": "string"},
    },
    "list_open_tickets": {
        "fn": tool_list_open_tickets,
        "description": "Elenca i ticket aperti. Parametro priority opzionale (P1/P2/P3)",
        "params": {"priority": "string opzionale"},
    },
}


# =============================================================================
# BACKEND A — LangGraph stateful (conosci già questo dal Day 3)
# =============================================================================

class LangGraphState(TypedDict):
    """Stato del grafo LangGraph."""
    messages: Any  # list con add_messages reducer
    task_id: str
    tokens_in: int
    tokens_out: int
    tool_calls_log: List[dict]
    fast_mode: bool


def _make_langgraph_backend() -> Optional[Any]:
    """Costruisce il grafo LangGraph per il backend A."""
    if not LANGGRAPH_AVAILABLE or not LANGCHAIN_AVAILABLE:
        return None

    # Costruisci tool LangChain da TOOL_REGISTRY
    @tool
    def lookup_ticket(ticket_id: str) -> str:
        """Recupera i dettagli di un ticket ITSM dato il suo ID."""
        return json.dumps(tool_lookup_ticket(ticket_id), ensure_ascii=False)

    @tool
    def search_kb(query: str, top_k: int = 3) -> str:
        """Ricerca nella Knowledge Base ITSM: policy, SLA, procedure."""
        return json.dumps(tool_search_kb(query, top_k=top_k), ensure_ascii=False)

    @tool
    def compute_sla(ticket_id: str) -> str:
        """Calcola lo stato SLA del ticket."""
        return json.dumps(tool_compute_sla(ticket_id), ensure_ascii=False)

    @tool
    def list_open_tickets(priority: str = "") -> str:
        """Elenca i ticket aperti. priority opzionale: P1, P2, P3."""
        return json.dumps(tool_list_open_tickets(priority or None), ensure_ascii=False)

    tools = [lookup_ticket, search_kb, compute_sla, list_open_tickets]

    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.1,
    ).bind_tools(tools)

    tool_map = {t.name: t for t in tools}

    # ── nodi ──────────────────────────────────────────────────────────────────

    SYSTEM = """Sei un assistente ITSM esperto. Rispondi in italiano.
Usa i tool disponibili per: recuperare ticket, cercare policy/procedure, calcolare SLA.
Fornisci risposte precise con dati concreti. Cita sempre ticket ID e KB ID rilevanti."""

    def agent_node(state: LangGraphState) -> Dict[str, Any]:
        msgs = state["messages"]
        if not msgs or not isinstance(msgs[0], SystemMessage):
            msgs = [SystemMessage(content=SYSTEM)] + list(msgs)
        _rate_limit()
        resp = llm.invoke(msgs)
        usage = getattr(resp, "usage_metadata", {}) or {}
        ti = usage.get("input_tokens", 0) or 0
        to = usage.get("output_tokens", 0) or 0
        return {
            "messages": [resp],
            "tokens_in": state["tokens_in"] + ti,
            "tokens_out": state["tokens_out"] + to,
        }

    def tool_node(state: LangGraphState) -> Dict[str, Any]:
        last = state["messages"][-1]
        tool_calls_log = []
        results = []
        for tc in (last.tool_calls or []):
            t = tool_map.get(tc["name"])
            if t:
                out = t.invoke(tc["args"])
                results.append(ToolMessage(content=str(out), tool_call_id=tc["id"]))
                tool_calls_log.append({"tool": tc["name"], "args": tc["args"], "result": str(out)[:200]})
        return {
            "messages": results,
            "tool_calls_log": state.get("tool_calls_log", []) + tool_calls_log,
        }

    def router(state: LangGraphState) -> str:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    # ── grafo ─────────────────────────────────────────────────────────────────

    from langgraph.graph.message import add_messages as _am
    from typing_extensions import Annotated as _Ann

    # Usiamo un TypedDict locale con Annotated per il reducer
    class _St(TypedDict):
        messages: _Ann[list, _am]
        task_id: str
        tokens_in: int
        tokens_out: int
        tool_calls_log: List[dict]
        fast_mode: bool

    g = StateGraph(_St)
    g.add_node("agent", agent_node)
    g.add_node("tools", tool_node)
    g.set_entry_point("agent")
    g.add_conditional_edges("agent", router, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")

    memory = MemorySaver()
    return g.compile(checkpointer=memory)


_langgraph_app = None

def _get_langgraph_app():
    global _langgraph_app
    if _langgraph_app is None:
        _langgraph_app = _make_langgraph_backend()
    return _langgraph_app


def run_langgraph(question: str, thread_id: str, fast: bool = False) -> Dict[str, Any]:
    """Esegui il task con il backend LangGraph."""
    t0 = time.monotonic()

    if fast or not LANGGRAPH_AVAILABLE or not GOOGLE_API_KEY:
        return _mock_answer(question, "langgraph")

    app = _get_langgraph_app()
    if app is None:
        print("  [Backend A] LangGraph non disponibile — uso mock")
        return _mock_answer(question, "langgraph")

    cfg = {"configurable": {"thread_id": thread_id}}
    state = app.invoke(
        {
            "messages": [HumanMessage(content=question)],
            "task_id": thread_id,
            "tokens_in": 0,
            "tokens_out": 0,
            "tool_calls_log": [],
            "fast_mode": fast,
        },
        config=cfg,
    )

    duration = (time.monotonic() - t0) * 1000
    last_msg = state["messages"][-1]
    answer = str(last_msg.content) if hasattr(last_msg, "content") else str(last_msg)

    return {
        "answer": answer,
        "backend": "langgraph",
        "tokens_in": state.get("tokens_in", 0),
        "tokens_out": state.get("tokens_out", 0),
        "tokens_total": state.get("tokens_in", 0) + state.get("tokens_out", 0),
        "cost_usd": _cost(state.get("tokens_in", 0), state.get("tokens_out", 0)),
        "tool_calls": len(state.get("tool_calls_log", [])),
        "tool_calls_log": state.get("tool_calls_log", []),
        "duration_ms": round(duration, 1),
    }


# =============================================================================
# BACKEND B — ReAct loop manuale (senza framework)
# =============================================================================

# Nota didattica:
#   Questo è esattamente ciò che fa LangGraph internamente, solo:
#   - più codice boilerplate
#   - nessun checkpoint/persistenza
#   - nessuna visualizzazione del grafo
#   - nessun routing condizionale built-in
#   Ma è utile per capire il pattern base prima di usare un framework.

REACT_SYSTEM = """Sei un assistente ITSM esperto che usa il pattern ReAct (Reasoning + Acting).

Hai accesso a questi tool:
{tools_description}

Per ogni step, usa ESATTAMENTE questo formato JSON:
{{
  "thought": "ragionamento interno su cosa fare",
  "action": "nome_tool o 'final_answer'",
  "action_input": {{parametri del tool}} o {{"answer": "risposta finale"}}
}}

Se hai tutte le informazioni, usa action="final_answer".
Rispondi sempre in italiano. Cita ticket ID e KB ID rilevanti."""


def _react_tools_description() -> str:
    lines = []
    for name, spec in TOOL_REGISTRY.items():
        lines.append(f"• {name}: {spec['description']} | params: {spec['params']}")
    return "\n".join(lines)


def run_react(question: str, thread_id: str, fast: bool = False) -> Dict[str, Any]:
    """
    Backend B: loop ReAct manuale.

    Implementa Reasoning-Acting senza framework:
      1. LLM ragiona e sceglie un tool (o produce la risposta finale)
      2. Eseguiamo il tool
      3. Aggiungiamo il risultato al contesto
      4. Ripetiamo fino a final_answer o max_iterations
    """
    t0 = time.monotonic()

    if fast or not LANGCHAIN_AVAILABLE or not GOOGLE_API_KEY:
        return _mock_answer(question, "react")

    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.1,
    )

    system = REACT_SYSTEM.format(tools_description=_react_tools_description())
    history: List[Any] = [
        SystemMessage(content=system),
        HumanMessage(content=question),
    ]

    tokens_in_total, tokens_out_total = 0, 0
    tool_calls_log: List[dict] = []
    max_iter = 6
    answer = ""

    for step in range(max_iter):
        _rate_limit()

        response = llm.invoke(history)
        usage = getattr(response, "usage_metadata", {}) or {}
        tokens_in_total += usage.get("input_tokens", 0) or 0
        tokens_out_total += usage.get("output_tokens", 0) or 0

        raw = str(response.content).strip()

        # Estrai JSON dal testo (il modello può aggiungere markdown)
        json_match = re.search(r"\{[\s\S]*\}", raw)
        if not json_match:
            # Risposta libera (non JSON) → trattiamo come risposta finale
            answer = raw
            break

        try:
            parsed = json.loads(json_match.group())
        except json.JSONDecodeError:
            answer = raw
            break

        thought = parsed.get("thought", "")
        action = parsed.get("action", "")
        action_input = parsed.get("action_input", {})

        print(f"  [ReAct step {step+1}] thought: {thought[:80]}...")
        print(f"  [ReAct step {step+1}] action: {action}")

        if action == "final_answer":
            answer = action_input.get("answer", raw)
            break

        # Esegui tool
        tool_spec = TOOL_REGISTRY.get(action)
        if not tool_spec:
            tool_result = {"error": f"Tool '{action}' non trovato. Disponibili: {list(TOOL_REGISTRY.keys())}"}
        else:
            try:
                tool_result = tool_spec["fn"](**action_input)
            except Exception as exc:
                tool_result = {"error": str(exc)}

        tool_calls_log.append({
            "step": step + 1,
            "thought": thought[:150],
            "tool": action,
            "args": action_input,
            "result": str(tool_result)[:300],
        })

        # Aggiorna contesto
        history.append(AIMessage(content=raw))
        history.append(HumanMessage(
            content=f"Tool '{action}' result:\n{json.dumps(tool_result, ensure_ascii=False)}\n\nContinua con il prossimo step ReAct."
        ))

    duration = (time.monotonic() - t0) * 1000

    return {
        "answer": answer or "Risposta non disponibile",
        "backend": "react",
        "tokens_in": tokens_in_total,
        "tokens_out": tokens_out_total,
        "tokens_total": tokens_in_total + tokens_out_total,
        "cost_usd": _cost(tokens_in_total, tokens_out_total),
        "tool_calls": len(tool_calls_log),
        "tool_calls_log": tool_calls_log,
        "duration_ms": round(duration, 1),
        "steps": len(tool_calls_log),
    }


# =============================================================================
# MOCK — fast mode per demo
# =============================================================================

import re as _re

def _mock_answer(question: str, backend: str) -> Dict[str, Any]:
    """Risposta deterministica per demo senza LLM."""
    q = question.lower()
    if "inc-1002" in q or "oracle" in q or "database" in q:
        answer = (
            "[INC-1002] Database Oracle produzione non risponde — P1 | Open\n"
            "SLA: 1 ora | Domain: database\n"
            "📚 KB-004: Procedura Recovery — controlla alert log Oracle, verifica tablespace, "
            "contatta DBA on-call.\n📋 KB-001: Policy P1 — war room entro 30min, notifica CTO."
        )
        sources = ["INC-1002", "KB-004", "KB-001"]
    elif "inc-1001" in q or "vpn" in q:
        answer = (
            "[INC-1001] VPN aziendale non raggiungibile — P2 | Open\n"
            "SLA: 4 ore\n"
            "📚 KB-003: Procedura VPN — verifica servizio gateway, controlla certificati SSL."
        )
        sources = ["INC-1001", "KB-003"]
    elif "p1" in q or "critico" in q or "priorit" in q:
        answer = (
            "Policy P1 (KB-001): Escalation immediata. SLA 1 ora. Notifica CTO entro 15 min. "
            "War room entro 30 min. Post-mortem obbligatorio entro 48 ore."
        )
        sources = ["KB-001"]
    elif "sla" in q:
        answer = "SLA (KB-005): P1=1h, P2=4h, P3=8h, P4=24h. Tempi di risposta: P1=15min, P2=30min."
        sources = ["KB-005"]
    else:
        answer = f"[MOCK-{backend}] Risposta per: '{question[:60]}'. Avvia senza --fast per risposta LLM."
        sources = []

    return {
        "answer": answer,
        "backend": f"{backend}_mock",
        "tokens_in": 120,
        "tokens_out": 90,
        "tokens_total": 210,
        "cost_usd": _cost(120, 90),
        "tool_calls": 2,
        "tool_calls_log": [
            {"step": 1, "tool": "lookup_ticket", "args": {}, "result": "mock"},
            {"step": 2, "tool": "search_kb", "args": {"query": question[:30]}, "result": "mock"},
        ],
        "duration_ms": 12.0,
        "mode": "mock",
    }


# =============================================================================
# COMPARE RUNNER — confronta entrambi i backend
# =============================================================================

def run_compare(question: str, fast: bool = False) -> Dict[str, Any]:
    """Esegui la stessa domanda su entrambi i backend e confronta i risultati."""
    print(f"\n{'═'*65}")
    print(f"  CONFRONTO BACKEND — {'FAST MODE' if fast else 'LLM mode'}")
    print(f"  Domanda: {question[:60]}{'...' if len(question)>60 else ''}")
    print(f"{'═'*65}")

    results = {}

    # Backend A: LangGraph
    print("\n▶ Backend A: LangGraph")
    thread_a = f"compare-lg-{uuid.uuid4().hex[:6]}"
    res_a = run_langgraph(question, thread_a, fast=fast)
    results["langgraph"] = res_a
    print(f"  ✓ Completato | tokens: {res_a['tokens_total']} | tool calls: {res_a['tool_calls']} | {res_a['duration_ms']}ms")

    # Backend B: ReAct
    print("\n▶ Backend B: ReAct manuale")
    res_b = run_react(question, f"compare-react-{uuid.uuid4().hex[:6]}", fast=fast)
    results["react"] = res_b
    print(f"  ✓ Completato | tokens: {res_b['tokens_total']} | tool calls: {res_b['tool_calls']} | {res_b['duration_ms']}ms")

    # Tabella comparativa
    print(f"\n{'─'*65}")
    print(f"  {'METRICA':<30} {'LANGGRAPH':>15} {'REACT':>15}")
    print(f"{'─'*65}")
    metrics = [
        ("Token totali", "tokens_total", lambda x: str(x)),
        ("Token input", "tokens_in", lambda x: str(x)),
        ("Token output", "tokens_out", lambda x: str(x)),
        ("Costo (USD)", "cost_usd", lambda x: f"${x:.6f}"),
        ("Tool calls", "tool_calls", lambda x: str(x)),
        ("Latenza (ms)", "duration_ms", lambda x: str(x)),
    ]
    for label, key, fmt in metrics:
        va = fmt(res_a.get(key, 0))
        vb = fmt(res_b.get(key, 0))
        print(f"  {label:<30} {va:>15} {vb:>15}")
    print(f"{'─'*65}")

    print(f"\n📝 RISPOSTA LANGGRAPH:\n{res_a['answer'][:400]}{'...' if len(res_a['answer'])>400 else ''}")
    print(f"\n📝 RISPOSTA REACT:\n{res_b['answer'][:400]}{'...' if len(res_b['answer'])>400 else ''}")

    return results


# =============================================================================
# QDRANT DEMO — mostra le funzionalità vettoriali
# =============================================================================

def cmd_qdrant_demo(args):
    """Demo interattiva delle funzionalità Qdrant."""
    print(f"\n{'═'*65}")
    print("  QDRANT DEMO — Vector Search in azione")
    print(f"{'═'*65}\n")

    if not QDRANT_AVAILABLE:
        print("⚠ qdrant-client non installato: pip install qdrant-client")
        print("  Simulazione con ricerca keyword:\n")

    # Setup KB
    print("① Setup collection e indicizzazione documenti...")
    ok = qdrant_kb.setup()

    demo_queries = [
        ("escalation critica", None),
        ("come recuperare database oracle", None),
        ("policy SLA tempo risposta", "P1"),
        ("vpn troubleshooting", "P2"),
    ]

    print("\n② Ricerche semantiche:")
    for query, prio_filter in demo_queries:
        print(f"\n  Query: '{query}'" + (f" [filtro priority={prio_filter}]" if prio_filter else ""))
        results = qdrant_kb.search(query, top_k=2, filter_priority=prio_filter)
        for r in results:
            print(f"  → [{r['id']}] {r['title']}")
            print(f"    Score: {r['score']} | {r['content'][:80]}...")

    print(f"\n{'─'*65}")
    print("Nota: con embeddings reali (Google/OpenAI) la ricerca semantica cattura")
    print("sinonimi e concetti correlati, non solo keyword esatte.")
    print("Prova: 'server giù' → trova 'Database Oracle non risponde' via similarità semantica.")


# =============================================================================
# DECISION MATRIX — guida alla scelta del framework
# =============================================================================

def cmd_decision_matrix(args):
    """Stampa la matrice decisionale framework (da slide Day 4 Parte 2)."""
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  DECISION FRAMEWORK — Come scegliere il framework agentic       ║
╠══════════════════════════════════════════════════════════════════╣

┌─────────────────────────────────────────────────────────────────┐
│ Domanda                         → Framework suggerito           │
├─────────────────────────────────────────────────────────────────┤
│ Hai bisogno di HITL (human-in-  → LangGraph                    │
│ the-loop) e checkpoint?         │                               │
│                                 │                               │
│ Il task è RAG-heavy con molte   → LlamaIndex (+ LangGraph)     │
│ sorgenti dati eterogenee?       │                               │
│                                 │                               │
│ Vuoi workflow role-based con    → CrewAI                        │
│ agenti specializzati?           │                               │
│                                 │                               │
│ Multi-agent conversazionale     → AutoGen                       │
│ (agenti si parlano tra loro)?   │                               │
│                                 │                               │
│ Ambiente Microsoft .NET/Java?   → Semantic Kernel               │
│                                 │                               │
│ Prototipo rapido, single-agent? → OpenAI Agents SDK / ReAct    │
│                                 │                               │
│ Full managed, compliance AWS?   → Amazon Bedrock Agents        │
│                                 │                               │
│ GCP + Gemini + Search grounding?→ Vertex AI Agent Builder      │
└─────────────────────────────────────────────────────────────────┘

QUESTO LAB confronta:
  • LangGraph  → massima flessibilità, HITL, checkpoint SQLite
  • ReAct loop → minimo codice, nessun framework, comprensione profonda

Risultato atteso: stessa qualità, ma LangGraph ha:
  + checkpoint/persistenza conversazione
  + visualizzazione grafo (langgraph.io)
  + routing condizionale built-in
  + recovery da errori tool call
  - più codice di setup iniziale
""")


# =============================================================================
# CLI
# =============================================================================

def cmd_langgraph(args):
    qdrant_kb.setup()
    thread_id = f"lg-{uuid.uuid4().hex[:6]}"
    print(f"\n▶ Backend A: LangGraph | thread: {thread_id}")
    result = run_langgraph(args.question, thread_id, fast=args.fast)
    _print_result(result)
    _save_run("langgraph", args.question, result)


def cmd_react(args):
    qdrant_kb.setup()
    print(f"\n▶ Backend B: ReAct manuale")
    result = run_react(args.question, f"react-{uuid.uuid4().hex[:6]}", fast=args.fast)
    _print_result(result)
    _save_run("react", args.question, result)


def cmd_compare(args):
    qdrant_kb.setup()
    results = run_compare(args.question, fast=args.fast)
    _save_run("compare", args.question, results)


def _print_result(r: Dict[str, Any]) -> None:
    print(f"\n{'─'*65}")
    print(f"RISPOSTA [{r.get('backend','?')}]:\n{r['answer']}")
    print(f"\nToken: {r.get('tokens_total')} | Tool calls: {r.get('tool_calls')} | "
          f"Costo: ${r.get('cost_usd',0):.6f} | Latenza: {r.get('duration_ms')}ms")
    if r.get("tool_calls_log"):
        print("\nTool calls:")
        for tc in r["tool_calls_log"]:
            print(f"  [{tc.get('step','?')}] {tc.get('tool','?')} → {str(tc.get('result',''))[:80]}...")


def _save_run(mode: str, question: str, result: Any) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = RUNS_DIR / f"{mode}_{ts}.json"
    fname.write_text(json.dumps({
        "mode": mode, "question": question, "result": result,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n📁 Run salvata: {fname}")


def cmd_examples(args):
    print("""
╔══════════════════════════════════════════════════════════════════╗
║  LAB POMERIGGIO — Giorno 4 · Esempi                             ║
╚══════════════════════════════════════════════════════════════════╝

# Demo Qdrant (vector search, nessun LLM necessario)
python day4_enterprise/lab_pomeriggio_d4.py qdrant-demo

# Backend A: LangGraph (framework stateful)
python day4_enterprise/lab_pomeriggio_d4.py langgraph "Analizza INC-1002 e indicami la policy applicabile"
python day4_enterprise/lab_pomeriggio_d4.py langgraph "Quali ticket P1 sono aperti?" --fast

# Backend B: ReAct loop manuale (no framework)
python day4_enterprise/lab_pomeriggio_d4.py react "Qual è lo stato SLA di INC-1001?"

# Confronto side-by-side (CONSIGLIATO per l'esercitazione)
python day4_enterprise/lab_pomeriggio_d4.py compare "Analizza i ticket critici e suggerisci priorità"
python day4_enterprise/lab_pomeriggio_d4.py compare "Quali sono le procedure per un P1 database?" --fast

# Matrice decisionale framework
python day4_enterprise/lab_pomeriggio_d4.py decision-matrix

DOMANDE DI RIFLESSIONE:
  1. Guardando la tabella comparativa: quale backend usa meno token? Perché?
  2. Il ReAct manuale ha più o meno tool calls del LangGraph? Perché?
  3. In quale scenario useresti ReAct invece di LangGraph?
  4. Cosa aggiunge Qdrant rispetto alla ricerca keyword del Day 3?
""")


def main():
    parser = argparse.ArgumentParser(
        description="Day 4 — Lab Pomeriggio: Framework Comparison + Qdrant"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("qdrant-demo", help="Demo vector search con Qdrant")

    p_lg = sub.add_parser("langgraph", help="Backend LangGraph")
    p_lg.add_argument("question")
    p_lg.add_argument("--fast", action="store_true")

    p_re = sub.add_parser("react", help="Backend ReAct manuale")
    p_re.add_argument("question")
    p_re.add_argument("--fast", action="store_true")

    p_cmp = sub.add_parser("compare", help="Confronto side-by-side")
    p_cmp.add_argument("question")
    p_cmp.add_argument("--fast", action="store_true")

    sub.add_parser("decision-matrix", help="Guida alla scelta del framework")
    sub.add_parser("examples", help="Mostra esempi di comandi")

    args = parser.parse_args()
    {
        "qdrant-demo": cmd_qdrant_demo,
        "langgraph": cmd_langgraph,
        "react": cmd_react,
        "compare": cmd_compare,
        "decision-matrix": cmd_decision_matrix,
        "examples": cmd_examples,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
