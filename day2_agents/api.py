"""
day2_agents/api.py

REST API + Server-Sent Events (SSE) per itsm_agent.py.

Ogni endpoint di streaming apre una connessione SSE e manda al client
un evento per ogni trace generata dall'agente, in real-time.

Avvio:
    cd genai-course
    uvicorn day2_agents.api:app --reload --port 8000

Endpoint:
    GET  /examples                          → lista prompt di esempio
    POST /setup                             → setup-rag-data [--ingest]
    POST /manual                            → run sincrono, ritorna {answer, traces}
    POST /graph                             → run sincrono, ritorna {answer, traces}
    GET  /manual/stream?query=...           → SSE streaming trace + done finale
    GET  /graph/stream?query=...            → SSE streaming trace + done finale

Formato eventi SSE:
    event: trace
    data: {"step":1,"event":"llm_response","text":"...","timestamp":...}

    event: done
    data: {"answer":"...","total_traces":4}

    event: error
    data: {"detail":"..."}

Test rapido:
    curl http://localhost:8000/examples
    curl -X POST http://localhost:8000/manual \\
         -H "Content-Type: application/json" \\
         -d '{"query":"Mostrami INC-1002 e calcola lo SLA."}'
    curl -N "http://localhost:8000/manual/stream?query=Mostrami+INC-1002+e+calcola+lo+SLA."
    curl -N "http://localhost:8000/graph/stream?query=Analizza+INC-1002%2C+calcola+SLA+e+proponi+l%27azione.&auto_decision=approve"
"""

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
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Path setup: garantisce che `day2_agents` e il project root siano importabili
# sia quando si lancia con `uvicorn day2_agents.api:app` sia direttamente.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent.resolve()
_PROJECT_ROOT = _HERE.parent
for _p in [str(_HERE), str(_PROJECT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Import da itsm_agent — vengono importati dopo il path setup.
# ---------------------------------------------------------------------------
from itsm_agent import (  # noqa: E402
    EXAMPLE_PROMPTS,
    _trace_callback,
    extract_text_content,
    run_graph_agent,
    run_real_agent,
    setup_rag_data,
    resume_graph_agent,
)

# ---------------------------------------------------------------------------
# App FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ITSM Agent API",
    description=(
        "Espone i comandi di itsm_agent.py via REST + Server-Sent Events. "
        "Gli endpoint /stream mandano ogni evento di trace in real-time appena creato."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread pool dedicato per i run bloccanti dell'agente.
# Gli agenti usano time.sleep() e chiamate di rete, quindi non possono girare
# direttamente nell'event loop asyncio — vanno in un thread separato.
_executor = ThreadPoolExecutor(max_workers=4)

# ---------------------------------------------------------------------------
# Modelli di input/output
# ---------------------------------------------------------------------------


class ManualRequest(BaseModel):
    query: str
    max_iter: int = 5


class GraphRequest(BaseModel):
    query: str
    thread_id: str = "demo-itsm-001"
    auto_decision: Optional[str] = None  # "approve" | "reject" | None


class SetupRequest(BaseModel):
    ingest: bool = True


# ---------------------------------------------------------------------------
# Utilità SSE
# ---------------------------------------------------------------------------


def _sse_event(data: Any, event: str = "message") -> str:
    """Formatta un evento SSE secondo RFC 8895."""
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


# ---------------------------------------------------------------------------
# Generatori SSE per manual e graph
# ---------------------------------------------------------------------------


async def _stream_manual(query: str, max_iter: int) -> AsyncGenerator[str, None]:
    """
    Esegue run_real_agent in un thread e manda ogni trace come evento SSE
    appena viene generata, senza aspettare la fine del run.

    Meccanismo:
    1. Crea una asyncio.Queue condivisa tra il thread e l'async generator.
    2. Definisce on_event() che fa queue.put_nowait() in modo thread-safe
       tramite loop.call_soon_threadsafe().
    3. Imposta _trace_callback nel context copiato (copy_context) prima di
       passarlo all'executor — in questo modo il thread dell'agente vede il
       callback senza modifiche alle sue firme.
    4. Consuma la queue yield-ando eventi SSE finché non arriva il SENTINEL.
    5. Al termine emette l'evento "done" con la risposta finale.
    """
    loop: AbstractEventLoop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    SENTINEL = object()

    # Callback chiamato da append_trace nel thread dell'agente.
    def on_event(event_dict: dict) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, event_dict)

    # Copia il context dell'async handler e imposta il callback.
    # Il thread erediterà questo context tramite ctx.run().
    token = _trace_callback.set(on_event)
    ctx = copy_context()
    _trace_callback.reset(token)  # ripristina nel context corrente

    # Wrapper che esegue l'agente e poi segnala la fine con SENTINEL.
    def run_agent():
        try:
            return ctx.run(run_real_agent, query, max_iter)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, SENTINEL)

    future = loop.run_in_executor(_executor, run_agent)

    # Consuma la queue e manda gli eventi SSE uno per uno.
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
        answer, traces = await future
    except Exception as exc:
        yield _sse_event({"detail": str(exc)}, event="error")
        return

    yield _sse_event(
        {"answer": answer, "total_traces": len(traces)},
        event="done",
    )


async def _stream_graph(
    query: str,
    thread_id: str,
    auto_decision: Optional[str],
) -> AsyncGenerator[str, None]:
    """
    Versione SSE per run_graph_agent.
    run_graph_agent ritorna un dict (stato LangGraph) invece di (answer, traces).
    La risposta finale viene estratta dall'ultimo messaggio nello stato.
    """
    loop: AbstractEventLoop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    SENTINEL = object()

    def on_event(event_dict: dict) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, event_dict)

    token = _trace_callback.set(on_event)
    ctx = copy_context()
    _trace_callback.reset(token)

    def run_agent():
        try:
            return ctx.run(run_graph_agent, query, thread_id, auto_decision)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, SENTINEL)

    future = loop.run_in_executor(_executor, run_agent)

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
        result = await future
    except Exception as exc:
        yield _sse_event({"detail": str(exc)}, event="error")
        return

    if isinstance(result, dict) and result.get("__interrupt__"):
        # Il grafo è sospeso: notifica il client con i dati dell'azione critica
        # e il thread_id necessario per riprendere.
        interrupts = result.get("__interrupt__", [])
        pending = interrupts[0].value if interrupts else {}
        yield _sse_event(
            {
                "thread_id": thread_id,
                "pending_action": pending,
                "message": (
                    "Il grafo richiede approvazione umana. "
                    "Chiama GET /graph/resume/stream?thread_id=...&decision=approve|reject"
                ),
            },
            event="interrupted",
        )
        return

    # Caso normale: grafo completato
    answer = ""
    if isinstance(result, dict):
        messages = result.get("messages", [])
        if messages:
            answer = extract_text_content(getattr(messages[-1], "content", ""))
        traces = result.get("traces", [])
    else:
        traces = []

    yield _sse_event(
        {"answer": answer, "total_traces": len(traces)},
        event="done",
    )


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@app.get("/examples", summary="Lista prompt di esempio")
def get_examples() -> List[Dict[str, str]]:
    """Equivalente a: python itsm_agent.py examples"""
    return EXAMPLE_PROMPTS


@app.post("/setup", summary="Setup dati RAG")
def post_setup(req: SetupRequest) -> Dict[str, str]:
    """Equivalente a: python itsm_agent.py setup-rag-data [--ingest]"""
    try:
        setup_rag_data(ingest_now=req.ingest)
        return {"status": "ok", "ingest": str(req.ingest)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/manual", summary="Run agente manuale (sincrono)")
def post_manual(req: ManualRequest) -> Dict[str, Any]:
    """
    Equivalente a: python itsm_agent.py manual "..."

    Ritorna l'intera lista di trace alla fine del run.
    Per ricevere gli eventi in real-time usa GET /manual/stream.
    """
    try:
        answer, traces = run_real_agent(req.query, max_iter=req.max_iter)
        return {"answer": answer, "traces": traces}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/graph", summary="Run agente LangGraph (sincrono)")
def post_graph(req: GraphRequest) -> Dict[str, Any]:
    """
    Equivalente a: python itsm_agent.py graph "..." [--auto-decision approve]

    Ritorna l'intero stato LangGraph alla fine del run.
    Per ricevere gli eventi in real-time usa GET /graph/stream.
    """
    try:
        result = run_graph_agent(
            query=req.query,
            thread_id=req.thread_id,
            auto_decision=req.auto_decision,
        )
        if isinstance(result, dict):
            messages = result.get("messages", [])
            answer = ""
            if messages:
                answer = extract_text_content(getattr(messages[-1], "content", ""))
            return {
                "answer": answer,
                "traces": result.get("traces", []),
                "interrupted": bool(result.get("__interrupt__")),
            }
        return {"raw": str(result)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get(
    "/manual/stream",
    summary="Run agente manuale con SSE streaming",
    response_class=StreamingResponse,
)
async def get_manual_stream(
    query: str = Query(..., description="La domanda da fare all'agente"),
    max_iter: int = Query(5, description="Numero massimo di iterazioni ReAct"),
):
    """
    Equivalente a: python itsm_agent.py manual "..."

    Ogni evento di trace viene mandato al client via SSE appena generato.

    Esempio:
        curl -N "http://localhost:8000/manual/stream?query=Mostrami+INC-1002+e+calcola+lo+SLA."
    """
    return StreamingResponse(
        _stream_manual(query, max_iter),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disabilita buffering in nginx
        },
    )


@app.get(
    "/graph/stream",
    summary="Run agente LangGraph con SSE streaming",
    response_class=StreamingResponse,
)
async def get_graph_stream(
    query: str = Query(..., description="La domanda da fare all'agente"),
    thread_id: str = Query("demo-itsm-001", description="ID del thread LangGraph"),
    auto_decision: Optional[str] = Query(
        None,
        description="Decisione automatica per azioni critiche: 'approve' o 'reject'",
    ),
):
    """
    Equivalente a: python itsm_agent.py graph "..." [--auto-decision approve]

    Ogni evento di trace viene mandato al client via SSE appena generato.
    Se l'agente richiede approvazione umana (HITL), l'evento "done" conterrà
    "interrupted": true.

    Esempio:
        curl -N "http://localhost:8000/graph/stream?query=Analizza+INC-1002&auto_decision=approve"
    """
    return StreamingResponse(
        _stream_graph(query, thread_id, auto_decision),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

async def _stream_graph_resume(
    thread_id: str,
    decision: str,
) -> AsyncGenerator[str, None]:
    """
    Riprende un grafo sospeso dal suo checkpoint.
    Stessa meccanica SSE di _stream_graph, ma chiama resume_graph_agent
    invece di run_graph_agent (non ri-invia initial_state).
    """
    loop: AbstractEventLoop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    SENTINEL = object()

    def on_event(event_dict: dict) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, event_dict)

    token = _trace_callback.set(on_event)
    ctx = copy_context()
    _trace_callback.reset(token)

    def run_resume():
        try:
            return ctx.run(resume_graph_agent, thread_id, decision)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, SENTINEL)

    future = loop.run_in_executor(_executor, run_resume)

    while True:
        try:
            item = await asyncio.wait_for(queue.get(), timeout=180.0)
        except asyncio.TimeoutError:
            yield _sse_event({"detail": "Timeout durante resume"}, event="error")
            return
        if item is SENTINEL:
            break
        yield _sse_event(item, event="trace")

    try:
        result = await future
    except Exception as exc:
        yield _sse_event({"detail": str(exc)}, event="error")
        return

    answer = ""
    if isinstance(result, dict):
        messages = result.get("messages", [])
        if messages:
            answer = extract_text_content(getattr(messages[-1], "content", ""))
        traces = result.get("traces", [])
    else:
        traces = []

    yield _sse_event(
        {"answer": answer, "total_traces": len(traces)},
        event="done",
    )


@app.get(
    "/graph/resume/stream",
    summary="Riprendi un grafo sospeso (HITL) con SSE streaming",
    response_class=StreamingResponse,
)
async def get_graph_resume_stream(
    thread_id: str = Query(..., description="Stesso thread_id della chiamata /graph/stream"),
    decision: str = Query(..., description="'approve' o 'reject'"),
):
    """
    Riprende il grafo dal checkpoint sospeso dopo un interrupt().
    Chiama questo endpoint dopo aver ricevuto event: interrupted da /graph/stream.

    Esempio:
        curl -N "http://localhost:8000/graph/resume/stream?thread_id=demo-itsm-001&decision=approve"
    """
    if decision not in {"approve", "reject"}:
        raise HTTPException(status_code=422, detail="decision deve essere 'approve' o 'reject'")

    return StreamingResponse(
        _stream_graph_resume(thread_id, decision),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )