from __future__ import annotations

import asyncio
import json
import sys
import threading
import uuid
from asyncio import AbstractEventLoop
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

_HERE = Path(__file__).parent.resolve()
_PROJECT_ROOT = _HERE.parent
for _p in [str(_HERE), str(_PROJECT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from supervisor import EXAMPLE_PROMPTS, _hitl_callback, _trace_callback, run_graph, run_manual  # noqa: E402

app = FastAPI(
    title="Day3 Supervisor API",
    description=(
        "Espone gli endpoint manual e graph di day3_multiagent.supervisor "
        "con REST + Server-Sent Events per streaming real-time delle trace."
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

# Registry delle approvazioni HITL pendenti: run_id -> {event, approved}
_pending_approvals: Dict[str, Any] = {}


class ManualRequest(BaseModel):
    query: str
    fast: bool = False


class GraphRequest(BaseModel):
    query: str
    thread_id: str = "demo-day3-001"
    fast: bool = False


def _sse_event(data: Any, event: str = "message") -> str:
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


async def _stream_manual(query: str, fast: bool) -> AsyncGenerator[str, None]:
    loop: AbstractEventLoop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    SENTINEL = object()
    run_id = str(uuid.uuid4())

    def on_trace_event(event_dict: dict) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, event_dict)

    def on_hitl(action_data: dict) -> bool:
        payload = {**action_data, "task_id": run_id}
        loop.call_soon_threadsafe(
            queue.put_nowait,
            {"_sse_event_type": "approval_request", **payload},
        )
        evt = threading.Event()
        _pending_approvals[run_id] = {"event": evt, "approved": False}
        timed_out = not evt.wait(timeout=300.0)  # 5 min timeout
        record = _pending_approvals.pop(run_id, {})
        return False if timed_out else record.get("approved", False)

    token_trace = _trace_callback.set(on_trace_event)
    token_hitl = _hitl_callback.set(on_hitl)
    ctx = copy_context()
    _trace_callback.reset(token_trace)
    _hitl_callback.reset(token_hitl)

    def run_agent() -> Any:
        try:
            return ctx.run(run_manual, query, fast)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, SENTINEL)

    future = loop.run_in_executor(_executor, run_agent)

    while True:
        try:
            # Timeout > HITL wait (300s) per non uscire mentre si aspetta l'umano.
            item = await asyncio.wait_for(queue.get(), timeout=600.0)
        except asyncio.TimeoutError:
            yield _sse_event({"detail": "Timeout: nessun evento dall'agente"}, event="error")
            return

        if item is SENTINEL:
            break

        if isinstance(item, dict) and item.get("_sse_event_type") == "approval_request":
            payload = {k: v for k, v in item.items() if k != "_sse_event_type"}
            yield _sse_event(payload, event="approval_request")
        else:
            yield _sse_event(item, event="trace")

    try:
        answer, state = await future
    except Exception as exc:
        yield _sse_event({"detail": str(exc)}, event="error")
        return

    yield _sse_event(
        {"answer": answer, "total_traces": len(state.get("traces", []))},
        event="done",
    )


async def _stream_graph(
    query: str,
    thread_id: str,
    fast: bool,
) -> AsyncGenerator[str, None]:
    loop: AbstractEventLoop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    SENTINEL = object()
    run_id = f"graph-{thread_id}"

    def on_trace_event(event_dict: dict) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, event_dict)

    def on_hitl(action_data: dict) -> bool:
        payload = {**action_data, "task_id": run_id}
        loop.call_soon_threadsafe(
            queue.put_nowait,
            {"_sse_event_type": "approval_request", **payload},
        )
        evt = threading.Event()
        _pending_approvals[run_id] = {"event": evt, "approved": False}
        timed_out = not evt.wait(timeout=300.0)
        record = _pending_approvals.pop(run_id, {})
        return False if timed_out else record.get("approved", False)

    token_trace = _trace_callback.set(on_trace_event)
    token_hitl = _hitl_callback.set(on_hitl)
    ctx = copy_context()
    _trace_callback.reset(token_trace)
    _hitl_callback.reset(token_hitl)

    def run_agent() -> Any:
        try:
            return ctx.run(run_graph, query, thread_id, fast)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, SENTINEL)

    future = loop.run_in_executor(_executor, run_agent)

    while True:
        try:
            item = await asyncio.wait_for(queue.get(), timeout=600.0)
        except asyncio.TimeoutError:
            yield _sse_event({"detail": "Timeout: nessun evento dall'agente"}, event="error")
            return

        if item is SENTINEL:
            break

        if isinstance(item, dict) and item.get("_sse_event_type") == "approval_request":
            payload = {k: v for k, v in item.items() if k != "_sse_event_type"}
            yield _sse_event(payload, event="approval_request")
        else:
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
            last = messages[-1]
            answer = getattr(last, "content", str(last))
            if not isinstance(answer, str):
                answer = str(answer)

    yield _sse_event(
        {"answer": answer, "total_traces": len(result.get("traces", [])) if isinstance(result, dict) else 0},
        event="done",
    )


@app.get("/examples")
async def examples() -> List[Dict[str, str]]:
    return EXAMPLE_PROMPTS


@app.post("/manual")
async def manual(request: ManualRequest) -> Dict[str, Any]:
    answer, state = run_manual(request.query, fast_mode=request.fast)
    return {"answer": answer, "traces": state.get("traces", [])}


@app.post("/graph")
async def graph(request: GraphRequest) -> Dict[str, Any]:
    result = run_graph(request.query, thread_id=request.thread_id, fast_mode=request.fast)
    answer = ""
    if isinstance(result, dict):
        messages = result.get("messages", [])
        if messages:
            last = messages[-1]
            answer = getattr(last, "content", str(last))
            if not isinstance(answer, str):
                answer = str(answer)
    return {"answer": answer, "traces": result.get("traces", []) if isinstance(result, dict) else [], "state": result}


@app.get("/manual/stream")
async def manual_stream(
    query: str = Query(...),
    fast: bool = Query(False),
) -> StreamingResponse:
    return StreamingResponse(
        _stream_manual(query, fast),
        media_type="text/event-stream",
    )


@app.get("/graph/stream")
async def graph_stream(
    query: str = Query(...),
    thread_id: str = Query("demo-day3-001"),
    fast: bool = Query(False),
) -> StreamingResponse:
    return StreamingResponse(
        _stream_graph(query, thread_id, fast),
        media_type="text/event-stream",
    )


class ApproveRequest(BaseModel):
    approved: bool


@app.post("/approve/{run_id}")
async def approve_action(run_id: str, request: ApproveRequest) -> Dict[str, Any]:
    """
    Riceve la decisione umana (approvato/rifiutato) per una pending action HITL.
    Sblocca il thread dell'agente che è in attesa della conferma.
    """
    record = _pending_approvals.get(run_id)
    if not record:
        raise HTTPException(
            status_code=404,
            detail=f"Nessuna approvazione pendente per run_id={run_id!r}.",
        )
    record["approved"] = request.approved
    record["event"].set()
    return {"ok": True, "approved": request.approved, "run_id": run_id}
