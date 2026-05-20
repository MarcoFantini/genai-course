"""
day5_governance/eval_api.py

Server FastAPI leggero che espone le funzioni di eval_suite.py come API REST
e serve la UI statica su http://localhost:5174.

Avvio:
    cd /Users/marco.fantini/HCLTech/genai-course
    source .venv/bin/activate
    python day5_governance/eval_api.py

Endpoints:
    GET  /                      — UI HTML
    POST /eval/semantic         — semantic similarity
    POST /eval/judge            — LLM-as-a-judge (+ swap test)
    POST /eval/both             — semantic + judge in un colpo
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

_HERE = Path(__file__).parent.resolve()
_ROOT = _HERE.parent
for _p in [str(_HERE), str(_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import eval_suite  # noqa: E402

app = FastAPI(title="Eval Suite UI", version="1.0.0", docs_url="/docs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_UI_FILE = _HERE / "eval_ui.html"


# ---------------------------------------------------------------------------
# Modelli
# ---------------------------------------------------------------------------

class EvalRequest(BaseModel):
    question: str = ""
    expected: str
    actual: str
    fast: bool = True


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def ui():
    return _UI_FILE.read_text(encoding="utf-8")


@app.post("/eval/semantic")
async def api_semantic(req: EvalRequest):
    result = eval_suite.eval_semantic(req.expected, req.actual)
    return result


@app.post("/eval/judge")
async def api_judge(req: EvalRequest):
    result = eval_suite.eval_llm_judge(
        question=req.question,
        expected=req.expected,
        actual=req.actual,
        fast=req.fast,
    )
    return result


@app.post("/eval/both")
async def api_both(req: EvalRequest):
    semantic = eval_suite.eval_semantic(req.expected, req.actual)
    judge = eval_suite.eval_llm_judge(
        question=req.question,
        expected=req.expected,
        actual=req.actual,
        fast=req.fast,
    )
    return {"semantic": semantic, "judge": judge}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    print("\n" + "─" * 56)
    print("  Eval Suite UI — Day 5")
    print("  UI:   http://localhost:5174")
    print("  Docs: http://localhost:5174/docs")
    print("─" * 56 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=5174, log_level="warning")
