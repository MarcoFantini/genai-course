#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

echo ""
echo "======================================"
echo "  GenAI Course — Scegli cosa avviare"
echo "======================================"
echo "  1) Day 2 — ITSM Agent        (porta 8000)"
echo "  2) Day 3 — Multi-Agent ITSM  (porta 8000)"
echo "  3) Day 3 — MCP Lab            (porta 8001)"
echo "  4) Day 3 — MCP Lab + Tool Server remoto (porta 8001 + 8002)"
echo "======================================"
echo -n "Scelta [1-4]: "
read CHOICE

case "$CHOICE" in
  1)
    echo "[Day 2] Preparo i dati RAG..."
    python day2_agents/itsm_agent.py setup-rag-data --ingest
    python day2_agents/itsm_agent.py examples
    echo "[Day 2] Avvio API su :8000..."
    uvicorn day2_agents.api:app --reload --port 8000 &
    sleep 2
    open http://localhost:8000
    wait
    ;;
  2)
    echo "[Day 3] Avvio Multi-Agent Supervisor su :8000..."
    uvicorn day3_multiagent.api:app --reload --port 8000 &
    sleep 2
    open day3_multiagent/index.html
    wait
    ;;
  3)
    echo "[Day 3] Avvio MCP Lab su :8001..."
    uvicorn day3_multiagent.mcp_api:app --reload --port 8001 &
    sleep 2
    open http://localhost:8001
    wait
    ;;
  4)
    echo "[Day 3] Avvio MCP Tool Server su :8002..."
    uvicorn day3_multiagent.mcp_tool_server:app --reload --port 8002 &
    TOOL_SERVER_PID=$!
    sleep 1
    echo "[Day 3] Avvio MCP Lab su :8001..."
    uvicorn day3_multiagent.mcp_api:app --reload --port 8001 &
    sleep 2
    open http://localhost:8001
    echo "Premi Ctrl+C per fermare entrambi i server."
    wait
    ;;
  *)
    echo "Scelta non valida. Uscita."
    exit 1
    ;;
esac
