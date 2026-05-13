# GenAI Course — ITSM Agents con LangChain, LangGraph e MCP

Questo repository raccoglie i laboratori pratici del corso GenAI, organizzati per giornata. Ogni giorno introduce concetti progressivamente più avanzati: dal RAG classico agli agenti multi-step, fino all'architettura multi-agente con supervisore e al protocollo MCP.

## Avvio rapido

```bash
chmod +x start.sh
./start.sh
```

Il menu interattivo chiede cosa avviare:

| Scelta | Cosa parte | Porta | UI |
|--------|-----------|-------|----|
| 1 | Day 2 — ITSM Agent (RAG + tool) | 8000 | `http://localhost:8000` |
| 2 | Day 3 — Multi-Agent ITSM con HITL | 8000 | `day3_multiagent/index.html` |
| 3 | Day 3 — MCP Lab (tool locale) | 8001 | `http://localhost:8001` |
| 4 | Day 3 — MCP Lab + Tool Server remoto | 8001 + 8002 | `http://localhost:8001` |

---

## Requisiti

- Python 3.10+
- Google Cloud service account JSON in `service_account.json` (progetto `hclsw-gcp-wrkld-auto`, region `us-east1`)

```bash
pip install -r requirements.txt
```

---

## Day 1 — RAG con Chroma e Vertex AI

**Cartella:** `day1_morning_rag/`

Introduzione al Retrieval-Augmented Generation: ingestion di documenti (SRE, NIST, OWASP), chunking, embedding con Vertex AI, query via ChromaDB.

```bash
python day1_morning_rag/download_corpus.py
python day1_morning_rag/main.py
```

File chiave:
- `main.py` — pipeline RAG principale
- `bonus_rag.py` — variante con reranking
- `afternoon_langchain_rag.py` — RAG con LangChain
- `corpus/extracted/` — documenti sorgente (SRE, NIST AI RMF, OWASP LLM Top 10)

---

## Day 2 — ITSM Agent con tool e RAG

**Cartella:** `day2_agents/`

Agente ReAct costruito con LangChain e Vertex AI (`gemini-2.0-flash`). Gestisce ticket ITSM tramite tool strutturati: ricerca knowledge base, lookup record, calcolo SLA, azioni critiche con conferma.

**Avvio:** `./start.sh → 1`

```bash
# oppure manualmente:
python day2_agents/itsm_agent.py setup-rag-data --ingest
uvicorn day2_agents.api:app --reload --port 8000
open http://localhost:8000
```

File chiave:
- `itsm_agent.py` — tool (`search_kb`, `lookup_record`, `compute_sla`, `execute_critical_action`), agente ReAct, dati di esempio
- `api.py` — FastAPI con endpoint REST e streaming SSE
- `itsm_agent_visualize.py` — visualizzazione grafo LangGraph
- `data/` — policy ITSM usate dalla knowledge base (incident, SLA, escalation, HR, procurement…)

---

## Day 3 — Multi-Agent con LangGraph e MCP

**Cartella:** `day3_multiagent/`

### 3a — Supervisore Multi-Agente con HITL

Architettura a grafo con quattro nodi:
- **triage_agent** — classifica l'intento e smista
- **knowledge_agent** — risponde a domande su policy e knowledge base
- **action_agent** — propone azioni critiche (escalation, patch, rollback)
- **supervisor_node** — orchestra il flusso, gestisce Human-in-the-Loop (HITL)

Il HITL blocca l'esecuzione di azioni critiche finché l'utente non approva o rifiuta dalla UI (modal di approvazione via SSE).

**Avvio:** `./start.sh → 2`

```bash
# oppure manualmente:
uvicorn day3_multiagent.api:app --reload --port 8000
open day3_multiagent/index.html
```

File chiave:
- `supervisor.py` — grafo LangGraph, `AgentState`, nodi, HITL con `threading.Event`
- `api.py` — FastAPI, `POST /run`, `GET /stream/{run_id}`, `POST /approve/{run_id}`
- `index.html` — console web con feed SSE e modal di approvazione HITL

### 3b — MCP Lab

Dimostrazione del **Model Context Protocol (MCP)**: separazione tra host (agent) e server (tool). Il client MCP scopre i tool tramite discovery (`list_tools`) e li invoca tramite contratto standard (`call_tool`).

Due modalità:
- **Locale** — `MCPAdapter` in-process, tool nello stesso processo dell'agent
- **Remota** — `MCPHTTPAdapter` chiama un server HTTP separato su porta 8002, dimostrando che l'agent non sa dove girano i tool

**Avvio (locale):** `./start.sh → 3`  
**Avvio (remoto):** `./start.sh → 4`

```bash
# oppure manualmente:
uvicorn day3_multiagent.mcp_tool_server:app --port 8002 &
uvicorn day3_multiagent.mcp_api:app --port 8001
open http://localhost:8001
```

Nella console MCP:
1. Apri `http://localhost:8001`
2. Spunta **Remote** e inserisci `http://localhost:8002`
3. Il badge nel pannello di sinistra indica se il server remoto è online
4. Invia una query — il banner cambia da verde (locale) a viola (remoto)
5. Le trace dei tool call mostrano il tag `REMOTE` e il bordo viola

File chiave:
- `mcp_lab.py` — `MCPAdapter`, `MCPHTTPAdapter`, `DOMAIN_TOOLS`, `KnowledgeAgent`, CLI
- `mcp_api.py` — FastAPI porta 8001, endpoint `/mcp/agent/stream`, serve la console HTML
- `mcp_tool_server.py` — server MCP porta 8002, espone solo i tool (`/mcp/tools`, `/mcp/tools/{name}/call`, `/health`)
- `mcp_console.html` — console web a 3 colonne: tool registry, agent, trace feed

---

## Struttura del progetto

```
.
├── start.sh                     # menu interattivo di avvio
├── index.html                   # UI Day 2 (servita da Day 2 API)
├── requirements.txt
├── service_account.json         # credenziali GCP (non committare)
├── day1_morning_rag/            # RAG con Chroma
├── day2_agents/                 # ITSM Agent ReAct
│   ├── itsm_agent.py
│   └── api.py
└── day3_multiagent/             # Multi-agent + MCP
    ├── supervisor.py
    ├── api.py
    ├── index.html
    ├── mcp_lab.py
    ├── mcp_api.py
    ├── mcp_tool_server.py
    └── mcp_console.html
```

## Note

- `service_account.json` contiene credenziali GCP — non committare mai in repository pubblici.
- Variabile d'ambiente: `GOOGLE_APPLICATION_CREDENTIALS=./service_account.json` (impostata automaticamente dagli script).
- Per usare il backend MCP remoto via env: `MCP_BACKEND=http MCP_REMOTE_URL=http://localhost:8002`.
