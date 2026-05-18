from __future__ import annotations
import sys, io
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

"""
day4_enterprise/bonus_crewai.py

╔══════════════════════════════════════════════════════════════════════════════╗
║  Esercitazione Giorno 4 — BONUS                                             ║
║  CrewAI: pipeline ITSM role-based con agenti specializzati                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

Obiettivo:
  Implementare il flusso ITSM con CrewAI, mostrando il paradigma "role-based"
  in contrasto con il paradigma "graph-based" di LangGraph.

  Invece di nodi e archi (LangGraph), CrewAI usa:
    • Agent con role + goal + backstory (persona)
    • Task con description + expected_output + agente assegnato
    • Crew che orchestra agenti e task in sequenza (o in parallelo)

Architettura:

        Richiesta ITSM
              │
              ▼
  ┌───────────────────────┐
  │  TriageAgent           │  Classifica priorità, identifica ticket
  │  role: ITSM Triage     │  tools: lookup_ticket, list_open_tickets
  └────────────┬──────────┘
               │ output: analisi_triage
               ▼
  ┌───────────────────────┐
  │  KnowledgeAgent        │  Cerca policy e procedure applicabili
  │  role: Knowledge Mgr   │  tools: search_kb, compute_sla
  └────────────┬──────────┘
               │ output: policy_relevant + sla_status
               ▼
  ┌───────────────────────┐
  │  ReportAgent           │  Produce il report finale strutturato
  │  role: Report Writer   │  tools: (solo testo, usa output precedenti)
  └────────────┬──────────┘
               │
               ▼
        Report ITSM completo

Comandi:
    python day4_enterprise/bonus_crewai.py crew "INC-1002"
    python day4_enterprise/bonus_crewai.py crew "Analizza i ticket P1 aperti e proponi azioni"
    python day4_enterprise/bonus_crewai.py crew "INC-1001" --fast
    python day4_enterprise/bonus_crewai.py compare-paradigms   # LangGraph vs CrewAI (teorico)
    python day4_enterprise/bonus_crewai.py examples

Variabili .env:
    GOOGLE_API_KEY=...
    GEMINI_MODEL=gemini-2.5-flash
    MIN_SECONDS_BETWEEN_MODEL_CALLS=15
    PRICE_INPUT_PER_1M=0.10
    PRICE_OUTPUT_PER_1M=0.40

Nota didattica:
    CrewAI è eccellente per pipeline documentali e workflow role-based dove
    la metafora "team di specialisti" riflette bene il dominio reale.
    LangGraph è migliore per flussi con branching complesso, HITL e recovery.
    Questo lab mostra QUANDO scegliere CrewAI vs LangGraph (decision matrix).
"""

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# ── import opzionali ────────────────────────────────────────────────────────────

try:
    from crewai import Agent, Task, Crew, Process
    from crewai.tools import BaseTool as CrewBaseTool
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    print("[INFO] crewai non installato. pip install crewai")
    print("       Questo lab funziona anche senza CrewAI (modalità simulazione).")

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("[WARN] langchain-google-genai non installato.")

try:
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False


# ── config ─────────────────────────────────────────────────────────────────────

load_dotenv()

BASE_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = BASE_DIR.parent if BASE_DIR.name == "day4_enterprise" else BASE_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RUNS_DIR = BASE_DIR / "runs" / "day4_bonus"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
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
# DOMAIN DATA — stesso dominio ITSM degli esercizi precedenti
# =============================================================================

TICKETS: Dict[str, Dict[str, Any]] = {
    "INC-1001": {
        "id": "INC-1001", "priority": "P2", "status": "Open",
        "title": "VPN aziendale non raggiungibile da sede Milano",
        "description": "50 utenti non si connettono alla VPN. Impatto: lavoro da remoto bloccato.",
        "sla_hours": 4, "domain": "network",
        "assigned_to": "Network Team",
    },
    "INC-1002": {
        "id": "INC-1002", "priority": "P1", "status": "Open",
        "title": "Database Oracle produzione non risponde",
        "description": "DB principale down. Tutti i servizi applicativi bloccati. Revenue 10k€/h.",
        "sla_hours": 1, "domain": "database",
        "assigned_to": "DBA Team",
    },
    "INC-1003": {
        "id": "INC-1003", "priority": "P3", "status": "In Progress",
        "title": "Stampante ufficio 3° piano offline",
        "description": "HP offline. Workaround: usare quella al 2° piano.",
        "sla_hours": 8, "domain": "hardware",
        "assigned_to": "Helpdesk",
    },
    "INC-1004": {
        "id": "INC-1004", "priority": "P2", "status": "Open",
        "title": "Sistema SSO intermittente",
        "description": "Errori di login ogni ~15 min. SAML assertion fallisce.",
        "sla_hours": 4, "domain": "security",
        "assigned_to": "Security Team",
    },
}

KB_DOCS = [
    {
        "id": "KB-001",
        "title": "Policy P1 — Incident Critico",
        "content": "Incident P1: escalation immediata. SLA 1 ora. Notifica CTO e manager entro 15 min. War room entro 30 min. Post-mortem entro 48 ore.",
        "tags": ["P1", "escalation", "critico"],
    },
    {
        "id": "KB-002",
        "title": "Policy P2 — Incident Alto",
        "content": "Incident P2: presa in carico entro 30 min. SLA 4 ore. Notifica team lead entro 1 ora. Update ogni 2 ore al richiedente.",
        "tags": ["P2", "alto"],
    },
    {
        "id": "KB-003",
        "title": "Procedura VPN — Troubleshooting",
        "content": "1. Verifica servizio gateway VPN. 2. Controlla certificati SSL. 3. Riavvia gateway. 4. Bridge emergenza se >30 min. 5. Notifica networking team.",
        "tags": ["vpn", "network", "P2"],
    },
    {
        "id": "KB-004",
        "title": "Procedura Database Oracle — Recovery",
        "content": "1. Controlla alert log Oracle. 2. Verifica spazio tablespace. 3. V$SESSION per sessioni bloccanti. 4. Recover se archivelog attivo. 5. Contatta DBA on-call.",
        "tags": ["oracle", "database", "P1", "recovery"],
    },
    {
        "id": "KB-005",
        "title": "SLA Definitions",
        "content": "P1: risposta 15min, risoluzione 1h. P2: risposta 30min, risoluzione 4h. P3: risposta 2h, risoluzione 8h. P4: risposta 4h, risoluzione 24h.",
        "tags": ["sla", "definizioni"],
    },
]


# =============================================================================
# TOOL FUNCTIONS — usate sia dai CrewAI tool che dalla simulazione
# =============================================================================

def fn_lookup_ticket(ticket_id: str) -> str:
    t = TICKETS.get(ticket_id.upper())
    if t:
        return json.dumps(t, ensure_ascii=False, indent=2)
    return json.dumps({"error": f"Ticket {ticket_id} non trovato", "available": list(TICKETS.keys())})


def fn_search_kb(query: str, priority_tag: str = "") -> str:
    q = query.lower()
    results = []
    for doc in KB_DOCS:
        score = sum(1 for tag in doc["tags"] if tag.lower() in q or q in tag.lower())
        score += sum(1 for w in q.split() if w in doc["content"].lower() or w in doc["title"].lower())
        if priority_tag and priority_tag.upper() in doc["tags"]:
            score += 2
        if score > 0:
            results.append((score, doc))
    results.sort(key=lambda x: x[0], reverse=True)
    top = [d for _, d in results[:3]]
    return json.dumps({"results": top, "count": len(top)}, ensure_ascii=False, indent=2)


def fn_compute_sla(ticket_id: str) -> str:
    import random
    t = TICKETS.get(ticket_id.upper())
    if not t:
        return json.dumps({"error": f"Ticket {ticket_id} non trovato"})
    elapsed = round(random.uniform(0.1, t["sla_hours"] * 1.6), 2)
    remaining = max(0.0, t["sla_hours"] - elapsed)
    breach = remaining == 0.0
    return json.dumps({
        "ticket_id": ticket_id,
        "priority": t["priority"],
        "sla_hours": t["sla_hours"],
        "elapsed_hours": elapsed,
        "remaining_hours": round(remaining, 2),
        "status": "SLA_BREACH" if breach else "IN_SLA",
        "urgency": "CRITICA" if breach else ("ALTA" if remaining < t["sla_hours"] * 0.2 else "NORMALE"),
    }, ensure_ascii=False, indent=2)


def fn_list_open_tickets(priority: str = "") -> str:
    results = [
        t for t in TICKETS.values()
        if t["status"] in ("Open", "In Progress")
        and (not priority or t["priority"] == priority.upper())
    ]
    return json.dumps({
        "tickets": results,
        "count": len(results),
        "priorities": list(set(t["priority"] for t in results)),
    }, ensure_ascii=False, indent=2)


# =============================================================================
# CREWAI SETUP — agent, task, crew
# =============================================================================

def _make_gemini_llm():
    """Crea l'LLM Gemini compatibile con CrewAI."""
    if not LANGCHAIN_AVAILABLE or not GOOGLE_API_KEY:
        return None
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.1,
    )


def _make_crewai_tools():
    """
    Crea i tool CrewAI wrappando le nostre funzioni di dominio.

    CrewAI usa BaseTool con metodo _run() sincrono.
    L'integrazione con LangChain tools è supportata ma richiede langchain.
    """
    if not CREWAI_AVAILABLE or not PYDANTIC_AVAILABLE:
        return []

    class LookupTicketTool(CrewBaseTool):
        name: str = "lookup_ticket"
        description: str = "Recupera i dettagli completi di un ticket ITSM. Input: ticket_id (es. INC-1002)"

        def _run(self, ticket_id: str) -> str:
            return fn_lookup_ticket(ticket_id)

    class SearchKBTool(CrewBaseTool):
        name: str = "search_kb"
        description: str = "Cerca nella Knowledge Base ITSM. Input: query di ricerca (es. 'policy P1 escalation')"

        def _run(self, query: str) -> str:
            return fn_search_kb(query)

    class ComputeSLATool(CrewBaseTool):
        name: str = "compute_sla"
        description: str = "Calcola lo stato SLA del ticket. Input: ticket_id (es. INC-1001)"

        def _run(self, ticket_id: str) -> str:
            return fn_compute_sla(ticket_id)

    class ListOpenTicketsTool(CrewBaseTool):
        name: str = "list_open_tickets"
        description: str = "Elenca i ticket aperti. Input: priority (P1/P2/P3 opzionale)"

        def _run(self, priority: str = "") -> str:
            return fn_list_open_tickets(priority)

    return [LookupTicketTool(), SearchKBTool(), ComputeSLATool(), ListOpenTicketsTool()]


def build_crew(task_input: str) -> Optional[Any]:
    """
    Costruisce la Crew ITSM con 3 agenti specializzati.

    Nota didattica:
      In CrewAI, ogni agente ha una "persona" (role + goal + backstory).
      Questo permette al LLM di assumere la prospettiva dello specialista
      e produrre output più pertinenti al ruolo.
      È diverso da LangGraph dove la specializzazione avviene tramite
      system prompt e routing condizionale — non tramite persona.
    """
    if not CREWAI_AVAILABLE:
        return None

    llm = _make_gemini_llm()
    tools = _make_crewai_tools()
    triage_tools = tools  # tutti i tool al TriageAgent
    knowledge_tools = [t for t in tools if t.name in ("search_kb", "compute_sla")]
    report_tools = []  # il ReportAgent non esegue tool: usa solo il contesto

    # ── Agente 1: Triage ──────────────────────────────────────────────────────

    triage_agent = Agent(
        role="ITSM Triage Specialist",
        goal=(
            "Classificare e analizzare i ticket ITSM con precisione. "
            "Identificare la priorità, lo stato SLA e il team responsabile."
        ),
        backstory=(
            "Sei un esperto ITIL con 10 anni di esperienza nel triage degli incident. "
            "Sai riconoscere immediatamente la gravità di un problema e chi deve occuparsene. "
            "Lavori in italiano e fornisci sempre dati concreti, mai opinioni vaghe."
        ),
        tools=triage_tools,
        llm=llm,
        verbose=True,
        allow_delegation=False,  # non delega ad altri agenti
        max_iter=4,
    )

    # ── Agente 2: Knowledge Manager ───────────────────────────────────────────

    knowledge_agent = Agent(
        role="ITSM Knowledge Manager",
        goal=(
            "Trovare le policy, procedure e informazioni SLA più rilevanti "
            "per supportare la risoluzione del ticket analizzato dal Triage Specialist."
        ),
        backstory=(
            "Sei il custode della Knowledge Base aziendale. Conosci a memoria ogni policy, "
            "procedura di troubleshooting e accordo SLA. "
            "Fornisci sempre citazioni precise con ID documento (KB-XXX) "
            "e parafrasi dei contenuti rilevanti. Lavori in italiano."
        ),
        tools=knowledge_tools,
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3,
    )

    # ── Agente 3: Report Writer ───────────────────────────────────────────────

    report_agent = Agent(
        role="ITSM Report Writer",
        goal=(
            "Produrre un report ITSM strutturato, chiaro e azionabile "
            "che integri l'analisi del triage e le policy della Knowledge Base."
        ),
        backstory=(
            "Sei un technical writer specializzato in documentazione ITSM. "
            "Trasformi dati tecnici in report comprensibili per manager e operatori. "
            "I tuoi report seguono sempre la struttura: Sommario → Analisi → Azioni → SLA. "
            "Usi markdown, sei conciso ma completo. Lavori in italiano."
        ),
        tools=report_tools,
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=2,
    )

    # ── Task 1: Triage ────────────────────────────────────────────────────────

    triage_task = Task(
        description=(
            f"Analizza la seguente richiesta ITSM: '{task_input}'\n\n"
            "1. Recupera i dettagli del/i ticket coinvolti (usa lookup_ticket)\n"
            "2. Lista tutti i ticket aperti rilevanti (usa list_open_tickets)\n"
            "3. Calcola lo stato SLA (usa compute_sla)\n"
            "4. Produci: ID ticket, priorità, status SLA, team responsabile, "
            "stima impatto business."
        ),
        expected_output=(
            "Un'analisi strutturata con: ID ticket, priorità confermata, "
            "status SLA (in_sla/breach), elapsed/remaining hours, team assegnato, "
            "impatto stimato in italiano."
        ),
        agent=triage_agent,
    )

    # ── Task 2: Knowledge ─────────────────────────────────────────────────────

    knowledge_task = Task(
        description=(
            "Basandoti sull'analisi di triage ricevuta, cerca nella Knowledge Base "
            "le policy e procedure più rilevanti per la risoluzione.\n\n"
            "1. Cerca policy applicabili alla priorità identificata (usa search_kb)\n"
            "2. Cerca procedure di troubleshooting per il dominio tecnico\n"
            "3. Verifica gli accordi SLA applicabili\n"
            "4. Cita ogni documento con il suo ID (KB-XXX)"
        ),
        expected_output=(
            "Lista di policy e procedure rilevanti con: ID documento (KB-XXX), "
            "titolo, contenuto rilevante estratto, applicabilità al caso specifico."
        ),
        agent=knowledge_agent,
        context=[triage_task],  # riceve l'output del triage task
    )

    # ── Task 3: Report ────────────────────────────────────────────────────────

    report_task = Task(
        description=(
            "Produci il report ITSM finale integrando l'analisi di triage "
            "e le policy della Knowledge Base.\n\n"
            "Il report deve seguire questa struttura markdown:\n"
            "## 🔴 Sommario Esecutivo\n"
            "## 📋 Analisi Ticket\n"
            "## 📚 Policy e Procedure Applicabili\n"
            "## ⚡ Azioni Immediate (ordinate per urgenza)\n"
            "## 📊 Status SLA\n"
            "## 📝 Note Operative\n\n"
            "Sii specifico, usa dati concreti, cita ID ticket e KB."
        ),
        expected_output=(
            "Report ITSM completo in formato markdown, strutturato nelle sezioni indicate, "
            "con azioni concrete e prioritizzate, in italiano."
        ),
        agent=report_agent,
        context=[triage_task, knowledge_task],  # riceve output di entrambi
    )

    # ── Crew ──────────────────────────────────────────────────────────────────

    crew = Crew(
        agents=[triage_agent, knowledge_agent, report_agent],
        tasks=[triage_task, knowledge_task, report_task],
        process=Process.sequential,  # il report aspetta knowledge che aspetta triage
        verbose=True,
    )

    return crew


# =============================================================================
# SIMULAZIONE — quando CrewAI non è installato
# =============================================================================

def _simulate_crew(task_input: str, fast: bool = False) -> Dict[str, Any]:
    """
    Simula il flusso CrewAI con LLM diretto (fallback senza CrewAI installato).
    Mantiene la struttura 3-agent: triage → knowledge → report.
    """
    print("  [simulazione] CrewAI non disponibile — simulazione agenti sequenziali")

    if not LANGCHAIN_AVAILABLE or not GOOGLE_API_KEY or fast:
        return _mock_crew_result(task_input)

    from langchain_core.messages import HumanMessage, SystemMessage

    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.1,
    )

    tickets_str = json.dumps(list(TICKETS.values()), ensure_ascii=False, indent=2)
    kb_str = json.dumps(KB_DOCS, ensure_ascii=False, indent=2)

    results = {}
    tokens_total = 0

    # Step 1: Triage
    print("\n  [Agent 1: TriageSpecialist] analisi in corso...")
    _rate_limit()
    r1 = llm.invoke([
        SystemMessage(content=f"Sei un ITSM Triage Specialist. Tickets: {tickets_str}"),
        HumanMessage(content=f"Analizza: {task_input}. Fornisci: ID, priorità, SLA, impatto."),
    ])
    triage_output = str(r1.content)
    usage1 = getattr(r1, "usage_metadata", {}) or {}
    tokens_total += (usage1.get("input_tokens", 0) or 0) + (usage1.get("output_tokens", 0) or 0)
    results["triage"] = triage_output
    print(f"  ✓ Triage completato")

    # Step 2: Knowledge
    print("\n  [Agent 2: KnowledgeManager] ricerca in corso...")
    _rate_limit()
    r2 = llm.invoke([
        SystemMessage(content=f"Sei il Knowledge Manager ITSM. KB: {kb_str}"),
        HumanMessage(content=f"Cerca policy e procedure per: {triage_output[:400]}"),
    ])
    knowledge_output = str(r2.content)
    usage2 = getattr(r2, "usage_metadata", {}) or {}
    tokens_total += (usage2.get("input_tokens", 0) or 0) + (usage2.get("output_tokens", 0) or 0)
    results["knowledge"] = knowledge_output
    print(f"  ✓ Knowledge completato")

    # Step 3: Report
    print("\n  [Agent 3: ReportWriter] redazione in corso...")
    _rate_limit()
    r3 = llm.invoke([
        SystemMessage(content="Sei un ITSM Report Writer. Produci report markdown strutturati."),
        HumanMessage(content=(
            f"Triage: {triage_output[:300]}\n"
            f"Knowledge: {knowledge_output[:300]}\n\n"
            "Produci report con sezioni: Sommario, Analisi, Policy, Azioni, SLA."
        )),
    ])
    report_output = str(r3.content)
    usage3 = getattr(r3, "usage_metadata", {}) or {}
    tokens_total += (usage3.get("input_tokens", 0) or 0) + (usage3.get("output_tokens", 0) or 0)
    results["report"] = report_output
    print(f"  ✓ Report completato")

    return {
        "task_input": task_input,
        "agent_outputs": results,
        "final_report": report_output,
        "tokens_total": tokens_total,
        "cost_usd": _cost(tokens_total // 2, tokens_total // 2),
        "mode": "simulation_llm",
    }


def _mock_crew_result(task_input: str) -> Dict[str, Any]:
    """Risultato mock per fast mode / demo senza LLM."""
    q = task_input.lower()
    if "inc-1002" in q or "oracle" in q or "database" in q:
        report = """## 🔴 Sommario Esecutivo
**INC-1002** — Database Oracle produzione non risponde. Priorità P1. Impatto critico.

## 📋 Analisi Ticket
- **ID:** INC-1002 | **Priorità:** P1 | **Status:** Open
- **Descrizione:** DB principale down — revenue impact 10k€/h
- **Team:** DBA Team | **SLA:** 1 ora

## 📚 Policy e Procedure Applicabili
- **KB-001** Policy P1: war room entro 30min, notifica CTO entro 15min
- **KB-004** Procedura DB Oracle: alert log → tablespace → V$SESSION → DBA on-call

## ⚡ Azioni Immediate
1. 🚨 Notifica CTO e manager (entro 15 min — **SCADUTO**)
2. Attiva war room Ops team
3. DBA verifica alert log Oracle (`/u01/app/oracle/diag/rdbms/`)
4. Controlla spazio tablespace (`df -h /oracle`)
5. Esegui recovery automatico se archivelog attivo

## 📊 Status SLA
- ⚠️ SLA P1 = 1 ora — probabile breach, escalation immediata richiesta

## 📝 Note Operative
Caso critico: impatto diretto su revenue. Priorità assoluta su tutti gli altri ticket."""
    else:
        report = f"""## 🔴 Sommario Esecutivo
Report generato per: '{task_input[:60]}' (modalità mock)

## 📋 Analisi Ticket
Ticket disponibili: INC-1001 (P2), INC-1002 (P1), INC-1003 (P3), INC-1004 (P2)

## 📚 Policy Applicabili
KB-001 (P1), KB-002 (P2), KB-005 (SLA)

## ⚡ Azioni Immediate
1. Analizza ticket specificato con ID esatto
2. Verifica SLA status
3. Consulta KB per procedure specifiche

## 📝 Note
Usa --fast per demo, rimuovi per risposta LLM reale."""

    return {
        "task_input": task_input,
        "agent_outputs": {
            "triage": "[MOCK] Triage completato",
            "knowledge": "[MOCK] Ricerca KB completata",
            "report": report,
        },
        "final_report": report,
        "tokens_total": 450,
        "cost_usd": _cost(300, 150),
        "mode": "mock",
    }


# =============================================================================
# COMPARE PARADIGMS — teoria LangGraph vs CrewAI
# =============================================================================

def cmd_compare_paradigms(args):
    print("""
╔══════════════════════════════════════════════════════════════════╗
║  CONFRONTO PARADIGMI: LangGraph vs CrewAI                        ║
╚══════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────┬─────────────────────────────────────┐
│           LANGGRAPH                  │            CREWAI                   │
├─────────────────────────────────────┼─────────────────────────────────────┤
│ Paradigma: grafo orientato          │ Paradigma: team di specialisti      │
│   nodi + archi + routing            │   agent + task + crew               │
│                                     │                                     │
│ Flusso: esplicito nel codice        │ Flusso: dichiarativo (Process.seq)  │
│   add_conditional_edges()           │   context=[task_precedente]         │
│                                     │                                     │
│ Stato: TypedDict condiviso          │ Stato: output task come stringa     │
│   reducer per merge                 │   passato come context              │
│                                     │                                     │
│ Checkpoint: ✓ SQLite/Postgres       │ Checkpoint: ✗ (non nativo)         │
│ HITL: ✓ interrupt_before/after      │ HITL: ✗ (limitato)                 │
│ Visualizzazione grafo: ✓            │ Visualizzazione: crew.kickoff()     │
│                                     │                                     │
│ Curva apprendimento: ALTA           │ Curva apprendimento: BASSA          │
│ Boilerplate: ALTO                   │ Boilerplate: BASSO                  │
│ Flessibilità routing: MASSIMA       │ Flessibilità routing: LIMITATA      │
│                                     │                                     │
│ ✅ QUANDO USARE:                    │ ✅ QUANDO USARE:                    │
│   - Branching condizionale         │   - Pipeline sequenziali            │
│   - Human approval step            │   - Workflow documentali            │
│   - Recovery da errori             │   - Team role-based                 │
│   - Multi-agent con shared state   │   - Content generation              │
│   - Produzione enterprise          │   - Prototipazione rapida           │
│                                     │                                     │
│ ❌ QUANDO NON USARE:               │ ❌ QUANDO NON USARE:               │
│   - Task lineari semplici          │   - Branching complesso             │
│   - Team piccoli senza DevOps      │   - HITL obbligatorio               │
│   - Prototipo in 1 ora             │   - Recovery critico                │
└─────────────────────────────────────┴─────────────────────────────────────┘

QUESTO LAB ITSM:
  • LangGraph (Day 3): supervisor → triage/knowledge/action → supervisor
    → perfetto per HITL ("confermi questa azione critica?") e recovery
  • CrewAI (questo bonus): triage → knowledge → report writer
    → perfetto per pipeline documentali sequenziali senza branching

DOMANDE DI RIFLESSIONE:
  1. Cosa succederebbe se il ticket richiedesse approvazione umana? (HITL)
     → LangGraph: interrupt_before="action_node" ✓
     → CrewAI: necessita codice custom ✗
  2. Se l'agent triage fallisce, come gestisci il retry?
     → LangGraph: conditional edge verso error_handler ✓
     → CrewAI: max_iter + retry nativo ✓ (ma meno controllo)
  3. Come aggiungeresti un 4° agente "EscalationManager" solo se P1?
     → LangGraph: conditional edge nel grafo ✓ (facile)
     → CrewAI: conditional task (supportato in CrewAI 0.65+) ✓
""")


# =============================================================================
# CLI
# =============================================================================

def cmd_crew(args):
    """Esegui la crew CrewAI (o simulazione)."""
    task_input = args.task
    fast = getattr(args, "fast", False)
    t0 = time.monotonic()

    print(f"\n{'═'*65}")
    print(f"  CREWAI LAB — ITSM Multi-Agent Crew")
    print(f"  Task: {task_input[:60]}{'...' if len(task_input)>60 else ''}")
    print(f"  Modalità: {'fast (mock)' if fast else ('CrewAI' if CREWAI_AVAILABLE else 'simulazione')}")
    print(f"{'═'*65}\n")

    if CREWAI_AVAILABLE and not fast and GOOGLE_API_KEY:
        print("Avvio Crew (3 agenti, processo sequenziale)...")
        try:
            crew = build_crew(task_input)
            if crew is None:
                raise RuntimeError("Crew non disponibile")

            crew_result = crew.kickoff()
            duration_ms = (time.monotonic() - t0) * 1000

            # CrewAI restituisce un oggetto CrewOutput
            report_text = str(crew_result) if crew_result else "Nessun output"

            result = {
                "task_input": task_input,
                "final_report": report_text,
                "tokens_total": getattr(crew_result, "token_usage", {}).get("total_tokens", 0) if hasattr(crew_result, "token_usage") else 0,
                "mode": "crewai",
                "duration_ms": round(duration_ms, 1),
            }
        except Exception as exc:
            print(f"\n⚠ Errore CrewAI: {exc}")
            print("  Fallback a simulazione LLM...\n")
            result = _simulate_crew(task_input, fast=False)
            result["duration_ms"] = round((time.monotonic() - t0) * 1000, 1)
    elif fast:
        result = _mock_crew_result(task_input)
        result["duration_ms"] = round((time.monotonic() - t0) * 1000, 1)
    else:
        result = _simulate_crew(task_input, fast=fast)
        result["duration_ms"] = round((time.monotonic() - t0) * 1000, 1)

    # Output
    print(f"\n{'═'*65}")
    print("REPORT FINALE:\n")
    print(result["final_report"])
    print(f"\n{'─'*65}")
    print(f"Token totali: {result.get('tokens_total', 'N/A')}")
    print(f"Costo stimato: ${result.get('cost_usd', 0):.6f}")
    print(f"Durata: {result.get('duration_ms', 0)} ms")
    print(f"Modalità: {result.get('mode', '?')}")

    # Salva run
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_file = RUNS_DIR / f"crew_{ts}_{uuid.uuid4().hex[:6]}.json"
    run_file.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n📁 Run salvata: {run_file}")


def cmd_examples(args):
    print("""
╔══════════════════════════════════════════════════════════════════╗
║  LAB BONUS — Giorno 4 · CrewAI · Esempi                         ║
╚══════════════════════════════════════════════════════════════════╝

# Crew su ticket specifico (fast mode per demo)
python day4_enterprise/bonus_crewai.py crew "INC-1002" --fast

# Crew con LLM reale
python day4_enterprise/bonus_crewai.py crew "Analizza INC-1002 e fornisci piano di recovery"
python day4_enterprise/bonus_crewai.py crew "Quali ticket P1 sono aperti? Cosa devo fare prima?"

# Confronto paradigmi (no LLM)
python day4_enterprise/bonus_crewai.py compare-paradigms

INSTALLAZIONE CREWAI:
    pip install crewai crewai-tools

ALTERNATIVE AL BONUS (tutti e tre usano lo stesso dominio ITSM):
    • LlamaIndex: sostituisce il search_kb con RAG pipeline completa
      pip install llama-index llama-index-llms-google
    • AutoGen: conversazione multi-agent (agenti si parlano tramite messaggi)
      pip install pyautogen
    • Google Vertex AI Agent Builder: versione managed su GCP

SETUP LANGFUSE (per osservabilità sulla crew):
    pip install langfuse
    # .env: LANGFUSE_SECRET_KEY=sk-lf-... LANGFUSE_PUBLIC_KEY=pk-lf-...
    # CrewAI supporta Langfuse callback nativo:
    #   from langfuse.crewai import LangfuseCallbackHandler
    #   crew = Crew(..., callbacks=[LangfuseCallbackHandler()])
""")


def main():
    parser = argparse.ArgumentParser(
        description="Day 4 — Lab Bonus: CrewAI ITSM Multi-Agent"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_crew = sub.add_parser("crew", help="Avvia la crew ITSM")
    p_crew.add_argument("task", help="Task/domanda ITSM da assegnare alla crew")
    p_crew.add_argument("--fast", action="store_true", help="Modalità mock senza LLM")

    sub.add_parser("compare-paradigms", help="Confronto teorico LangGraph vs CrewAI")
    sub.add_parser("examples", help="Mostra esempi di comandi")

    args = parser.parse_args()

    {
        "crew": cmd_crew,
        "compare-paradigms": cmd_compare_paradigms,
        "examples": cmd_examples,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
