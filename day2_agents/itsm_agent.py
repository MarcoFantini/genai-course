from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
import time
from datetime import datetime, timedelta
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_vertexai import ChatVertexAI
from pydantic import BaseModel, Field
from typing_extensions import Annotated

try:
    from langgraph.graph.message import add_messages
except Exception:
    # LangGraph non è obbligatorio per il loop manuale.
    # Questo fallback permette almeno di importare il file se LangGraph non è installato.
    def add_messages(x: Any) -> Any:  # type: ignore
        return x


"""
day2_agents/itsm_agent.py

Esercitazione Giorno 2:
- la RAG del Giorno 1 diventa un tool dentro un agent
- tool calling/function calling
- loop ReAct manuale: Reason -> Act -> Observe -> Reason ...
- tool registry
- stop conditions
- gestione errori robusta
- trace strutturate
- stato custom per evoluzione LangGraph
- Human-in-the-loop con edge condizionale per azioni critiche

Prima di usare il tool RAG:
    python day2_agents/itsm_agent.py setup-rag-data --ingest

Oppure, manualmente:
    python day1_morning_rag/main.py setup-data
    python day1_morning_rag/main.py ingest

Modalità:
    manual  -> esegue il loop ReAct scritto a mano, utile per capire cosa succede sotto LangGraph
    graph   -> esegue una versione LangGraph con stato, checkpoint in memoria e HITL didattico

Comandi:
    python day2_agents/itsm_agent.py setup-rag-data --ingest
    python day2_agents/itsm_agent.py examples
    python day2_agents/itsm_agent.py manual "Mostrami INC-1002 e calcola lo SLA."
    python day2_agents/itsm_agent.py graph "Mostrami INC-1002, calcola lo SLA e proponi l'azione." --auto-decision approve
    (opzionale) python day2_agents/itsm_agent.py graph "Mostrami INC-1002, calcola lo SLA e proponi l'azione." --auto-decision reject

Variabili .env:
    GOOGLE_APPLICATION_CREDENTIALS=./service_account.json
    GOOGLE_CLOUD_PROJECT=hclsw-gcp-wrkld-auto
    GOOGLE_CLOUD_LOCATION=us-east1
    GEMINI_MODEL=gemini-2.5-flash
    MIN_SECONDS_BETWEEN_MODEL_CALLS=15
    TOP_K=3
    ALLOW_FALLBACK_KB=true

Nota didattica:
    LangChain usa la docstring di una funzione decorata con @tool come descrizione base
    del tool. Per questo le docstring dei tool sono volutamente esplicite:
    aiutano il modello a decidere quando chiamare quale strumento.
"""


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
# Import della RAG del Giorno 1
# ---------------------------------------------------------------------

DAY1_IMPORT_ERROR: Optional[Exception] = None
day1_retrieve = None
day1_ingest = None
DAY1_DATA_DIR: Optional[Path] = None

try:
    from day1_morning_rag.main import DATA_DIR as _DAY1_DATA_DIR
    from day1_morning_rag.main import ingest as _day1_ingest
    from day1_morning_rag.main import retrieve as _day1_retrieve

    DAY1_DATA_DIR = Path(_DAY1_DATA_DIR)
    day1_ingest = _day1_ingest
    day1_retrieve = _day1_retrieve

except Exception as exc:
    DAY1_IMPORT_ERROR = exc


# ---------------------------------------------------------------------
# Knowledge base didattica estesa
# ---------------------------------------------------------------------
# Questa KB può essere scritta dentro day1_morning_rag/data e indicizzata con
# la RAG del Giorno 1. Il tool search_kb_tool userà poi day1_morning_rag.main.retrieve().

SAMPLE_DOCS: Dict[str, str] = {
    "itsm_incident_policy.md": """
# ITSM Incident Management Policy

Questa policy definisce le regole operative per la gestione di incidenti,
malfunzionamenti applicativi, disservizi infrastrutturali e interruzioni
di servizi business-critical.

## Definizione di incidente

Un incidente è qualsiasi evento non pianificato che interrompe o degrada
la qualità di un servizio IT.

Sono incidenti, per esempio:
- indisponibilità di un servizio di posta elettronica;
- errore applicativo in produzione;
- degrado prestazionale percepito dagli utenti;
- failure di job batch critici;
- mancato accesso a sistemi aziendali;
- indisponibilità di API usate da processi business;
- rallentamento significativo di una piattaforma interna.

## Informazioni minime obbligatorie

Un ticket incidente deve contenere:
- servizio coinvolto;
- ambiente interessato, per esempio produzione, collaudo o sviluppo;
- data e ora di inizio del problema;
- numero stimato di utenti impattati;
- impatto business;
- eventuale workaround disponibile;
- log, screenshot o messaggi di errore disponibili;
- priorità proposta dal richiedente.

## Classificazione iniziale

La classificazione iniziale può essere proposta dall'utente, ma deve essere
verificata dal team ITSM. Il team può modificare priorità, componente,
owner e assegnatario se le informazioni raccolte indicano un impatto
diverso da quello dichiarato.

## Comunicazione

Per incidenti P1 e P2 deve essere mantenuto un aggiornamento periodico
nel ticket. Gli aggiornamenti devono indicare:
- stato corrente;
- ipotesi di causa;
- azioni in corso;
- workaround disponibile;
- prossimo checkpoint;
- eventuale owner tecnico o gestionale.

## Chiusura

Un incidente può essere chiuso solo quando:
- il servizio è stato ripristinato;
- l'utente o il service owner ha confermato il ripristino;
- sono state documentate causa, azione correttiva e impatto;
- per P1 e P2 è stata valutata la necessità di problem management.
""",
    "itsm_sla_policy.md": """
# ITSM SLA Policy

Questa policy definisce le soglie di presa in carico, aggiornamento ed
escalation per i ticket ITSM.

## Priorità P1

Un ticket P1 indica un incidente critico con impatto su produzione,
servizi essenziali, revenue, sicurezza o un numero significativo di utenti.

Esempi di P1:
- servizio e-mail aziendale indisponibile per molti utenti;
- impossibilità di accedere al sistema di produzione;
- blocco di un batch critico di fatturazione;
- indisponibilità di una API usata da canali digitali;
- incidente con possibile impatto di sicurezza;
- indisponibilità di sistemi usati per chiusure contabili o reporting executive.

SLA di presa in carico P1:
- presa in carico entro 30 minuti;
- aggiornamento operativo ogni 30 minuti;
- escalation al team on-call se non preso in carico entro 30 minuti;
- coinvolgimento del service manager se non esiste workaround entro 60 minuti.

## Priorità P2

Un ticket P2 indica un problema rilevante con impatto limitato oppure
con workaround disponibile.

SLA di presa in carico P2:
- presa in carico entro 4 ore lavorative;
- aggiornamento entro la giornata lavorativa;
- escalation al service manager se non aggiornato entro 4 ore lavorative.

## Priorità P3

Un ticket P3 indica una richiesta ordinaria, un'anomalia non bloccante
o un'attività pianificabile.

SLA di presa in carico P3:
- presa in carico entro 24 ore lavorative;
- aggiornamento secondo pianificazione del team.

## Priorità P4

Un ticket P4 indica attività informativa, richiesta minore o backlog item
senza impatto immediato sul servizio.

SLA di presa in carico P4:
- presa in carico entro 72 ore lavorative;
- aggiornamento secondo priorità di backlog.

## Near breach

Un ticket è considerato near breach quando il tempo residuo è inferiore
o uguale al 20% della soglia SLA applicabile. In stato near breach il team
deve verificare owner, prossimo passo e necessità di escalation preventiva.

## Violazione SLA

Un ticket è in violazione SLA quando il tempo trascorso supera la soglia
applicabile alla priorità del ticket. In caso di violazione:
- deve essere verificato l'owner;
- deve essere aggiornata la comunicazione;
- deve essere valutata escalation tecnica o gestionale;
- per P1 è richiesta attenzione immediata.
""",
    "itsm_escalation_policy.md": """
# ITSM Escalation Policy

L'escalation è il processo con cui un ticket viene portato all'attenzione
di un livello superiore di responsabilità tecnica o gestionale.

## Escalation tecnica

L'escalation tecnica deve essere usata quando:
- il team assegnatario non possiede le competenze necessarie;
- il problema richiede un team specialistico, per esempio SRE, network,
  database, security, mainframe o application owner;
- il workaround non è disponibile;
- i log indicano un fault infrastrutturale o applicativo complesso;
- il problema coinvolge più componenti e non è chiaro il punto di failure.

## Escalation gestionale

L'escalation gestionale deve essere usata quando:
- lo SLA è violato;
- un P1 non ha owner effettivo;
- il business impact è alto;
- il cliente interno richiede visibilità executive;
- più team sono coinvolti e manca coordinamento;
- la comunicazione verso stakeholder business non è sufficiente.

## Azioni critiche

Le seguenti azioni sono considerate critiche e richiedono conferma umana:
- aprire una escalation formale verso il team on-call;
- notificare il service manager;
- cambiare priorità di un ticket verso P1;
- avviare una comunicazione di major incident;
- chiudere un ticket P1 o P2;
- applicare un workaround con impatto su produzione;
- approvare un emergency change.

L'agente AI può proporre queste azioni, ma non deve eseguirle senza
approvazione esplicita di un operatore umano.

## Contenuto minimo di una escalation

Una escalation deve includere:
- ticket id;
- priorità;
- servizio coinvolto;
- owner corrente;
- impatto business;
- stato SLA;
- motivo dell'escalation;
- azione richiesta al team destinatario;
- prossimo checkpoint temporale.
""",
    "change_management_policy.md": """
# Change Management Policy

Un change è una modifica controllata a un servizio, una configurazione,
un'infrastruttura o un'applicazione.

## Standard change

Uno standard change è pre-approvato, ripetibile e a basso rischio.

Esempi:
- riavvio controllato di un servizio non critico;
- rotazione ordinaria di certificati già pianificata;
- aggiornamento di configurazione documentato e reversibile;
- deploy di patch minori già validate.

## Normal change

Un normal change richiede approvazione secondo processo standard.
Deve includere:
- descrizione;
- motivazione;
- rischio;
- piano di test;
- piano di rollback;
- finestra di esecuzione;
- impatto previsto;
- approvatore.

## Emergency change

Un emergency change può essere richiesto durante un incidente critico,
ma deve essere tracciato e giustificato.

Deve indicare:
- incidente collegato;
- rischio della modifica;
- piano di rollback;
- approvatore;
- finestra temporale;
- evidenza post-change.

## Relazione con incidenti

Per un P1 senza workaround, un emergency change può essere proposto
dal team tecnico. L'agente AI può suggerire la necessità di valutare
un emergency change, ma non può approvarlo autonomamente.
""",
    "on_call_policy.md": """
# On-call and Major Incident Policy

Il team on-call garantisce copertura per incidenti critici fuori orario
o per servizi classificati business-critical.

## Attivazione on-call

Il team on-call deve essere attivato quando:
- un P1 non è preso in carico entro 30 minuti;
- un servizio critico è indisponibile;
- un incidente ha impatto su sicurezza, revenue o obblighi contrattuali;
- più sistemi correlati risultano degradati;
- il service owner richiede supporto immediato.

## Major incident

Un major incident deve essere valutato quando:
- il problema impatta molti utenti;
- il servizio essenziale è indisponibile;
- non esiste workaround;
- la comunicazione verso stakeholder business è necessaria;
- il danno potenziale supera la normale operatività del team di supporto.

La dichiarazione di major incident richiede approvazione umana.

## Comunicazione major incident

La comunicazione deve includere:
- descrizione sintetica del problema;
- servizi impattati;
- utenti o business unit impattate;
- workaround disponibile o assente;
- owner tecnico;
- prossimo aggiornamento previsto;
- canale di comunicazione ufficiale.

## Fine major incident

La chiusura di un major incident richiede:
- conferma del ripristino;
- aggiornamento finale agli stakeholder;
- apertura eventuale di problem record;
- retrospettiva post-incident.
""",
    "knowledge_article_guidelines.md": """
# Knowledge Article Guidelines

Una knowledge article deve contenere:
- sintomi osservabili;
- causa nota o ipotesi principale;
- passaggi di diagnosi;
- workaround;
- soluzione definitiva;
- log o comandi utili;
- servizi e componenti coinvolti;
- data ultimo aggiornamento.

## Uso da parte dell'agente AI

L'agente AI può cercare knowledge article e proporre una risposta al ticket.
Se la risposta è basata su documentazione, deve indicare le fonti.
Se la documentazione non contiene informazioni sufficienti, l'agente
deve dichiarare che non trova evidenza nei documenti disponibili.

## Qualità della risposta

Una risposta basata su knowledge article deve:
- distinguere fatto documentato da ipotesi;
- non promettere risoluzioni non presenti nella documentazione;
- suggerire raccolta log se il contesto è insufficiente;
- proporre escalation se la policy lo richiede;
- evitare azioni distruttive senza approvazione umana.

## Aggiornamento knowledge base

Dopo un incidente P1 o P2, il team dovrebbe valutare se:
- creare una nuova knowledge article;
- aggiornare una knowledge article esistente;
- collegare l'incidente a un problem record;
- documentare workaround e causa radice.
""",
}


# ---------------------------------------------------------------------
# Modelli Pydantic
# ---------------------------------------------------------------------

class Comment(BaseModel):
    author: str
    text: str
    created_at: str


class TicketRecord(BaseModel):
    id: str
    key: str
    record_type: str = "incident"
    summary: str
    description: str
    priority: str = Field(pattern=r"^P[1-4]$")
    status: str
    service: str
    environment: str
    component: str
    elapsed_hours: float = Field(ge=0)
    owner: str
    assignee: Optional[str] = None
    reporter: str
    affected_users: int = Field(ge=0)
    business_impact: str
    workaround_available: bool
    labels: List[str] = Field(default_factory=list)
    linked_records: List[str] = Field(default_factory=list)
    comments: List[Comment] = Field(default_factory=list)
    opened_at: datetime


class KBHit(BaseModel):
    source: str
    snippet: str
    score: Optional[float] = None
    distance: Optional[float] = None
    chunk_index: Optional[int] = None


class SLAResult(BaseModel):
    id: str
    priority: str
    threshold_hours: float
    elapsed_hours: float
    elapsed_hours_buisness: float
    elapsed_hours_tot: float
    remaining_hours: float
    status: str
    recommendation: str
    critical: bool
    reason: str


class PendingAction(BaseModel):
    action_type: str
    ticket_id: str
    priority: str
    owner: str
    critical: bool
    reason: str


class ToolResponse(BaseModel):
    success: bool
    results: List[Any] = Field(default_factory=list)
    error: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


class TraceEvent(BaseModel):
    step: int
    event: str
    timestamp: float
    tool: Optional[str] = None
    args: Dict[str, Any] = Field(default_factory=dict)
    result: Optional[Any] = None
    error: Optional[str] = None
    text: Optional[str] = None


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    ticket: Optional[dict]
    sla: Optional[dict]
    pending_action: Optional[dict]
    approved: Optional[bool]
    traces: List[dict]


def model_to_dict(model: BaseModel) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def to_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------
# Database operativo simulato, stile Jira / ITSM
# ---------------------------------------------------------------------

SAMPLE_RECORDS: Dict[str, TicketRecord] = {
    "INC-1002": TicketRecord(
        id="INC-1002",
        key="ITSM-1002",
        summary="Interruzione servizio e-mail per utenti area Finance",
        description=(
            "Diversi utenti dell'area Finance non riescono ad accedere alla posta. "
            "Il client mostra timeout e la webmail restituisce errore 503."
        ),
        priority="P1",
        status="In Progress",
        service="Corporate Email",
        environment="production",
        component="mail-gateway",
        elapsed_hours=1.0,
        owner="team-sre",
        assignee="m.rossi",
        reporter="finance.ops",
        affected_users=180,
        business_impact="Chiusura mensile rallentata; rischio ritardo report CFO.",
        workaround_available=False,
        labels=["email", "finance", "production", "p1"],
        linked_records=["CHG-2201"],
        opened_at=datetime.fromisoformat("2026-05-12T22:15:00"),
        comments=[
            Comment(
                author="finance.ops",
                text="Impatto confermato su più uffici.",
                created_at="2026-05-11T09:20:00",
            ),
            Comment(
                author="team-sre",
                text="Verifica in corso su mail gateway e bilanciatore.",
                created_at="2026-05-11T09:42:00",
            ),
        ],
    ),
    "INC-1003": TicketRecord(
        id="INC-1003",
        key="ITSM-1003",
        summary="Degrado prestazionale API ordini procurement",
        description=(
            "Le API di creazione ordine rispondono lentamente. Il servizio è disponibile, "
            "ma alcune chiamate superano 12 secondi."
        ),
        priority="P2",
        status="Open",
        service="Procurement Platform",
        environment="production",
        component="purchase-order-api",
        elapsed_hours=3.6,
        owner="team-app-procurement",
        assignee=None,
        reporter="s2p.business",
        affected_users=35,
        business_impact=(
            "Rallentamento nella creazione ordini; workaround tramite inserimento manuale disponibile."
        ),
        workaround_available=True,
        labels=["api", "procurement", "latency", "p2"],
        linked_records=[],
        opened_at=datetime.fromisoformat("2026-05-10T09:15:00"),
        comments=[
            Comment(
                author="s2p.business",
                text="Il workaround è scomodo ma utilizzabile.",
                created_at="2026-05-11T10:15:00",
            ),
        ],
    ),
    "INC-1004": TicketRecord(
        id="INC-1004",
        key="ITSM-1004",
        summary="Richiesta accesso dashboard sales",
        description="Nuovo utente richiede accesso read-only alla dashboard commerciale.",
        priority="P3",
        status="Open",
        service="Sales Analytics",
        environment="production",
        component="bi-dashboard",
        elapsed_hours=5.0,
        owner="team-bi",
        assignee="l.bianchi",
        reporter="sales.ops",
        affected_users=1,
        business_impact="Nessun impatto bloccante; richiesta ordinaria.",
        workaround_available=True,
        labels=["access-request", "sales", "p3"],
        linked_records=[],
        opened_at=datetime.fromisoformat("2026-05-10T09:15:00"),
        comments=[],
    ),
    "INC-1005": TicketRecord(
        id="INC-1005",
        key="ITSM-1005",
        summary="ABEND su job batch di fatturazione notturna",
        description=(
            "Il job batch BILLING_CLOSE_01 ha terminato con ABEND. "
            "La catena successiva di reportistica non è partita."
        ),
        priority="P1",
        status="Open",
        service="Billing Batch",
        environment="production",
        component="mainframe-batch",
        elapsed_hours=0.35,
        owner="team-mainframe",
        assignee=None,
        reporter="batch.monitoring",
        affected_users=12,
        business_impact=(
            "Rischio ritardo produzione report giornalieri e riconciliazione ricavi."
        ),
        workaround_available=False,
        labels=["mainframe", "billing", "abend", "p1"],
        linked_records=["PRB-778"],
        opened_at=datetime.fromisoformat("2026-05-10T09:15:00"),
        comments=[
            Comment(
                author="batch.monitoring",
                text="Codice ABEND rilevato: S0C7.",
                created_at="2026-05-11T06:10:00",
            ),
        ],
    ),
}


# ---------------------------------------------------------------------
# Dataset RAG del Giorno 1: setup opzionale
# ---------------------------------------------------------------------

def setup_rag_data(ingest_now: bool = False) -> None:
    if DAY1_DATA_DIR is None:
        raise RuntimeError(
            "Non riesco a importare day1_morning_rag.main. "
            f"Errore originale: {DAY1_IMPORT_ERROR}"
        )

    DAY1_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for filename, content in SAMPLE_DOCS.items():
        path = DAY1_DATA_DIR / filename
        path.write_text(textwrap.dedent(content).strip(), encoding="utf-8")

    print(f"Documenti ITSM estesi scritti in: {DAY1_DATA_DIR}")

    if ingest_now:
        if day1_ingest is None:
            raise RuntimeError("Funzione ingest del Giorno 1 non disponibile.")

        print("Avvio ingest della RAG del Giorno 1...")
        day1_ingest()


# ---------------------------------------------------------------------
# Funzioni di dominio
# ---------------------------------------------------------------------

def fallback_keyword_search(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    q = (query or "").lower().strip()
    hits: List[KBHit] = []

    for source, text in SAMPLE_DOCS.items():
        tokens = [
            tok
            for tok in re.findall(r"[a-zA-ZÀ-ÿ0-9]+", q)
            if len(tok) > 1
        ]

        score = sum(1 for tok in tokens if tok in text.lower())

        if score > 0:
            snippet = re.sub(r"\s+", " ", text).strip()
            hits.append(
                KBHit(
                    source=source,
                    snippet=snippet[:900],
                    score=float(score),
                    distance=None,
                    chunk_index=None,
                )
            )

    hits.sort(key=lambda x: x.score or 0.0, reverse=True)
    return [model_to_dict(hit) for hit in hits[:top_k]]


def search_kb(query: str, top_k: int = 3) -> Dict[str, Any]:
    q = (query or "").strip()

    if not q:
        return model_to_dict(
            ToolResponse(
                success=False,
                error="query vuota",
            )
        )

    if day1_retrieve is not None:
        try:
            chunks = day1_retrieve(q, top_k=top_k)

            hits = [
                model_to_dict(
                    KBHit(
                        source=str(chunk.get("source", "unknown")),
                        snippet=str(chunk.get("text", "")),
                        distance=float(chunk.get("distance", 0.0)),
                        chunk_index=int(chunk.get("chunk_index", -1)),
                    )
                )
                for chunk in chunks
            ]

            return model_to_dict(
                ToolResponse(
                    success=True,
                    results=hits,
                    error=None,
                    meta={
                        "retriever": "day1_morning_rag.main.retrieve",
                        "top_k": top_k,
                    },
                )
            )

        except Exception as exc:
            if not ALLOW_FALLBACK_KB:
                return model_to_dict(
                    ToolResponse(
                        success=False,
                        error=(
                            "Errore nella RAG del Giorno 1. "
                            "Esegui: python day1_morning_rag/main.py ingest. "
                            f"Dettaglio: {exc}"
                        ),
                        meta={"retriever": "day1_morning_rag.main.retrieve"},
                    )
                )

            fallback_hits = fallback_keyword_search(q, top_k)

            return model_to_dict(
                ToolResponse(
                    success=bool(fallback_hits),
                    results=fallback_hits,
                    error=(
                        None
                        if fallback_hits
                        else f"RAG non disponibile e fallback senza risultati. Dettaglio RAG: {exc}"
                    ),
                    meta={
                        "retriever": "fallback_keyword_search",
                        "warning": (
                            "La RAG del Giorno 1 ha generato un errore; "
                            "usato fallback lessicale."
                        ),
                    },
                )
            )

    if not ALLOW_FALLBACK_KB:
        return model_to_dict(
            ToolResponse(
                success=False,
                error=(
                    "day1_morning_rag.main non importabile e fallback disabilitato. "
                    f"Errore import: {DAY1_IMPORT_ERROR}"
                ),
            )
        )

    fallback_hits = fallback_keyword_search(q, top_k)

    return model_to_dict(
        ToolResponse(
            success=bool(fallback_hits),
            results=fallback_hits,
            error=None if fallback_hits else "Nessun risultato nella KB fallback.",
            meta={
                "retriever": "fallback_keyword_search",
                "warning": "day1_morning_rag.main.retrieve non disponibile.",
            },
        )
    )


def lookup_record(record_id: str) -> Dict[str, Any]:
    rid = (record_id or "").strip().upper()

    if not rid:
        return model_to_dict(
            ToolResponse(
                success=False,
                error="record_id vuoto",
            )
        )

    rec = SAMPLE_RECORDS.get(rid)

    if not rec:
        return model_to_dict(
            ToolResponse(
                success=False,
                results=[],
                error=f"Record {rid} non trovato",
                meta={"available_records": sorted(SAMPLE_RECORDS.keys())},
            )
        )

    return model_to_dict(
        ToolResponse(
            success=True,
            results=[model_to_dict(rec)],
            error=None,
        )
    )


def compute_sla(ticket: Dict[str, Any]) -> Dict[str, Any]:
    try:
        normalized = TicketRecord(**ticket)
    except Exception as exc:
        return model_to_dict(
            ToolResponse(
                success=False,
                error=f"Ticket non valido secondo lo schema operativo: {exc}",
            )
        )

    thresholds = {
        "P1": 0.5,
        "P2": 4.0,
        "P3": 24.0,
        "P4": 72.0,
    }
    
    threshold = thresholds.get(normalized.priority, 24.0)
    diff = datetime.now() - normalized.opened_at
    buisness_hours_diff = diff.total_seconds() / 3600
    
    if normalized.priority != "P1" and normalized.opened_at is not None:
        buisness_hours_diff = buisness_hours_elapsed(normalized.opened_at)
        print("con quella data e priorità considero ", buisness_hours_diff, " buisness hours")

    remaining = round(threshold - normalized.elapsed_hours - buisness_hours_diff, 3)

    if remaining < 0:
        status = "violated"
    elif remaining <= threshold * 0.2:
        status = "near_breach"
    else:
        status = "ok"

    if status == "violated":
        recommendation = "escalate"
        reason = (
            f"SLA violato: priorità {normalized.priority}, soglia {threshold}h, "
            f"tempo trascorso {normalized.elapsed_hours + buisness_hours_diff}h."
        )
    elif status == "near_breach":
        recommendation = "prepare_escalation"
        reason = (
            f"Ticket vicino alla violazione SLA: rimangono {remaining}h "
            f"su soglia {threshold}h."
        )
    else:
        recommendation = "continue"
        reason = (
            f"SLA ancora entro soglia: rimangono {remaining}h "
            f"su soglia {threshold}h."
        )

    critical = recommendation == "escalate" or (
        normalized.priority == "P1" and recommendation == "prepare_escalation"
    )

    sla = SLAResult(
        id=normalized.id,
        priority=normalized.priority,
        threshold_hours=threshold,
        elapsed_hours=normalized.elapsed_hours,
        elapsed_hours_tot=normalized.elapsed_hours + buisness_hours_diff,
        elapsed_hours_buisness=buisness_hours_diff,
        remaining_hours=remaining,
        status=status,
        recommendation=recommendation,
        critical=critical,
        reason=reason,
    )

    return model_to_dict(
        ToolResponse(
            success=True,
            results=[model_to_dict(sla)],
            error=None,
        )
    )


def build_pending_action(
    ticket: Dict[str, Any],
    sla: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if not sla:
        return None

    recommendation = sla.get("recommendation")
    critical = bool(sla.get("critical"))

    if not critical:
        return None

    action = PendingAction(
        action_type="open_formal_escalation",
        ticket_id=str(ticket.get("id", sla.get("id", "unknown"))),
        priority=str(ticket.get("priority", sla.get("priority", "unknown"))),
        owner=str(ticket.get("owner", "")),
        critical=True,
        reason=str(
            sla.get(
                "reason",
                "Azione critica richiesta dalla policy SLA.",
            )
        ),
    )

    return model_to_dict(action)


def execute_critical_action(
    action: Dict[str, Any],
    approved: bool,
) -> Dict[str, Any]:
    if not action:
        return model_to_dict(
            ToolResponse(
                success=False,
                error="Nessuna azione pendente.",
            )
        )

    if not approved:
        return model_to_dict(
            ToolResponse(
                success=True,
                results=[
                    {
                        "executed": False,
                        "action": action,
                        "message": (
                            "Azione critica non eseguita perché non approvata "
                            "dall'operatore umano."
                        ),
                    }
                ],
            )
        )

    return model_to_dict(
        ToolResponse(
            success=True,
            results=[
                {
                    "executed": True,
                    "action": action,
                    "message": (
                        f"Escalation formale simulata per ticket {action.get('ticket_id')} "
                        f"verso owner {action.get('owner')}."
                    ),
                }
            ],
        )
    )


# ---------------------------------------------------------------------
# Tool LangChain
# ---------------------------------------------------------------------

@tool
def search_kb_tool(query: str, top_k: int = 3) -> str:
    """
    Cerca policy, procedure, SLA, escalation e knowledge article nella knowledge base ITSM.
    Usa la RAG del Giorno 1 tramite day1_morning_rag.main.retrieve().

    Nota didattica:
    LangChain usa questa docstring come descrizione base del tool quando il decoratore
    @tool viene usato nella forma semplice.
    """
    return to_json(search_kb(query, top_k))


@tool
def lookup_record_tool(record_id: str) -> str:
    """
    Recupera un record operativo ITSM/Jira-like dato un id, per esempio INC-1002.
    Restituisce dati strutturati del ticket: priorità, stato, owner, servizio,
    ambiente, impatto business, tempo trascorso, commenti e record collegati.

    Nota didattica:
    LangChain usa questa docstring come descrizione base del tool quando il decoratore
    @tool viene usato nella forma semplice.
    """
    return to_json(lookup_record(record_id))


@tool
def compute_sla_tool(
    id: str,
    priority: str,
    elapsed_hours: float,
    owner: str = "",
    key: str = "UNKNOWN",
    summary: str = "",
    description: str = "",
    status: str = "Open",
    service: str = "unknown",
    environment: str = "production",
    component: str = "unknown",
    assignee: Optional[str] = None,
    reporter: str = "unknown",
    affected_users: int = 0,
    business_impact: str = "",
    workaround_available: bool = False,
    opened_at: datetime = datetime.now()
) -> str:
    """
    Calcola stato SLA e raccomandazione operativa per un ticket ITSM.
    Accetta campi strutturati invece di una stringa JSON annidata.

    Usalo dopo lookup_record_tool quando devi valutare:
    - violazione SLA;
    - near breach;
    - necessità di escalation;
    - priorità operativa del prossimo passo.

    Nota didattica:
    LangChain usa questa docstring come descrizione base del tool quando il decoratore
    @tool viene usato nella forma semplice.
    """
    ticket = {
        "id": id,
        "key": key,
        "record_type": "incident",
        "summary": summary or f"Ticket {id}",
        "description": description or summary or f"Ticket {id}",
        "priority": priority,
        "status": status,
        "service": service,
        "environment": environment,
        "component": component,
        "elapsed_hours": elapsed_hours,
        "owner": owner,
        "assignee": assignee,
        "reporter": reporter,
        "affected_users": affected_users,
        "business_impact": business_impact,
        "workaround_available": workaround_available,
        "opened_at": opened_at,
        "labels": [],
        "linked_records": [],
        "comments": [],
    }

    return to_json(compute_sla(ticket))

@tool
def list_opens_ticket_tools(priority: Optional[str] = None) -> str:
    """
    Questo tool restituisce una lista di ticket aperti filtrati per priorità
    Un ticket Aperto vuol dire stato diverso da Closed
    se non viene fornita la priorità li restituisce tutti

    Nota didattica:
    LangChain usa questa docstring come descrizione base del tool quando il decoratore
    @tool viene usato nella forma semplice.
    """
    open_tickets = [
        rec for rec in SAMPLE_RECORDS.values()
        if rec.status != "Closed"
    ]
    
    if priority:
        open_tickets = [t for t in open_tickets if t.priority == priority]
    
    return to_json(
        model_to_dict(
            ToolResponse(
                success=True,
                results=[t.model_dump(include={"id", "summary", "priority", "status", "elapsed_hours"}) for t in open_tickets],
                error=None,
            )
        )
    )

TOOLS = [search_kb_tool, lookup_record_tool, compute_sla_tool, list_opens_ticket_tools]
TOOL_MAP = {t.name: t for t in TOOLS}


# ---------------------------------------------------------------------
# Prompt e messaggi iniziali
# ---------------------------------------------------------------------

SYSTEM_PROMPT = """
Sei un agente ITSM enterprise per un laboratorio di sviluppo software.

Obiettivo:
- aiutare l'operatore a interpretare policy ITSM, record operativi e SLA;
- usare strumenti quando servono dati aggiornati, calcoli o documentazione;
- distinguere chiaramente tra evidenza documentale, dato operativo e inferenza.

Regole operative:
1. Se la domanda riguarda policy, SLA, escalation, on-call o knowledge article, usa search_kb_tool.
2. Se la domanda cita un record come INC-1002, recuperalo con lookup_record_tool.
3. Se devi valutare SLA, violazione o near breach, usa compute_sla_tool con campi strutturati.
4. Non inventare ticket, policy o tempi SLA.
5. Se un tool fallisce, spiega l'errore e proponi il prossimo passo ragionevole.
6. Le azioni critiche, come escalation formale, major incident o chiusura P1/P2,
   devono essere solo proposte: richiedono conferma umana.
7. Rispondi in italiano, con tono operativo e sintetico.
8. Quando usi documentazione, cita le fonti restituite dal tool RAG.

Formato consigliato della risposta finale:
- Sintesi
- Evidenze usate
- Valutazione SLA, se richiesta
- Raccomandazione
- Eventuale conferma umana richiesta
""".strip()


def build_initial_messages(query: str) -> List[Any]:
    return [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            f"""
Richiesta utente:
{query}

Contesto didattico:
- Questo agent può rispondere direttamente solo se la richiesta è generale.
- Se servono policy, record o calcoli SLA, deve usare i tool.
- La RAG del Giorno 1 è disponibile come search_kb_tool.
- Il database operativo simulato è disponibile come lookup_record_tool.
- Il calcolo SLA deterministico è disponibile come compute_sla_tool.
""".strip(),
        ),
    ]

# ---------------------------------------------------------------------
# Utility output
# ---------------------------------------------------------------------

def buisness_hours_elapsed(opened_at: datetime) -> float:
    """
    Calcola le ore trascorse tra opened_at e adesso, 
    escludendo i fine settimana.
    """
    # 1. Prendiamo il momento attuale (stessa timezone di opened_at se presente)
    now = datetime.now(opened_at.tzinfo) if opened_at.tzinfo else datetime.now()
    
    # Se opened_at è nel futuro rispetto a ora, restituiamo 0
    if opened_at >= now:
        return 0.0

    total_seconds = 0
    current_date = opened_at

    # 2. Cicliamo giorno per giorno
    while current_date.date() <= now.date():
        # weekday() restituisce 0 per Lunedì, ..., 5 per Sabato, 6 per Domenica
        if current_date.weekday() < 5:
            
            # Definiamo l'inizio e la fine del conteggio per il giorno corrente
            day_start = current_date
            
            # La fine del giorno corrente è la mezzanotte successiva 
            # o l'orario attuale se siamo nell'ultimo giorno
            day_end = datetime.combine(current_date.date(), datetime.max.time()).replace(tzinfo=opened_at.tzinfo)
            if day_end > now:
                day_end = now
                
            # Calcoliamo la differenza in secondi per questo giorno specifico
            delta = (day_end - day_start).total_seconds()
            if delta > 0:
                total_seconds += delta
        
        # Passiamo alla mezzanotte del giorno successivo
        current_date = datetime.combine(current_date.date() + timedelta(days=1), datetime.min.time()).replace(tzinfo=opened_at.tzinfo)

    # 3. Trasformiamo i secondi totali in ore
    return total_seconds / 3600

# Esempio di utilizzo:
# data_apertura = datetime(2023, 10, 20, 10, 0) # Un venerdì
# ore = business_hours_elapsed(data_apertura)
# print(f"Ore lavorative trascorse: {ore:.2f}")

def extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        chunks = []

        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                if item["text"].strip():
                    chunks.append(item["text"].strip())

            elif isinstance(item, str) and item.strip():
                chunks.append(item.strip())

        return "\n".join(chunks).strip()

    return str(content).strip()


def parse_tool_result(result: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(result)

        if isinstance(parsed, dict):
            return parsed

        return {
            "success": True,
            "results": [parsed],
            "error": None,
        }

    except Exception:
        return {
            "success": True,
            "results": [result],
            "error": None,
        }


def append_trace(
    traces: List[dict],
    *,
    step: int,
    event: str,
    tool_name: Optional[str] = None,
    args: Optional[Dict[str, Any]] = None,
    result: Optional[Any] = None,
    error: Optional[str] = None,
    text: Optional[str] = None,
) -> None:
    event_dict = model_to_dict(
        TraceEvent(
            step=step,
            event=event,
            tool=tool_name,
            args=args or {},
            result=result,
            error=error,
            text=text,
            timestamp=time.time(),
        )
    )
    traces.append(event_dict)
    # Notifica il callback SSE se attivo (impostato da api.py per streaming real-time).
    # Ogni append_trace sparso nel file beneficia automaticamente di questo hook.
    cb = _trace_callback.get()
    if cb is not None:
        cb(event_dict)


def execute_tool_call(
    tc: Dict[str, Any],
    step: int,
    traces: List[dict],
) -> str:
    name = tc.get("name")
    args = tc.get("args", {}) or {}

    if name not in TOOL_MAP:
        result = to_json(
            {
                "success": False,
                "error": f"Tool sconosciuto: {name}",
            }
        )
    else:
        try:
            result = TOOL_MAP[name].invoke(args)

        except Exception as exc:
            result = to_json(
                {
                    "success": False,
                    "error": f"Errore durante tool {name}: {exc}",
                }
            )

    append_trace(
        traces,
        step=step,
        event="tool_call",
        tool_name=name,
        args=args,
        result=parse_tool_result(result),
    )

    return result


# ---------------------------------------------------------------------
# Agent manuale: loop ReAct migliorato
# ---------------------------------------------------------------------

def run_real_agent(
    query: str,
    max_iter: int = 5,
) -> Tuple[str, List[dict]]:
    """
    Esegue un agent con tool calling usando un loop ReAct manuale.

    LOOP REACT USATO QUI:
    1. Reason:
       il modello legge messaggi, risultati precedenti e decide cosa fare.

    2. Act:
       se il modello richiede tool_calls, il runtime Python esegue i tool.

    3. Observe:
       il risultato dei tool viene aggiunto ai messaggi come ToolMessage.

    4. Repeat:
       il modello viene richiamato con le osservazioni aggiornate.

    5. Stop:
       se non ci sono tool_calls, la risposta testuale è finale.
    """
    traces: List[dict] = []

    try:
        llm = get_vertex_llm()
    except Exception as exc:
        append_trace(
            traces,
            step=0,
            event="configuration_error",
            error=str(exc),
        )
        return str(exc), traces

    llm_with_tools = llm.bind_tools(TOOLS)

    messages: List[Any] = build_initial_messages(query)

    start_time = time.time()
    last_call = 0.0
    total_tool_calls = 0

    for step in range(1, max_iter + 1):
        if time.time() - start_time > MAX_TOTAL_SECONDS:
            append_trace(
                traces,
                step=step,
                event="stop",
                error=f"Timeout agente raggiunto dopo {MAX_TOTAL_SECONDS}s",
            )
            return "Timeout agente raggiunto", traces

        now = time.time()
        wait_s = MIN_SECONDS_BETWEEN_MODEL_CALLS - (now - last_call)

        if wait_s > 0:
            time.sleep(wait_s)

        # -----------------------------------------------------------------
        # REASON
        # Il modello decide se rispondere direttamente o chiamare strumenti.
        # -----------------------------------------------------------------
        ai_msg = llm_with_tools.invoke(messages)
        last_call = time.time()
        messages.append(ai_msg)

        tool_calls = getattr(ai_msg, "tool_calls", None) or []
        text = extract_text_content(getattr(ai_msg, "content", ""))

        append_trace(
            traces,
            step=step,
            event="llm_response",
            text=text if text else "[tool-call only]",
            result={
                "tool_calls": [
                    {
                        "id": tc.get("id"),
                        "name": tc.get("name"),
                        "args": tc.get("args", {}),
                    }
                    for tc in tool_calls
                ]
            },
        )

        # -----------------------------------------------------------------
        # STOP CONDITION
        # Nessun tool richiesto => risposta finale.
        # -----------------------------------------------------------------
        if not tool_calls:
            return (text if text else "Nessuna risposta testuale"), traces

        total_tool_calls += len(tool_calls)

        if total_tool_calls > MAX_TOOL_CALLS:
            append_trace(
                traces,
                step=step,
                event="stop",
                error=(
                    f"Troppe chiamate tool: {total_tool_calls}. "
                    "Possibile loop agentico."
                ),
            )
            return "Troppe chiamate tool: possibile loop agentico", traces

        # -----------------------------------------------------------------
        # ACT + OBSERVE
        # Esecuzione tool e reinserimento del risultato nel contesto.
        # -----------------------------------------------------------------
        for tc in tool_calls:
            name = tc.get("name", "")
            result = execute_tool_call(tc, step, traces)

            messages.append(
                ToolMessage(
                    content=result,
                    name=name,
                    tool_call_id=tc.get("id", f"tool-call-{step}-{name}"),
                )
            )

    append_trace(
        traces,
        step=max_iter,
        event="stop",
        error="Max iterations raggiunte",
    )

    return "Max iterations raggiunte", traces


# ---------------------------------------------------------------------
# LangGraph: stato custom + edge condizionale + HITL
# ---------------------------------------------------------------------

def build_langgraph_agent():
    """
    Costruisce una versione LangGraph dell'agent.

    Grafo didattico:

        START
          |
          v
        agent
          |
          |-- se ci sono tool_calls --> tools
          |                              |
          |                              v
          |                           risk_check
          |                              |
          |                              |-- se azione critica --> human_approval
          |                              |                         |
          |                              |                         v
          |                              |                    execute_action
          |                              |                         |
          |                              |                         v
          |                              ----------------------> agent
          |
          |-- se non ci sono tool_calls --> END

    La parte Human-in-the-loop usa interrupt():
    il grafo si sospende quando serve conferma umana su un'azione critica.
    """
    try:
        from langgraph.graph import END, START, StateGraph
        from langgraph.types import Command, interrupt
        from langgraph.checkpoint.memory import MemorySaver

    except Exception as exc:
        raise RuntimeError(
            "LangGraph non è installato o non è importabile. "
            "Installa con: pip install langgraph. "
            f"Dettaglio: {exc}"
        )

    llm = get_vertex_llm()
    llm_with_tools = llm.bind_tools(TOOLS)

    def agent_node(state: AgentState) -> Dict[str, Any]:
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response = llm_with_tools.invoke(messages)

        traces = list(state.get("traces") or [])

        append_trace(
            traces,
            step=len(traces) + 1,
            event="graph_llm_response",
            text=(
                extract_text_content(getattr(response, "content", ""))
                or "[tool-call only]"
            ),
            result={
                "tool_calls": [
                    {
                        "id": tc.get("id"),
                        "name": tc.get("name"),
                        "args": tc.get("args", {}),
                    }
                    for tc in (getattr(response, "tool_calls", None) or [])
                ]
            },
        )

        return {
            "messages": [response],
            "traces": traces,
        }

    def route_after_agent(state: AgentState) -> str:
        last = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", None) or []
        return "tools" if tool_calls else END

    def tools_node(state: AgentState) -> Dict[str, Any]:
        last = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", None) or []

        tool_messages = []
        traces = list(state.get("traces") or [])
        ticket = state.get("ticket")
        sla = state.get("sla")

        step = len(traces) + 1

        for tc in tool_calls:
            result = execute_tool_call(tc, step, traces)
            parsed = parse_tool_result(result)
            name = tc.get("name", "")

            if (
                name == "lookup_record_tool"
                and parsed.get("success")
                and parsed.get("results")
            ):
                ticket = parsed["results"][0]

            if (
                name == "compute_sla_tool"
                and parsed.get("success")
                and parsed.get("results")
            ):
                sla = parsed["results"][0]

            tool_messages.append(
                ToolMessage(
                    content=result,
                    name=name,
                    tool_call_id=tc.get("id", f"graph-tool-call-{step}-{name}"),
                )
            )

        return {
            "messages": tool_messages,
            "ticket": ticket,
            "sla": sla,
            "traces": traces,
        }

    def risk_check_node(state: AgentState) -> Dict[str, Any]:
        ticket = state.get("ticket")
        sla = state.get("sla")
        pending_action = state.get("pending_action")

        traces = list(state.get("traces") or [])

        if ticket and sla and not pending_action:
            pending_action = build_pending_action(ticket, sla)

            if pending_action:
                append_trace(
                    traces,
                    step=len(traces) + 1,
                    event="critical_action_detected",
                    result=pending_action,
                )

        return {
            "pending_action": pending_action,
            "traces": traces,
        }

    def route_after_risk_check(state: AgentState) -> str:
        pending = state.get("pending_action")
        approved = state.get("approved")

        if pending and approved is None:
            return "human_approval"

        return "agent"

    def human_approval_node(state: AgentState) -> Dict[str, Any]:
        pending = state.get("pending_action")

        decision = interrupt(
            {
                "message": "Azione critica richiesta. Vuoi approvare?",
                "pending_action": pending,
                "allowed_decisions": ["approve", "reject"],
            }
        )

        if isinstance(decision, dict):
            raw_decision = str(decision.get("decision", "")).lower().strip()
        else:
            raw_decision = str(decision).lower().strip()

        approved = raw_decision in {
            "approve",
            "approved",
            "yes",
            "y",
            "si",
            "sì",
        }

        traces = list(state.get("traces") or [])

        append_trace(
            traces,
            step=len(traces) + 1,
            event="human_approval",
            result={
                "decision": raw_decision,
                "approved": approved,
            },
        )

        return {
            "approved": approved,
            "traces": traces,
        }

    def execute_action_node(state: AgentState) -> Dict[str, Any]:
        pending = state.get("pending_action")
        approved = bool(state.get("approved"))

        execution = execute_critical_action(pending or {}, approved)

        traces = list(state.get("traces") or [])

        append_trace(
            traces,
            step=len(traces) + 1,
            event="critical_action_execution",
            result=execution,
        )

        operator_message = HumanMessage(
            content=(
                "Esito controllo umano e azione critica:\n"
                f"{execution}\n\n"
                "Ora produci una risposta finale per l'utente indicando cosa è stato approvato "
                "o non approvato e quali evidenze hai usato."
            )
        )

        return {
            "messages": [operator_message],
            "traces": traces,
        }

    builder = StateGraph(AgentState)

    builder.add_node("agent", agent_node)
    builder.add_node("tools", tools_node)
    builder.add_node("risk_check", risk_check_node)
    builder.add_node("human_approval", human_approval_node)
    builder.add_node("execute_action", execute_action_node)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route_after_agent)
    builder.add_edge("tools", "risk_check")
    builder.add_conditional_edges("risk_check", route_after_risk_check)
    builder.add_edge("human_approval", "execute_action")
    builder.add_edge("execute_action", "agent")

    checkpointer = MemorySaver()
    graph = builder.compile(checkpointer=checkpointer)

    return graph, Command


# Module-level cache: (graph, Command) costruiti una sola volta per processo.
# Il MemorySaver all'interno del grafo persiste lo stato tra invocazioni con lo
# stesso thread_id, rendendo possibile la ripresa dopo un interrupt() senza
# perdere il checkpoint sospeso.
_langgraph_cache: Optional[Tuple[Any, Any]] = None


def get_langgraph_agent() -> Tuple[Any, Any]:
    """Restituisce (graph, Command), costruendo il grafo solo alla prima chiamata."""
    global _langgraph_cache
    if _langgraph_cache is None:
        _langgraph_cache = build_langgraph_agent()
    return _langgraph_cache


def run_graph_agent(
    query: str,
    thread_id: str = "demo-itsm-001",
    auto_decision: Optional[str] = None,
) -> Dict[str, Any]:
    # Usa il grafo cachato: lo stesso MemorySaver viene riusato tra chiamate,
    # quindi il checkpoint sospeso sopravvive e la ripresa con Command(resume=...)
    # funziona con lo stesso thread_id.
    graph, Command = get_langgraph_agent()

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    initial_state: AgentState = {
        "messages": [
            HumanMessage(content=query)
        ],
        "ticket": None,
        "sla": None,
        "pending_action": None,
        "approved": None,
        "traces": [],
    }

    result = graph.invoke(initial_state, config=config)

    interrupts = result.get("__interrupt__") if isinstance(result, dict) else None

    if interrupts and auto_decision:
        result = graph.invoke(
            Command(resume={"decision": auto_decision}),
            config=config,
        )

    return result

def resume_graph_agent(
    thread_id: str,
    decision: str,
) -> Dict[str, Any]:
    """
    Riprende un grafo sospeso su interrupt() con la decisione umana.
    Non riavvia il grafo: usa Command(resume=...) sullo stesso checkpoint.
    Funziona solo se get_langgraph_agent() è già stato chiamato in questo
    processo (stesso MemorySaver in memoria).
    """
    graph, Command = get_langgraph_agent()
    config = {"configurable": {"thread_id": thread_id}}
    return graph.invoke(
        Command(resume={"decision": decision}),
        config=config,
    )
# ---------------------------------------------------------------------
# Prompt di esempio
# ---------------------------------------------------------------------

EXAMPLE_PROMPTS: List[Dict[str, str]] = [
    {
        "title": "1. Multi-tool classico: record + SLA",
        "peculiarity": "Richiede lookup_record_tool e poi compute_sla_tool.",
        "prompt": "Mostrami il record INC-1002 e calcola lo SLA.",
    },
    {
        "title": "2. RAG pura su policy",
        "peculiarity": (
            "Non cita un ticket: dovrebbe usare search_kb_tool e rispondere citando fonti."
        ),
        "prompt": "Quando un ticket P1 deve essere escalato al team on-call?",
    },
    {
        "title": "3. Near breach P2",
        "peculiarity": (
            "Richiede record operativo e calcolo SLA, ma non necessariamente escalation immediata."
        ),
        "prompt": "Controlla INC-1003: è vicino alla violazione SLA? Qual è il prossimo passo?",
    },
    {
        "title": "4. Gestione errore",
        "peculiarity": (
            "Record inesistente: deve usare lookup_record_tool e gestire il fallimento "
            "senza inventare dati."
        ),
        "prompt": "Recupera INC-9999, calcola lo SLA e dimmi chi deve intervenire.",
    },
    {
        "title": "5. Human-in-the-loop",
        "peculiarity": (
            "Caso critico: in modalità graph deve generare pending_action e chiedere "
            "conferma umana."
        ),
        "prompt": (
            "Analizza INC-1002, verifica la policy di escalation, calcola lo SLA "
            "e, se serve, proponi l'escalation formale."
        ),
    },
]


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2, default=str))


def cmd_setup_rag_data(args: argparse.Namespace) -> None:
    setup_rag_data(ingest_now=args.ingest)


def cmd_examples(_: argparse.Namespace) -> None:
    for item in EXAMPLE_PROMPTS:
        print("=" * 100)
        print(item["title"])
        print(f"Peculiarità: {item['peculiarity']}")
        print(f"Prompt: {item['prompt']}")


def cmd_manual(args: argparse.Namespace) -> None:
    answer, traces = run_real_agent(args.query, max_iter=args.max_iter)

    print("\nANSWER")
    print("=" * 100)
    print(answer)

    print("\nTRACE STRUTTURATA")
    print("=" * 100)
    print_json(traces)


def cmd_graph(args: argparse.Namespace) -> None:
    result = run_graph_agent(
        query=args.query,
        thread_id=args.thread_id,
        auto_decision=args.auto_decision,
    )

    print("\nGRAPH RESULT")
    print("=" * 100)

    # HITL interattivo: se il grafo è sospeso per approvazione e non è stata
    # passata --auto-decision dalla CLI, chiedi all'utente in modo interattivo
    # e riprendi il grafo dallo stesso checkpoint (stesso thread_id).
    if isinstance(result, dict) and result.get("__interrupt__") and not args.auto_decision:
        print("\nIl grafo ha richiesto approvazione umana.")
        print_json(result.get("__interrupt__"))
        print()

        while True:
            raw = input("Decisione [approve / reject]: ").strip().lower()
            if raw in {"approve", "a"}:
                decision = "approve"
                break
            if raw in {"reject", "r"}:
                decision = "reject"
                break
            print("Inserisci 'approve' (o 'a') oppure 'reject' (o 'r').")

        # Riprende dal checkpoint sospeso: il grafo cachato ha lo stesso
        # MemorySaver con lo stato salvato, thread_id identifica la sessione.
        graph, Command = get_langgraph_agent()
        config = {"configurable": {"thread_id": args.thread_id}}
        result = graph.invoke(
            Command(resume={"decision": decision}),
            config=config,
        )
        print(f"\n→ Ripreso con decisione: {decision}")

    messages = result.get("messages", []) if isinstance(result, dict) else []

    if messages:
        final = messages[-1]
        print(extract_text_content(getattr(final, "content", final)))
    else:
        print_json(result)

    print("\nSTATE/TRACES")
    print("=" * 100)
    print_json(result.get("traces", []) if isinstance(result, dict) else result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Esercitazione Giorno 2: agent ITSM con tool calling, "
            "RAG del Giorno 1, loop ReAct e LangGraph HITL."
        )
    )

    subparsers = parser.add_subparsers(required=True)

    setup_parser = subparsers.add_parser("setup-rag-data")
    setup_parser.add_argument("--ingest", action="store_true")
    setup_parser.set_defaults(func=cmd_setup_rag_data)

    examples_parser = subparsers.add_parser("examples")
    examples_parser.set_defaults(func=cmd_examples)

    manual_parser = subparsers.add_parser("manual")
    manual_parser.add_argument("query", type=str)
    manual_parser.add_argument("--max-iter", type=int, default=5)
    manual_parser.set_defaults(func=cmd_manual)

    graph_parser = subparsers.add_parser("graph")
    graph_parser.add_argument("query", type=str)
    graph_parser.add_argument("--thread-id", type=str, default="demo-itsm-001")
    graph_parser.add_argument(
        "--auto-decision",
        choices=["approve", "reject"],
        default=None,
    )
    graph_parser.set_defaults(func=cmd_graph)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()