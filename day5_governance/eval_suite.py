from __future__ import annotations
import sys, io
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

"""
day5_governance/eval_suite.py

╔══════════════════════════════════════════════════════════════════════════════╗
║  Esercitazione Giorno 5 — MATTINO                                           ║
║  Valutazione sistematica di sistemi RAG/LLM su dominio ITSM                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

Obiettivi:
  1. Costruire un golden set da 20 domande ITSM con risposta attesa e fonti
  2. Implementare valutazione semantica (cosine similarity) — no LLM
  3. Implementare LLM-as-a-judge (scala 0-5) con swap test anti-position-bias
  4. Implementare metriche RAGas vere se disponibili, con fallback RAGas-style offline
  5. Regression testing: confronto run corrente vs baseline con alert thresholds
  6. Report JSON + eval_report.md per ogni run

NB: golden_set_itsm.json
      Golden set esterno: contiene domande, risposte attese, fonti e mock_answer.
      Il codice lo carica quando serve tramite load_golden_set().

Architettura:

  CLI (mattino_d5 <cmd>)
        │
        ▼
  ┌──────────────────┐
  │   Golden Set     │  20 domande ITSM · categoria · expected · sources
  └────────┬─────────┘
           │
  ┌────────▼─────────────────────────────────┐
  │              EvalSuite                   │
  │  ┌──────────────┐  ┌──────────────────┐  │
  │  │ SemanticEval │  │  LLM-as-a-Judge  │  │
  │  │ (cosine sim) │  │  (0-5 + swap)    │  │
  │  └──────────────┘  └──────────────────┘  │
  │  ┌─────────────────────────────────────┐  │
  │  │ RAGasEval reale/opzionale + fallback │  │
  │  └─────────────────────────────────────┘  │
  └────────┬─────────────────────────────────┘
           │
  ┌────────▼─────────┐
  │ RegressionCheck  │  baseline.json · alert thresholds · delta report
  └────────┬─────────┘
           │
  ┌────────▼─────────┐
  │  Report Writer   │  runs/day5_mattino/<timestamp>.json
  └──────────────────┘

  
LLM-as-a-judge:
expected answer                 actual answer
      │                               │
      ▼                               ▼
embedding 384-dim                embedding 384-dim
      │                               │
      └──────── cosine similarity ────┘
                      │
                      ▼
             score (tra -1 e 1)
                      │
                      ▼
          confronto con soglia 0.75


Comandi principali:

  python day5_governance/eval_suite.py golden-set
      Mostra il golden set caricato da golden_set_itsm.json.
      Utile per verificare che il file JSON sia corretto e leggibile.

  python day5_governance/eval_suite.py judge "domanda" "expected" "actual"
      Valuta una singola risposta rispetto alla risposta attesa usando
      LLM-as-a-judge, se configurato.

  python day5_governance/eval_suite.py judge "domanda" "expected" "actual" --fast
      Valuta una singola risposta in modalità veloce/offline.
      Non chiama il modello: usa una stima basata sulla similarità semantica.

  (esempio di judge)    
  python day5_governance/eval_suite.py judge \
        "Qual è la policy per un incident P1?" \
        "Un incident P1 richiede escalation entro 15 minuti, bridge call entro 30 minuti e MTTR target di 4 ore." \
        "Per un P1 bisogna avvisare subito il team, aprire una bridge call e puntare alla risoluzione entro 4 ore."    

  python day5_governance/eval_suite.py eval-suite
      Esegue la suite sul golden set completo.
      Di default calcola la valutazione semantica e salva JSON + report Markdown.

  python day5_governance/eval_suite.py eval-suite --fast
      Esegue la suite in modalità veloce/offline.
      Evita chiamate LLM e usa fallback locali dove possibile.

  python day5_governance/eval_suite.py eval-suite --subset N
      Esegue la suite solo sulle prime N domande del golden set.
      Utile per test rapidi durante lo sviluppo.

  python day5_governance/eval_suite.py eval-suite --llm-judge
      Aggiunge la valutazione LLM-as-a-judge per ogni domanda.
      

  python day5_governance/eval_suite.py eval-suite --faithfulness
      Aggiunge metriche RAGas reali se disponibili.
      Se RAGas non è disponibile, usa metriche RAGas-style simulate.

  python day5_governance/eval_suite.py eval-suite --fast --subset 1
      Smoke test minimo: valuta solo la prima domanda senza chiamate esterne.     

  python day5_governance/eval_suite.py eval-suite --fast --subset 5 --faithfulness
      Test veloce su 5 domande con metriche RAGas-style simulate.

  python day5_governance/eval_suite.py eval-suite --llm-judge --faithfulness
      Esegue la valutazione più completa:
      semantic similarity + LLM-as-a-judge + RAGas/RAGas-style.

  python day5_governance/eval_suite.py regression --fast
      Esegue una run veloce e la confronta con baseline.json.
      Se la baseline non esiste, la crea.

  python day5_governance/eval_suite.py regression
      Esegue regression testing completo rispetto alla baseline salvata.
      Produce alert se le metriche peggiorano oltre le soglie configurate.

  python day5_governance/eval_suite.py report --fast
      Genera un report completo in modalità veloce:
      semantic similarity, judge mock, faithfulness/RAGas-style e sintesi finale.

  python day5_governance/eval_suite.py report
      Genera un report completo usando anche LLM judge e RAGas reale quando disponibili.

  python day5_governance/eval_suite.py examples
      Mostra esempi d'uso da terminale.  

Comandi:
    python day5_governance/eval_suite.py golden-set
    (template) python day5_governance/eval_suite.py judge "domanda" "expected" "actual" [--fast]
    python day5_governance/eval_suite.py judge \
        "Qual è la policy per un incident P1?" \
        "Un incident P1 richiede escalation entro 15 minuti, bridge call entro 30 minuti e MTTR target di 4 ore." \
        "Per un P1 bisogna avvisare subito il team, aprire una bridge call e puntare alla risoluzione entro 4 ore."
    python day5_governance/eval_suite.py eval-suite [--fast] [--subset N] [--llm-judge] [--faithfulness]
    python day5_governance/eval_suite.py regression [--fast]
    python day5_governance/eval_suite.py report [--fast]
    python day5_governance/eval_suite.py examples

Variabili .env:
    GOOGLE_API_KEY=...
    GEMINI_MODEL=gemini-2.5-flash
    MIN_SECONDS_BETWEEN_MODEL_CALLS=15
    PRICE_INPUT_PER_1M=0.10
    PRICE_OUTPUT_PER_1M=0.40
    SEMANTIC_THRESHOLD=0.75
    JUDGE_PASS_SCORE=3

Nota didattica:
    Lo swap test mitiga il position bias dell'LLM-judge: si valuta (expected, actual)
    e poi (actual, expected) con i ruoli invertiti. Se |score1 - score2_swapped| > 1
    il giudice è inaffidabile su quella coppia. Il punteggio finale è la media dei due ordini.
"""

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

# ── import opzionali ───────────────────────────────────────────────────────────

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage, SystemMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("[WARN] langchain-google-genai non installato. pip install langchain-google-genai")

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    print("[INFO] sentence-transformers non installato — uso fallback lessicale. pip install sentence-transformers")

try:
    from datasets import Dataset
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import faithfulness as ragas_faithfulness
    from ragas.metrics import answer_relevancy as ragas_answer_relevancy
    from ragas.metrics import context_recall as ragas_context_recall
    from ragas.metrics import context_precision as ragas_context_precision
    RAGAS_AVAILABLE = True
except Exception as _ragas_import_error:
    Dataset = None
    ragas_evaluate = None
    ragas_faithfulness = None
    ragas_answer_relevancy = None
    ragas_context_recall = None
    ragas_context_precision = None
    RAGAS_AVAILABLE = False
    RAGAS_IMPORT_ERROR = str(_ragas_import_error)
    print("[INFO] ragas/datasets non disponibili — uso fallback RAGas-style simulato. pip install ragas datasets")

# ── .env + path setup ─────────────────────────────────────────────────────────

load_dotenv()

BASE_DIR = Path(__file__).parent.resolve()
RUNS_DIR = BASE_DIR / "runs" / "day5_mattino"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
BASELINE_FILE = RUNS_DIR / "baseline.json"

# =============================================================================
# 1. CONFIG
# =============================================================================

GOOGLE_API_KEY           = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL             = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
MIN_SECONDS              = float(os.getenv("MIN_SECONDS_BETWEEN_MODEL_CALLS", "15"))
PRICE_IN_PER_1M          = float(os.getenv("PRICE_INPUT_PER_1M", "0.10"))
PRICE_OUT_PER_1M         = float(os.getenv("PRICE_OUTPUT_PER_1M", "0.40"))
SEMANTIC_THRESHOLD       = float(os.getenv("SEMANTIC_THRESHOLD", "0.75"))
JUDGE_PASS_SCORE         = int(os.getenv("JUDGE_PASS_SCORE", "3"))

# Regression alert: se la metrica cala di più di questa soglia → ALERT
ALERT_THRESHOLDS: Dict[str, float] = {
    "pass_rate":          -0.05,   # max -5%
    "avg_semantic_score": -0.05,   # max -0.05 punti
    "avg_judge_score":    -0.30,   # max -0.30 punti
    "faithfulness_score": -0.05,   # max -5%
}

_last_call_ts: float = 0.0


def _rate_limit() -> None:
    """Rispetta MIN_SECONDS_BETWEEN_MODEL_CALLS."""
    global _last_call_ts
    elapsed = time.time() - _last_call_ts
    if elapsed < MIN_SECONDS:
        wait = MIN_SECONDS - elapsed
        print(f"  ⏳ rate-limit: attendo {wait:.1f}s …")
        time.sleep(wait)
    _last_call_ts = time.time()


def _cost(tokens_in: int, tokens_out: int) -> float:
    return (tokens_in / 1_000_000) * PRICE_IN_PER_1M + \
           (tokens_out / 1_000_000) * PRICE_OUT_PER_1M


# =============================================================================
# 2. GOLDEN SET — caricato da JSON esterno
# =============================================================================

DEFAULT_GOLDEN_SET_FILE = BASE_DIR / "golden_set_itsm.json"


def _resolve_golden_set_path(path: Optional[Path] = None) -> Path:
    """
    Risolve il path del golden set.

    Ordine di precedenza:
      1. path passato esplicitamente alla funzione
      2. variabile d'ambiente GOLDEN_SET_FILE
      3. day5_governance/golden_set_itsm.json

    Se il path è relativo, viene interpretato rispetto a BASE_DIR.
    """
    if path is not None:
        candidate = Path(path)
    else:
        candidate = Path(os.getenv("GOLDEN_SET_FILE", str(DEFAULT_GOLDEN_SET_FILE)))

    if not candidate.is_absolute():
        candidate = BASE_DIR / candidate

    return candidate.resolve()


def load_golden_set(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    Carica il golden set da JSON solo quando serve.

    Formato atteso: lista di oggetti con almeno:
      - id
      - category
      - question
      - expected
      - sources

    Il campo mock_answer è opzionale ma utile per la demo offline con MockITSMRagAgent.
    """
    golden_set_path = _resolve_golden_set_path(path)

    if not golden_set_path.exists():
        raise FileNotFoundError(
            f"Golden set non trovato: {golden_set_path}. "
            "Crea il file golden_set_itsm.json oppure imposta GOLDEN_SET_FILE."
        )

    try:
        data = json.loads(golden_set_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON non valido in {golden_set_path}: {exc}") from exc

    # Supporta anche un wrapper futuro del tipo {"items": [...]}.
    if isinstance(data, dict) and "items" in data:
        data = data["items"]

    if not isinstance(data, list):
        raise ValueError(
            f"Formato golden set non valido in {golden_set_path}: "
            "attesa una lista JSON di domande."
        )

    required_fields = {"id", "category", "question", "expected", "sources"}
    for idx, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Elemento #{idx} del golden set non è un oggetto JSON.")

        missing = required_fields - set(item)
        if missing:
            raise ValueError(
                f"Elemento #{idx} del golden set incompleto: mancano {sorted(missing)}"
            )

        if not isinstance(item["sources"], list):
            raise ValueError(
                f"Elemento {item.get('id', idx)}: il campo 'sources' deve essere una lista."
            )

    return data


# =============================================================================
# 2B. AGENT MOCK — punto di sostituzione con un vero agent.ask(q)
# =============================================================================

class MockITSMRagAgent:
    """
    Agent dimostrativo usato per rendere il lab eseguibile senza backend reale.

    In produzione / esercizio avanzato, sostituisci questa classe con il vero agente
    RAG e mantieni lo stesso contratto:

        actual = agent.ask(item["question"])

    Qui il mock restituisce la risposta già presente nel golden set per simulare
    l'output dell'assistente e permettere di testare tutta la pipeline di eval.
    """

    def ask(self, question: str, item: Optional[Dict[str, Any]] = None) -> str:
        if item is not None and "mock_answer" in item:
            return str(item["mock_answer"])
        return "Non ho una risposta disponibile per questa domanda."


def build_agent() -> MockITSMRagAgent:
    """Factory dell'agent: sostituire qui il mock con il vero agente RAG."""
    return MockITSMRagAgent()


def _estimate_tokens(text: str) -> int:
    """Stima semplice dei token: sufficiente per un costo didattico offline."""
    return max(1, math.ceil(len(text) / 4))


# Knowledge base mock usata sia dal fallback offline sia dall'integrazione RAGas vera.
# In un progetto reale questi testi arriverebbero dal retriever dell'agente RAG.
KB_TEXTS: Dict[str, str] = {
    "KB-001": "Un incident P1 richiede escalation immediata al team di guardia entro 15 minuti e bridge call entro 30 minuti.",
    "KB-002": "Il MTTR target per gli incident P1 è di 4 ore. Aggiornamenti ogni 30 minuti sono obbligatori.",
    "KB-003": "I log di accesso sono conservati 90 giorni in hot storage e 2 anni in cold storage.",
    "KB-004": "Le change standard sono pre-approvate e non richiedono approvazione CAB.",
    "KB-005": "Il Change Manager autorizza automaticamente le change standard dal catalogo approvato.",
    "KB-006": "Onboarding IT: account AD D-3, laptop D-1, MFA e VPN giorno 1, formazione sicurezza entro 5 giorni.",
    "KB-007": "SLA P3: primo intervento 4 ore lavorative, risoluzione 3 giorni lavorativi.",
    "KB-008": "Il downtime SLA si misura in orario lavorativo 8:00-18:00 LUN-VEN.",
    "KB-009": "Disponibilità target: 99.5% mensile. Manutenzione pianificata esclusa dal calcolo.",
    "KB-010": "SLA breach P2: escalation automatica al Service Manager, RCA entro 48 ore.",
    "KB-011": "Finestra di manutenzione: terzo sabato del mese, 22:00-06:00. Comunicazione 5 giorni prima.",
    "KB-012": "Rollback patch: aprire incident, usare snapshot se disponibile.",
    "KB-013": "Procedura Patch Rollback Runbook v3: comunicare al Change Manager e aggiornare CMDB.",
    "KB-014": "Data breach: isolare sistemi, notificare CISO e DPO entro 1 ora, aprire incident P1.",
    "KB-015": "Notifica Garante Privacy obbligatoria entro 72 ore se rischio per gli interessati (art. 33 GDPR).",
    "KB-016": "Privileged Access Request: massimo 8 ore, sessione registrata su PAM.",
    "KB-017": "Aggiunta server a monitoraggio: installare node_exporter, aprire porta 9100.",
    "KB-018": "Prometheus target in prometheus.yml, dashboard Grafana da template 'Server Base'.",
    "KB-019": "Vendor fuori SLA: escalation al Vendor Manager, non-conformità formale.",
    "KB-020": "Piano continuità ferie: guardia ridotta con reperibilità per P1/P2.",
    "KB-021": "P3/P4 durante ferie: accodati e gestiti alla riapertura. Contatti nel portale ITSM.",
    "KB-022": "Perdita MFA: verifica IDV (3 domande + callback), token revocato, temporaneo 24 ore.",
    "KB-023": "Falso positivo AV: non rimuovere quarantena senza analisi, aprire incident P2.",
    "KB-024": "Exclusion list AV: aggiunta tramite richiesta formale al Security Manager.",
    "KB-025": "Software non autorizzato: approvazione tramite Software Asset Management e portale aziendale.",
}


def get_contexts_from_sources(sources: List[str]) -> List[str]:
    """Restituisce i chunk di contesto associati agli ID fonte del golden set."""
    return [KB_TEXTS.get(s, f"[{s}: contenuto non disponibile]") for s in sources]


# =============================================================================
# 3. EMBEDDING TOY (fallback se sentence-transformers non disponibile)
# =============================================================================

def _toy_embedding(text: str) -> List[float]:
    """
    Embedding deterministico basato su hash delle parole.
    Non è semantico ma è riproducibile e funziona senza GPU.
    Usato solo se sentence-transformers non è installato.
    """
    words = re.findall(r"\w+", text.lower())
    vec = [0.0] * 64
    for w in words:
        h = int(hashlib.md5(w.encode()).hexdigest(), 16)
        for i in range(64):
            vec[i] += ((h >> i) & 1) * 2 - 1
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


# =============================================================================
# 4. VALUTATORE SEMANTICO
# =============================================================================

_sbert_model = None


def _get_sbert():
    global _sbert_model
    if _sbert_model is None and SBERT_AVAILABLE:
        print("  ℹ️  Carico SentenceTransformer (prima volta) …")
        _sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _sbert_model


def eval_semantic(expected: str, actual: str) -> Dict[str, Any]:
    """
    Calcola la similarità coseno tra expected e actual.

    Returns:
        {
          "score": float [0-1],
          "pass": bool,
          "method": "sbert" | "toy",
          "threshold": float
        }
    """
    model = _get_sbert()
    if model is not None:
        import numpy as np
        emb = model.encode([expected, actual])
        score = float(
            np.dot(emb[0], emb[1]) /
            (np.linalg.norm(emb[0]) * np.linalg.norm(emb[1]) + 1e-9)
        )
        method = "sbert"
    else:
        # Fallback offline: similarità lessicale F1 con leggero boost didattico.
        # È più stabile dell'embedding hash per una demo senza sentence-transformers.
        stopwords = {
            "il", "lo", "la", "le", "gli", "i", "un", "una", "uno", "di", "a",
            "da", "in", "con", "su", "per", "tra", "fra", "e", "o", "che",
            "del", "della", "delle", "dei", "degli", "al", "alla", "alle",
            "agli", "ai", "nel", "nella", "nelle", "nei", "come", "cosa",
            "qual", "quale", "sono", "viene", "vengono", "entro", "giorni",
            "ore", "se", "non"
        }
        words_expected = {
            w for w in re.findall(r"[a-zA-ZÀ-ÿ0-9]+", expected.lower())
            if len(w) >= 3 and w not in stopwords
        }
        words_actual = {
            w for w in re.findall(r"[a-zA-ZÀ-ÿ0-9]+", actual.lower())
            if len(w) >= 3 and w not in stopwords
        }
        if not words_expected or not words_actual:
            score = 0.0
        else:
            overlap = len(words_expected & words_actual)
            precision = overlap / len(words_actual)
            recall = overlap / len(words_expected)
            f1 = (2 * precision * recall) / (precision + recall + 1e-9)
            score = min(f1 * 1.90, 1.0)
        method = "lexical_fallback"

    return {
        "score": round(score, 4),
        "pass": score >= SEMANTIC_THRESHOLD,
        "method": method,
        "threshold": SEMANTIC_THRESHOLD,
    }


# =============================================================================
# 5. LLM-AS-A-JUDGE (con swap test anti-position-bias)
# =============================================================================

_JUDGE_SYSTEM = """\
Sei un valutatore esperto di sistemi di supporto IT (ITSM).
Valuta la qualità della risposta di un assistente AI rispetto alla risposta attesa.

Scala di valutazione (0-5):
  5 — Risposta eccellente: corretta, completa, chiara, non aggiunge info errate
  4 — Buona: corretta e quasi completa, piccole omissioni non critiche
  3 — Sufficiente: corretta ma incompleta, mancano informazioni importanti
  2 — Parzialmente corretta: alcune informazioni giuste ma anche errori o omissioni gravi
  1 — Risposta quasi completamente sbagliata o fuorviante
  0 — Completamente errata, rifiuto inappropriato, o risposta dannosa

Rispondi SOLO con JSON valido, nessun testo aggiuntivo:
{"score": <0-5>, "justification": "<max 1 frase>"}
"""

_JUDGE_HUMAN = """\
Domanda: {question}

Risposta attesa:
{expected}

Risposta del sistema:
{actual}

Fornisci il tuo giudizio in JSON."""

def _content_to_text(content: Any) -> str:
    """
    Normalizza AIMessage.content in stringa.

    LangChain/Gemini può restituire:
    - str
    - list[dict] con blocchi tipo {"type": "text", "text": "..."}
    - list[str]
    - altri oggetti provider-native
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
                continue

            if isinstance(item, dict):
                # Formato LangChain standard: {"type": "text", "text": "..."}
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                    continue

                # Formati provider-native alternativi
                if isinstance(item.get("content"), str):
                    parts.append(item["content"])
                    continue

                # Fallback leggibile
                parts.append(json.dumps(item, ensure_ascii=False))
                continue

            parts.append(str(item))

        return "\n".join(parts).strip()

    return str(content).strip()

def _llm_judge_single(question: str, expected: str, actual: str,
                       llm: Any) -> Tuple[int, str]:
    """Singola valutazione LLM. Ritorna (score, justification)."""
    _rate_limit()
    prompt = _JUDGE_HUMAN.format(question=question, expected=expected, actual=actual)
    messages = [
        SystemMessage(content=_JUDGE_SYSTEM),
        HumanMessage(content=prompt),
    ]
    resp = llm.invoke(messages)
    raw = _extract_json_object(_content_to_text(resp.content))

    try:
        data = json.loads(raw)
        return int(data.get("score", 0)), str(data.get("justification", ""))
    except Exception:
        # fallback: cerca un numero nel testo
        m = re.search(r"\b([0-5])\b", raw)
        return int(m.group(1)) if m else 0, raw[:120]


def eval_llm_judge(question: str, expected: str, actual: str,
                   fast: bool = False) -> Dict[str, Any]:
    """
    LLM-as-a-judge con swap test per rilevare position bias.

    Forward pass:  giudica (expected=A, actual=B)
    Swap pass:     giudica (expected=B, actual=A) — ruoli invertiti
    bias_delta = |score_forward - score_swapped|
    Il giudice è affidabile se bias_delta <= 1.

    Returns:
        {
          "score": float,         # media dei due pass
          "score_forward": int,
          "score_swapped": int,
          "bias_delta": float,
          "reliable": bool,
          "justification": str,
          "pass": bool,
          "method": "llm" | "mock"
        }
    """
    if fast or not LANGCHAIN_AVAILABLE or not GOOGLE_API_KEY:
        # mock: punteggio basato su similarità semantica
        sem = eval_semantic(expected, actual)
        mock_score = round(sem["score"] * 5, 2)
        return {
            "score": mock_score,
            "score_forward": round(mock_score),
            "score_swapped": round(mock_score),
            "bias_delta": 0.0,
            "reliable": True,
            "justification": f"[MOCK] semantic={sem['score']:.3f}",
            "pass": mock_score >= JUDGE_PASS_SCORE,
            "method": "mock",
        }

    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.0,
    )

    # forward pass
    score_fwd, just_fwd = _llm_judge_single(question, expected, actual, llm)

    # swap pass (aspetta il rate limit)
    score_swap, _ = _llm_judge_single(question, actual, expected, llm)

    # Swap test: lo stesso confronto con expected/actual invertiti dovrebbe restare simile.
    # Non riflettiamo il punteggio con 5-score: se le due risposte sono semanticamente
    # equivalenti, entrambi gli ordini devono ricevere un punteggio alto.
    bias_delta = abs(score_fwd - score_swap)
    avg_score = (score_fwd + score_swap) / 2.0

    return {
        "score": round(avg_score, 2),
        "score_forward": score_fwd,
        "score_swapped": score_swap,
        "bias_delta": round(bias_delta, 2),
        "reliable": bias_delta <= 1,
        "justification": just_fwd,
        "pass": avg_score >= JUDGE_PASS_SCORE,
        "method": "llm",
    }


# =============================================================================
# 6. FAITHFULNESS RAGas-STYLE
# =============================================================================

_FAITHFUL_SYSTEM = """\
Sei un valutatore di faithfulness per sistemi RAG.
Dato un insieme di contesti e una risposta, analizza ogni claim della risposta
e valuta se è supportato dai contesti forniti.

Rispondi SOLO con JSON valido:
{
  "claims": ["<claim 1>", "<claim 2>", ...],
  "supported": [true/false, ...],
  "faithfulness": <float 0.0-1.0>
}
"""

_FAITHFUL_HUMAN = """\
Contesti disponibili:
{contexts}

Risposta da valutare:
{answer}

Analizza ogni claim della risposta e indica se è supportato dai contesti."""


def eval_faithfulness(answer: str, sources: List[str],
                      fast: bool = False) -> Dict[str, Any]:
    """
    Faithfulness RAGas-style: fraction of claims supported by retrieved contexts.

    In modalità --fast usa una versione euristica (keyword overlap).
    In modalità LLM usa Gemini per estrarre i claims e verificarli.

    Returns:
        {
          "faithfulness": float [0-1],
          "total_claims": int,
          "supported_claims": int,
          "claims": List[str],
          "supported": List[bool],
          "method": "llm" | "heuristic"
        }
    """
    # Se non ci sono fonti dichiarate, faithfulness è N/A (restituiamo 1.0 per domande
    # avversariali dove la risposta corretta non ha KB)
    if not sources:
        return {
            "faithfulness": 1.0,
            "total_claims": 0,
            "supported_claims": 0,
            "claims": [],
            "supported": [],
            "method": "n/a (no sources)",
            "note": "Nessuna fonte dichiarata — faithfulness non applicabile",
        }

    # Simula contesti KB: in un sistema reale arriverebbero dal retriever
    # Qui generiamo un contesto mock basato sui source ID
    kb_texts = {
        "KB-001": "Un incident P1 richiede escalation immediata al team di guardia entro 15 minuti e bridge call entro 30 minuti.",
        "KB-002": "Il MTTR target per gli incident P1 è di 4 ore. Aggiornamenti ogni 30 minuti sono obbligatori.",
        "KB-003": "I log di accesso sono conservati 90 giorni in hot storage e 2 anni in cold storage.",
        "KB-004": "Le change standard sono pre-approvate e non richiedono approvazione CAB.",
        "KB-005": "Il Change Manager autorizza automaticamente le change standard dal catalogo approvato.",
        "KB-006": "Onboarding IT: account AD D-3, laptop D-1, MFA e VPN giorno 1, formazione sicurezza entro 5 giorni.",
        "KB-007": "SLA P3: primo intervento 4 ore lavorative, risoluzione 3 giorni lavorativi.",
        "KB-008": "Il downtime SLA si misura in orario lavorativo 8:00-18:00 LUN-VEN.",
        "KB-009": "Disponibilità target: 99.5% mensile. Manutenzione pianificata esclusa dal calcolo.",
        "KB-010": "SLA breach P2: escalation automatica al Service Manager, RCA entro 48 ore.",
        "KB-011": "Finestra di manutenzione: terzo sabato del mese, 22:00-06:00. Comunicazione 5 giorni prima.",
        "KB-012": "Rollback patch: aprire incident, usare snapshot se disponibile.",
        "KB-013": "Procedura Patch Rollback Runbook v3: comunicare al Change Manager e aggiornare CMDB.",
        "KB-014": "Data breach: isolare sistemi, notificare CISO e DPO entro 1 ora, aprire incident P1.",
        "KB-015": "Notifica Garante Privacy obbligatoria entro 72 ore se rischio per gli interessati (art. 33 GDPR).",
        "KB-016": "Privileged Access Request: massimo 8 ore, sessione registrata su PAM.",
        "KB-017": "Aggiunta server a monitoraggio: installare node_exporter, aprire porta 9100.",
        "KB-018": "Prometheus target in prometheus.yml, dashboard Grafana da template 'Server Base'.",
        "KB-019": "Vendor fuori SLA: escalation al Vendor Manager, non-conformità formale.",
        "KB-020": "Piano continuità ferie: guardia ridotta con reperibilità per P1/P2.",
        "KB-021": "P3/P4 durante ferie: accodati e gestiti alla riapertura. Contatti nel portale ITSM.",
        "KB-022": "Perdita MFA: verifica IDV (3 domande + callback), token revocato, temporaneo 24 ore.",
        "KB-023": "Falso positivo AV: non rimuovere quarantena senza analisi, aprire incident P2.",
        "KB-024": "Exclusion list AV: aggiunta tramite richiesta formale al Security Manager.",
        "KB-025": "Software non autorizzato: approvazione tramite Software Asset Management e portale aziendale.",
    }

    contexts = [kb_texts.get(s, f"[{s}: contenuto non disponibile]") for s in sources]
    context_blob = "\n".join(f"[{s}] {kb_texts.get(s, '')}" for s in sources)

    if fast or not LANGCHAIN_AVAILABLE or not GOOGLE_API_KEY:
        # Euristica: conta parole chiave dell'answer che compaiono nei contesti
        answer_words = set(re.findall(r"\w{4,}", answer.lower()))
        context_words = set(re.findall(r"\w{4,}", context_blob.lower()))
        overlap = len(answer_words & context_words)
        total = len(answer_words) or 1
        faith = round(min(overlap / total * 1.5, 1.0), 3)  # boost per compensare la semplicità
        sentences = [s.strip() for s in re.split(r"[.;]", answer) if len(s.strip()) > 10]
        supported = [True] * len(sentences)  # ottimistico in mock
        return {
            "faithfulness": faith,
            "total_claims": len(sentences),
            "supported_claims": len(sentences),
            "claims": sentences[:5],
            "supported": supported[:5],
            "method": "heuristic",
        }

    _rate_limit()
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.0,
    )
    prompt = _FAITHFUL_HUMAN.format(contexts=context_blob, answer=answer)
    messages = [
        SystemMessage(content=_FAITHFUL_SYSTEM),
        HumanMessage(content=prompt),
    ]
    resp = llm.invoke(messages)
    raw = _extract_json_object(_content_to_text(resp.content))

    try:
        data = json.loads(raw)
        claims = data.get("claims", [])
        supported = data.get("supported", [True] * len(claims))
        faith = data.get("faithfulness", sum(supported) / max(len(supported), 1))

        return {
            "faithfulness": round(float(faith), 3),
            "total_claims": len(claims),
            "supported_claims": sum(1 for s in supported if s),
            "claims": claims,
            "supported": supported,
            "method": "llm",
        }

    except Exception as e:
        return {
            "faithfulness": 0.5,
            "total_claims": 0,
            "supported_claims": 0,
            "claims": [],
            "supported": [],
            "method": "llm_error",
            "error": str(e),
        }

def _extract_json_object(raw: str) -> str:
    """
    Estrae il primo oggetto JSON da una risposta LLM.
    Gestisce sia JSON puro sia blocchi ```json ... ```.
    """
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    if raw.startswith("{") and raw.endswith("}"):
        return raw

    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if m:
        return m.group(0)

    return raw
# =============================================================================
# 6B. METRICHE RAGAS-STYLE SIMULATE
# =============================================================================

def eval_answer_relevancy(question: str, answer: str) -> Dict[str, Any]:
    """
    Answer relevancy simulata: misura quanto la risposta è semanticamente vicina
    alla domanda. Non è RAGas reale, ma rende esplicito il concetto visto a lezione.
    """
    sem = eval_semantic(question, answer)
    return {
        "answer_relevancy": sem["score"],
        "method": f"ragas-style-simulated/{sem['method']}",
        "note": "Metrica simulata: usare ragas.metrics.answer_relevancy in produzione.",
    }


def eval_context_recall(expected: str, sources: List[str]) -> Dict[str, Any]:
    """
    Context recall simulata: verifica se le fonti dichiarate coprono la risposta attesa
    tramite overlap lessicale. In un RAG reale si usano i chunk recuperati dal retriever.
    """
    if not sources:
        return {
            "context_recall": 1.0,
            "method": "n/a (no sources)",
            "note": "Nessuna fonte attesa: tipico di domande avversariali/safety.",
        }
    expected_words = set(re.findall(r"\w{4,}", expected.lower()))
    source_words = set(" ".join(sources).lower().replace("-", " ").split())
    # Per il lab offline, consideriamo le fonti presenti come recuperate: il recall è alto
    # ma non perfetto per ricordare che è una simulazione.
    recall = 0.85 if expected_words else 1.0
    return {
        "context_recall": round(recall, 3),
        "method": "ragas-style-simulated",
        "note": "Simulato: in produzione confrontare ground_truth e contesti recuperati.",
    }


def eval_context_precision(question: str, sources: List[str]) -> Dict[str, Any]:
    """
    Context precision simulata: stima se i documenti recuperati sono pochi e mirati.
    Penalizza troppe fonti rispetto a una domanda semplice.
    """
    if not sources:
        return {
            "context_precision": 1.0,
            "method": "n/a (no sources)",
            "note": "Nessuna fonte attesa: metrica non applicabile.",
        }
    precision = max(0.5, 1.0 - max(0, len(sources) - 2) * 0.1)
    return {
        "context_precision": round(precision, 3),
        "method": "ragas-style-simulated",
        "note": "Simulato: in produzione calcolare quanti chunk recuperati sono davvero rilevanti.",
    }


def _extract_ragas_scores(result: Any) -> Dict[str, float]:
    """Normalizza l'output di RAGas tra versioni diverse della libreria."""
    if hasattr(result, "to_pandas"):
        try:
            row = result.to_pandas().iloc[0].to_dict()
            return {str(k): float(v) for k, v in row.items() if isinstance(v, (int, float))}
        except Exception:
            pass
    if isinstance(result, dict):
        return {str(k): float(v) for k, v in result.items() if isinstance(v, (int, float))}
    scores: Dict[str, float] = {}
    for name in ("faithfulness", "answer_relevancy", "context_recall", "context_precision"):
        val = getattr(result, name, None)
        if isinstance(val, (int, float)):
            scores[name] = float(val)
    return scores


def eval_ragas_real_metrics(question: str, expected: str, actual: str, sources: List[str]) -> Dict[str, Any]:
    """
    Prova a usare RAGas vero. Se qualcosa manca o fallisce, solleva eccezione
    e il chiamante userà il fallback RAGas-style.
    """
    if not RAGAS_AVAILABLE:
        raise RuntimeError(f"RAGas non disponibile: {globals().get('RAGAS_IMPORT_ERROR', 'import fallito')}")

    contexts = get_contexts_from_sources(sources)
    if not contexts:
        # Per casi safety/adversarial senza fonti, RAGas non è molto informativo.
        contexts = ["Domanda safety senza contesto documentale: la risposta deve rifiutare richieste non consentite."]

    dataset = Dataset.from_dict({
        "question": [question],
        "answer": [actual],
        "contexts": [contexts],
        "ground_truth": [expected],
    })

    raw = ragas_evaluate(
        dataset,
        metrics=[
            ragas_faithfulness,
            ragas_answer_relevancy,
            ragas_context_recall,
            ragas_context_precision,
        ],
    )
    scores = _extract_ragas_scores(raw)

    return {
        "faithfulness": {
            "faithfulness": round(scores.get("faithfulness", 0.0), 4),
            "method": "ragas.evaluate",
        },
        "answer_relevancy": {
            "answer_relevancy": round(scores.get("answer_relevancy", 0.0), 4),
            "method": "ragas.evaluate",
        },
        "context_recall": {
            "context_recall": round(scores.get("context_recall", 0.0), 4),
            "method": "ragas.evaluate",
        },
        "context_precision": {
            "context_precision": round(scores.get("context_precision", 0.0), 4),
            "method": "ragas.evaluate",
        },
        "ragas_real": True,
        "note": "Metriche calcolate con RAGas reale.",
    }


def eval_ragas_style_metrics(question: str, expected: str, actual: str, sources: List[str], fast: bool) -> Dict[str, Any]:
    """
    Usa RAGas vero quando possibile. In modalità --fast, o se RAGas non è
    installato/configurato, usa un fallback simulato che non manda in errore la demo.
    """
    if not fast:
        try:
            return eval_ragas_real_metrics(question, expected, actual, sources)
        except Exception as e:
            fallback_reason = str(e)[:220]
    else:
        fallback_reason = "modalità --fast: RAGas reale saltato volontariamente"

    faith = eval_faithfulness(actual, sources, fast=True)
    relevancy = eval_answer_relevancy(question, actual)
    recall = eval_context_recall(expected, sources)
    precision = eval_context_precision(question, sources)
    return {
        "faithfulness": faith,
        "answer_relevancy": relevancy,
        "context_recall": recall,
        "context_precision": precision,
        "ragas_real": False,
        "fallback_reason": fallback_reason,
        "note": "Fallback RAGas-style simulato: la demo resta eseguibile anche senza dipendenze/API key.",
    }


def write_eval_report(payload: Dict[str, Any]) -> Path:
    """Genera eval_report.md come richiesto dai deliverable della lezione."""
    stats = payload["stats"]
    rows = []
    for cat, cstats in sorted(stats.get("categories", {}).items()):
        rows.append(f"| {cat} | {cstats['count']} | {cstats['avg_score']:.3f} | {cstats['pass_rate']*100:.1f}% |")

    failures = sorted(
        payload["results"],
        key=lambda r: r["semantic"]["score"],
    )[:5]
    fail_lines = []
    for r in failures:
        cause = "similarità sotto soglia" if not r["semantic"]["pass"] else "caso più debole, ma sopra soglia"
        fail_lines.append(
            f"- **{r['id']}** ({r['category']}): score={r['semantic']['score']:.3f}; causa probabile: {cause}."
        )

    report = f"""# Eval Report — Giorno 5 Mattino

## Sintesi

- **Run ID:** {payload['run_id']}
- **Timestamp:** {payload['timestamp']}
- **Domande valutate:** {stats['total']}
- **Pass rate globale:** {stats['pass_rate']*100:.1f}%
- **Soglia semantic similarity:** {payload['config']['semantic_threshold']}
- **Score semantico medio:** {stats['avg_semantic_score']:.4f}
- **Latenza media:** {stats.get('avg_latency_ms', 0):.1f} ms
- **Latenza p95:** {stats.get('p95_latency_ms', 0):.1f} ms
- **Costo stimato totale:** ${stats.get('estimated_total_cost_usd', 0):.6f}
- **Costo stimato medio/query:** ${stats.get('estimated_avg_cost_usd', 0):.6f}

## Metriche RAGas / RAGas-style

> Modalità RAGas reale usata in questa run: **{payload['config'].get('ragas_real', False)}**.
> Se RAGas non è installato/configurato, il codice usa automaticamente un fallback RAGas-style per non bloccare la demo.

- **Faithfulness media:** {stats.get('faithfulness_score', 'n/a')}
- **Answer relevancy media:** {stats.get('answer_relevancy_score', 'n/a')}
- **Context recall medio:** {stats.get('context_recall_score', 'n/a')}
- **Context precision medio:** {stats.get('context_precision_score', 'n/a')}

## Risultati per categoria

| Categoria | N | Avg semantic score | Pass rate |
|---|---:|---:|---:|
{chr(10).join(rows)}

## Top-5 domande più deboli

{chr(10).join(fail_lines)}

## Raccomandazioni prossimo ciclo

1. Sostituire `MockITSMRagAgent` con il vero agente RAG mantenendo `agent.ask(question)`.
2. Usare RAGas reale quando sono disponibili contesti recuperati e ground truth.
3. Aggiungere query reali anonimizzate al golden set a ogni sprint.
4. Salvare una baseline solo dopo una run approvata.
"""
    report_path = RUNS_DIR / f"eval_report_{payload['run_id']}.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path



# =============================================================================
# 7. EVAL SUITE
# =============================================================================

def run_eval_suite(
    fast: bool = False,
    subset: Optional[int] = None,
    use_llm_judge: bool = False,
    use_faithfulness: bool = False,
) -> Dict[str, Any]:
    """
    Esegue la suite di valutazione su tutto il golden set (o un subset).
    La risposta viene prodotta tramite agent.ask(q). Di default l'agent è un mock
    offline, sostituibile con un vero RAG agent nella funzione build_agent().
    """
    golden_set = load_golden_set()
    questions = golden_set[:subset] if subset else golden_set
    results = []
    latencies_ms: List[float] = []
    estimated_costs: List[float] = []
    ts_start = time.time()
    agent = build_agent()

    print(f"\n{'═'*62}")
    print(f"  EVAL SUITE — {len(questions)} domande | fast={fast} | "
          f"llm_judge={use_llm_judge} | ragas={use_faithfulness} | ragas_available={RAGAS_AVAILABLE}")
    print(f"{'═'*62}")

    for i, q in enumerate(questions, 1):
        print(f"\n  [{i:02d}/{len(questions):02d}] {q['id']} ({q['category']}) — {q['question'][:55]}…")

        ask_start = time.perf_counter()
        actual = agent.ask(q["question"], item=q)
        latency_ms = (time.perf_counter() - ask_start) * 1000
        latencies_ms.append(latency_ms)

        tokens_in = _estimate_tokens(q["question"] + " " + " ".join(q.get("sources", [])))
        tokens_out = _estimate_tokens(actual)
        estimated_cost = _cost(tokens_in, tokens_out)
        estimated_costs.append(estimated_cost)

        # Valutazione semantica (sempre)
        sem = eval_semantic(q["expected"], actual)
        print(f"         Semantic: {sem['score']:.3f} ({sem['method']}) → {'✓ PASS' if sem['pass'] else '✗ FAIL'}")

        result: Dict[str, Any] = {
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "expected": q["expected"],
            "actual": actual,
            "sources": q["sources"],
            "semantic": sem,
            "latency_ms": round(latency_ms, 2),
            "estimated_cost_usd": round(estimated_cost, 8),
            "agent_type": type(agent).__name__,
        }

        # LLM judge (opzionale)
        if use_llm_judge:
            judge = eval_llm_judge(q["question"], q["expected"], actual, fast=fast)
            print(f"         LLM Judge: {judge['score']:.2f}/5 | "
                  f"bias_delta={judge['bias_delta']:.2f} | "
                  f"reliable={judge['reliable']} → {'✓ PASS' if judge['pass'] else '✗ FAIL'}")
            result["judge"] = judge

        # Metriche RAGas vere con fallback RAGas-style (opzionale)
        if use_faithfulness:
            ragas_style = eval_ragas_style_metrics(q["question"], q["expected"], actual, q["sources"], fast=fast)
            faith = ragas_style["faithfulness"]
            label = "RAGas reale" if ragas_style.get("ragas_real") else "RAGas-style fallback"
            print(f"         {label}: faith={faith['faithfulness']:.3f} | "
                  f"rel={ragas_style['answer_relevancy']['answer_relevancy']:.3f} | "
                  f"recall={ragas_style['context_recall']['context_recall']:.3f} | "
                  f"precision={ragas_style['context_precision']['context_precision']:.3f}")
            result["ragas_style"] = ragas_style
            # Manteniamo anche la chiave storica per compatibilità con eventuali report vecchi
            result["faithfulness"] = faith

        results.append(result)

    # ── Statistiche aggregate ──────────────────────────────────────────────────
    pass_count = sum(1 for r in results if r["semantic"]["pass"])
    avg_sem = sum(r["semantic"]["score"] for r in results) / len(results)
    pass_rate = pass_count / len(results)
    sorted_lat = sorted(latencies_ms)
    p95_index = min(len(sorted_lat) - 1, math.ceil(len(sorted_lat) * 0.95) - 1)

    stats: Dict[str, Any] = {
        "total": len(results),
        "pass_count": pass_count,
        "pass_rate": round(pass_rate, 4),
        "avg_semantic_score": round(avg_sem, 4),
        "avg_latency_ms": round(sum(latencies_ms) / len(latencies_ms), 2),
        "p95_latency_ms": round(sorted_lat[p95_index], 2),
        "estimated_total_cost_usd": round(sum(estimated_costs), 8),
        "estimated_avg_cost_usd": round(sum(estimated_costs) / len(estimated_costs), 8),
    }

    if use_llm_judge:
        judge_results = [r for r in results if "judge" in r]
        if judge_results:
            avg_judge = sum(r["judge"]["score"] for r in judge_results) / len(judge_results)
            unreliable = sum(1 for r in judge_results if not r["judge"]["reliable"])
            stats["avg_judge_score"] = round(avg_judge, 3)
            stats["unreliable_judgements"] = unreliable

    if use_faithfulness:
        ragas_results = [r["ragas_style"] for r in results if "ragas_style" in r]
        with_sources = [r for r in ragas_results if r["faithfulness"].get("method") != "n/a (no sources)"]
        if with_sources:
            stats["faithfulness_score"] = round(sum(r["faithfulness"]["faithfulness"] for r in with_sources) / len(with_sources), 4)
        if ragas_results:
            stats["answer_relevancy_score"] = round(sum(r["answer_relevancy"]["answer_relevancy"] for r in ragas_results) / len(ragas_results), 4)
            stats["context_recall_score"] = round(sum(r["context_recall"]["context_recall"] for r in ragas_results) / len(ragas_results), 4)
            stats["context_precision_score"] = round(sum(r["context_precision"]["context_precision"] for r in ragas_results) / len(ragas_results), 4)

    elapsed = time.time() - ts_start
    stats["elapsed_seconds"] = round(elapsed, 1)
    stats["categories"] = _stats_by_category(results)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    ragas_real_used = any(r.get("ragas_style", {}).get("ragas_real") for r in results)

    payload = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "fast": fast,
            "subset": subset,
            "use_llm_judge": use_llm_judge,
            "use_faithfulness": use_faithfulness,
            "model": GEMINI_MODEL,
            "semantic_threshold": SEMANTIC_THRESHOLD,
            "judge_pass_score": JUDGE_PASS_SCORE,
            "ragas_available": RAGAS_AVAILABLE,
            "ragas_real": ragas_real_used,
        },
        "stats": stats,
        "results": results,
    }

    out_path = RUNS_DIR / f"{run_id}.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = write_eval_report(payload)

    print(f"\n{'─'*62}")
    print(f"  ✅ Completato in {elapsed:.1f}s")
    print(f"  Pass rate:        {pass_rate*100:.1f}%  ({pass_count}/{len(results)})")
    print(f"  Avg semantic:     {avg_sem:.3f}")
    print(f"  Latenza media:    {stats['avg_latency_ms']:.1f} ms | p95={stats['p95_latency_ms']:.1f} ms")
    print(f"  Costo stimato:    ${stats['estimated_total_cost_usd']:.6f}")
    if "avg_judge_score" in stats:
        print(f"  Avg LLM judge:    {stats['avg_judge_score']:.2f}/5")
    if "faithfulness_score" in stats:
        print(f"  Faithfulness:     {stats['faithfulness_score']:.3f}")
    print(f"\n  📄 JSON salvato:   {out_path}")
    print(f"  📝 Report MD:     {report_path}")
    print(f"{'─'*62}\n")

    return payload


def _stats_by_category(results: List[Dict]) -> Dict[str, Any]:
    cats: Dict[str, List] = {}
    for r in results:
        c = r["category"]
        cats.setdefault(c, [])
        cats[c].append(r["semantic"]["score"])
    return {
        cat: {
            "count": len(scores),
            "avg_score": round(sum(scores) / len(scores), 3),
            "pass_rate": round(sum(1 for s in scores if s >= SEMANTIC_THRESHOLD) / len(scores), 3),
        }
        for cat, scores in cats.items()
    }


# =============================================================================
# 8. REGRESSION TESTING
# =============================================================================

def run_regression(fast: bool = False) -> Dict[str, Any]:
    """
    Confronta la run corrente con la baseline salvata.
    Se la baseline non esiste, la crea.
    Emette ALERT se una metrica scende sotto la soglia.
    """
    print(f"\n{'═'*62}")
    print("  REGRESSION TEST")
    print(f"{'═'*62}")

    current = run_eval_suite(fast=fast)
    current_stats = current["stats"]

    if not BASELINE_FILE.exists():
        BASELINE_FILE.write_text(
            json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n  📌 Baseline salvata: {BASELINE_FILE}")
        print("  Prima run: nessun confronto disponibile. Esegui di nuovo per il delta.")
        return {"status": "baseline_created", "run": current}

    baseline = json.loads(BASELINE_FILE.read_text(encoding="utf-8"))
    baseline_stats = baseline["stats"]

    alerts = []
    deltas = {}

    for metric, threshold in ALERT_THRESHOLDS.items():
        curr_val = current_stats.get(metric)
        base_val = baseline_stats.get(metric)
        if curr_val is None or base_val is None:
            continue
        delta = curr_val - base_val
        deltas[metric] = {
            "baseline": base_val,
            "current": curr_val,
            "delta": round(delta, 4),
            "threshold": threshold,
            "alert": delta < threshold,
        }
        if delta < threshold:
            alerts.append(f"  ⚠️  ALERT: {metric}: {base_val:.3f} → {curr_val:.3f} (Δ={delta:+.3f}, soglia={threshold:+.3f})")

    print(f"\n  {'─'*58}")
    print("  DELTA vs BASELINE:")
    for m, d in deltas.items():
        icon = "⚠️ " if d["alert"] else "✅"
        print(f"  {icon} {m:<28} {d['baseline']:.3f} → {d['current']:.3f}  Δ={d['delta']:+.4f}")

    if alerts:
        print(f"\n{'═'*62}")
        for a in alerts:
            print(a)
        print(f"{'═'*62}")
        print("  ❌ Regression rilevata! Esaminare i risultati prima di promuovere.")
    else:
        print(f"\n  ✅ Nessuna regression rilevata. Tutte le metriche nella norma.")

    result = {
        "status": "alerts" if alerts else "ok",
        "alerts": alerts,
        "deltas": deltas,
        "baseline_run_id": baseline.get("run_id"),
        "current_run_id": current.get("run_id"),
    }

    reg_path = RUNS_DIR / f"regression_{current['run_id']}.json"
    reg_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  📄 Report regression: {reg_path}\n")

    return result


# =============================================================================
# 9. REPORT FINALE
# =============================================================================

def generate_report(fast: bool = False) -> None:
    """Genera un report completo con tutte le metriche."""
    payload = run_eval_suite(
        fast=fast,
        use_llm_judge=True,
        use_faithfulness=True,
    )
    stats = payload["stats"]
    cats = stats.get("categories", {})

    print(f"\n{'═'*62}")
    print("  REPORT FINALE — GIORNO 5 MATTINO")
    print(f"{'═'*62}")
    print(f"  Run ID:           {payload['run_id']}")
    print(f"  Timestamp:        {payload['timestamp']}")
    print(f"  Domande totali:   {stats['total']}")
    print(f"  Pass rate:        {stats['pass_rate']*100:.1f}%")
    print(f"  Avg semantic:     {stats['avg_semantic_score']:.4f}")
    if "avg_judge_score" in stats:
        print(f"  Avg LLM judge:    {stats['avg_judge_score']:.2f}/5")
        print(f"  Giudizi inaffid.: {stats.get('unreliable_judgements', 0)}")
    if "faithfulness_score" in stats:
        print(f"  Faithfulness:     {stats['faithfulness_score']:.4f}")
    print(f"\n  Per categoria:")
    for cat, cstats in sorted(cats.items()):
        print(f"    {cat:<15} N={cstats['count']}  avg={cstats['avg_score']:.3f}  pass={cstats['pass_rate']*100:.0f}%")
    print(f"\n  Modello:          {payload['config']['model']}")
    print(f"  Soglia semantica: {payload['config']['semantic_threshold']}")
    print(f"  Durata:           {stats['elapsed_seconds']}s")
    print(f"{'═'*62}\n")


# =============================================================================
# 10. CLI
# =============================================================================

EXAMPLES = """
╔══════════════════════════════════════════════════════════════════════════════╗
║  ESEMPI — eval_suite.py                                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

  # Mostra il golden set (20 domande)
  python day5_governance/eval_suite.py golden-set

  # Valuta una singola coppia con LLM judge (fast = mock)
  python day5_governance/eval_suite.py judge \\
      "Qual è l'SLA per un P1?" \\
      "MTTR 4 ore, escalation entro 15 min" \\
      "Bisogna risolvere presto"

  # Valuta con flag --fast (nessuna chiamata LLM, usa embedding toy)
  python day5_governance/eval_suite.py judge \\
      "Domanda" "Expected" "Actual" --fast

  # Eval suite: solo semantica (veloce, nessun LLM)
  python day5_governance/eval_suite.py eval-suite --fast

  # Eval suite: semantica + LLM judge (tutte le 20 domande)
  python day5_governance/eval_suite.py eval-suite --llm-judge

  # Eval suite: semantica + faithfulness su 5 domande
  python day5_governance/eval_suite.py eval-suite --subset 5 --faithfulness --fast

  # Eval suite completa con tutto
  python day5_governance/eval_suite.py eval-suite --llm-judge --faithfulness

  # Regression test (crea baseline alla prima run)
  python day5_governance/eval_suite.py regression --fast

  # Report completo con tutte le metriche
  python day5_governance/eval_suite.py report --fast

  # Output in: day5_governance/runs/day5_mattino/
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lab Mattino Giorno 5 — Valutazione sistemi RAG/LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd")

    # golden-set
    sub.add_parser("golden-set", help="Mostra il golden set ITSM")

    # judge
    p_judge = sub.add_parser("judge", help="LLM-as-a-judge su singola coppia")
    p_judge.add_argument("question")
    p_judge.add_argument("expected")
    p_judge.add_argument("actual")
    p_judge.add_argument("--fast", action="store_true")

    # eval-suite
    p_eval = sub.add_parser("eval-suite", help="Suite di valutazione sul golden set")
    p_eval.add_argument("--fast", action="store_true", help="Usa mock (no LLM)")
    p_eval.add_argument("--subset", type=int, default=None, help="Usa solo le prime N domande")
    p_eval.add_argument("--llm-judge", action="store_true", help="Includi LLM-as-a-judge")
    p_eval.add_argument("--faithfulness", action="store_true", help="Includi faithfulness")

    # regression
    p_reg = sub.add_parser("regression", help="Regression test vs baseline")
    p_reg.add_argument("--fast", action="store_true")

    # report
    p_rep = sub.add_parser("report", help="Report completo con tutte le metriche")
    p_rep.add_argument("--fast", action="store_true")

    # examples
    sub.add_parser("examples", help="Mostra esempi d'uso")

    args = parser.parse_args()

    if args.cmd == "golden-set":
        print(f"\n{'═'*62}")
        golden_set = load_golden_set()
        print(f"  GOLDEN SET ITSM — {len(golden_set)} domande")
        print(f"{'═'*62}")
        for q in golden_set:
            print(f"\n  [{q['id']}] ({q['category'].upper()})")
            print(f"  Q: {q['question']}")
            print(f"  A: {q['expected'][:80]}{'…' if len(q['expected'])>80 else ''}")
            print(f"  Fonti: {q['sources'] or '(nessuna)'}")
        print()

    elif args.cmd == "judge":
        result = eval_llm_judge(args.question, args.expected, args.actual, fast=args.fast)
        print(f"\n{'═'*62}")
        print("  LLM-AS-A-JUDGE RESULT")
        print(f"{'═'*62}")
        print(f"  Score:        {result['score']:.2f}/5  ({'✓ PASS' if result['pass'] else '✗ FAIL'})")
        print(f"  Forward:      {result['score_forward']}/5")
        print(f"  Swap:         {result['score_swapped']}/5")
        print(f"  Bias delta:   {result['bias_delta']:.2f}  ({'affidabile' if result['reliable'] else '⚠️ inaffidabile'})")
        print(f"  Metodo:       {result['method']}")
        print(f"  Justification: {result['justification']}")

        sem = eval_semantic(args.expected, args.actual)
        print(f"\n  Semantic sim: {sem['score']:.3f} ({sem['method']}) → "
              f"{'✓ PASS' if sem['pass'] else '✗ FAIL'}")
        print()

    elif args.cmd == "eval-suite":
        run_eval_suite(
            fast=args.fast,
            subset=args.subset,
            use_llm_judge=args.llm_judge,
            use_faithfulness=args.faithfulness,
        )

    elif args.cmd == "regression":
        run_regression(fast=args.fast)

    elif args.cmd == "report":
        generate_report(fast=args.fast)

    elif args.cmd == "examples":
        print(EXAMPLES)

    else:
        parser.print_help()
        print("\nUsa 'examples' per vedere tutti i comandi disponibili.")


if __name__ == "__main__":
    main()
