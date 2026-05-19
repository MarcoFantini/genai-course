"""
Toolkit CLI per la governance e conformità all'EU AI Act (Reg. 2024/1689) per scopo didattico.

Backend LLM: Google Gemini via LangChain (fallback euristico se API key assente).

════════════════════════════════════════════════════════════════
 COMANDI DISPONIBILI
════════════════════════════════════════════════════════════════

  ai-act-levels
      Stampa i 4 livelli di rischio dell'AI Act con esempi e obblighi.
      Uso: python ai_act.py ai-act-levels

  classify <description> [--fast]
      Classifica un sistema AI in uno dei 4 livelli (inaccettabile / alto /
      limitato / minimo). Senza --fast usa Gemini; con --fast usa regex.
      Uso: python ai_act.py classify "Sistema di scoring creditizio automatico"
      Uso: python ai_act.py classify "Chatbot HR" --fast

  check-prompt <prompt>
      Verifica la conformità Art. 50 di un system prompt (disclosure AI,
      assenza di istruzioni a fingersi umano, ecc.). Restituisce score 0-100.
      Uso: python ai_act.py check-prompt "Sei un assistente IT."

  patch-prompt <prompt> [--fast]
      Corregge automaticamente un system prompt non conforme aggiungendo
      la disclosure Art. 50. Con --fast usa template standard; senza usa LLM.
      Uso: python ai_act.py patch-prompt "Sei un assistente IT." --fast

  validate-raci
      Valida la matrice RACI di governance inclusa nel modulo (SAMPLE_RACI).
      Verifica che ogni attività abbia esattamente un Accountable (A).
      Uso: python ai_act.py validate-raci

  gen-onepager [--fast]
      Genera un documento di governance one-pager per il sistema ITSM di
      esempio: classifica AI Act + compliance check + patch prompt + salvataggio
      JSON timestampato in runs/day5_pomeriggio/.
      Uso: python ai_act.py gen-onepager
      Uso: python ai_act.py gen-onepager --fast

  examples
      Placeholder — rimanda alla docstring per esempi completi.
      Uso: python ai_act.py examples

════════════════════════════════════════════════════════════════
 VARIABILI D'AMBIENTE (.env)
════════════════════════════════════════════════════════════════

  GOOGLE_API_KEY                   chiave API Google (abilita Gemini)
  GEMINI_MODEL                     default: gemini-2.5-flash
  MIN_SECONDS_BETWEEN_MODEL_CALLS  default: 15  (rate limiting)
  PRICE_INPUT_PER_1M               default: 0.10 USD
  PRICE_OUTPUT_PER_1M              default: 0.40 USD
"""

from __future__ import annotations
import sys, io
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage, SystemMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

load_dotenv()

BASE_DIR = Path(__file__).parent.resolve()
RUNS_DIR = BASE_DIR / "runs" / "day5_pomeriggio"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

GOOGLE_API_KEY  = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL    = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
MIN_SECONDS     = float(os.getenv("MIN_SECONDS_BETWEEN_MODEL_CALLS", "15"))
PRICE_IN_PER_1M = float(os.getenv("PRICE_INPUT_PER_1M", "0.10"))
PRICE_OUT_PER_1M = float(os.getenv("PRICE_OUTPUT_PER_1M", "0.40"))

_last_call_ts: float = 0.0

def _rate_limit() -> None:
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

AI_ACT_LEVELS: Dict[str, Dict[str, Any]] = {
    "inaccettabile": {
        "label": "Rischio Inaccettabile (Art. 5)",
        "color": "🔴",
        "description": "Sistemi esplicitamente vietati dal Regolamento AI Act.",
        "examples": [
            "Manipolazione subliminale del comportamento umano",
            "Sfruttamento di vulnerabilità di specifici gruppi (età, disabilità)",
            "Social scoring governativo generalizzato",
            "Riconoscimento facciale real-time in spazi pubblici",
            "Sistemi di previsione comportamenti criminali basati solo su caratteristiche personali",
        ],
        "obligations": [
            "VIETATO sviluppare, distribuire o usare nell'UE",
            "Sanzione fino al 7% del fatturato globale annuo",
        ],
        "articles": ["Art. 5"],
    },
    "alto": {
        "label": "Alto Rischio (Art. 6-7 + Allegato III)",
        "color": "🟠",
        "description": "Sistemi con impatto significativo su salute, sicurezza o diritti fondamentali.",
        "examples": [
            "Sistemi di selezione CV e decisioni occupazionali",
            "Scoring creditizio e valutazione solvibilità",
            "Diagnostica medica e trattamenti clinici",
            "Sistemi di sicurezza infrastrutture critiche",
            "Sistemi giudiziari e di law enforcement",
        ],
        "obligations": [
            "Valutazione conformità prima del deployment (CONFORMITY ASSESSMENT)",
            "Sistema di gestione del rischio documentato",
            "Governance dei dati di training (rappresentatività, bias)",
            "Documentazione tecnica completa e aggiornata",
            "Logging automatico delle operazioni (AUDIT TRAIL)",
            "Supervisione umana obbligatoria (HUMAN OVERSIGHT)",
            "Robustezza, accuratezza e cybersecurity certificate",
            "Registrazione nel database UE dei sistemi ad alto rischio",
            "Sanzione fino al 3% del fatturato globale annuo",
        ],
        "articles": ["Art. 6", "Art. 7", "Art. 9-15", "Allegato III"],
    },
    "limitato": {
        "label": "Rischio Limitato (Art. 50)",
        "color": "🟡",
        "description": "Sistemi con obblighi di trasparenza.",
        "examples": [
            "Chatbot di customer service",
            "Assistenti AI aziendali (ITSM, HR, knowledge base)",
            "Sistemi di generazione di testi/immagini/audio",
            "Deepfake e contenuti sintetici",
        ],
        "obligations": [
            "DISCLOSURE obbligatoria: informare l'utente che sta interagendo con AI",
            "Deepfake devono essere etichettati come generati da AI",
            "Il provider deve informare che il contenuto è sintetico",
            "Sanzione fino all'1.5% del fatturato globale annuo",
        ],
        "articles": ["Art. 50"],
    },
    "minimo": {
        "label": "Rischio Minimo",
        "color": "🟢",
        "description": "Sistemi a basso impatto. Nessun obbligo specifico.",
        "examples": [
            "Filtri antispam",
            "AI nei videogiochi (NPC)",
            "Sistemi di raccomandazione prodotti base",
            "Strumenti di produttività personale (spell-check, autocomplete)",
        ],
        "obligations": [
            "Nessun obbligo regolatorio specifico",
            "Raccomandate: documentazione, testing, monitoraggio volontario",
            "Codici di condotta volontari (Art. 95)",
        ],
        "articles": ["Art. 95 (codici condotta volontari)"],
    },
}

_CLASSIFY_SYSTEM = """\
Sei un esperto di AI Act EU (Regolamento (UE) 2024/1689).
Classifica il sistema AI descritto secondo uno dei 4 livelli di rischio:
- inaccettabile: sistemi vietati (Art. 5)
- alto: sistemi ad alto rischio (Art. 6-7, Allegato III)
- limitato: obblighi di trasparenza (Art. 50)
- minimo: nessun obbligo specifico

Rispondi SOLO con JSON valido:
{
  "level": "<inaccettabile|alto|limitato|minimo>",
  "confidence": <0.0-1.0>,
  "justification": "<spiegazione 1-2 frasi citando articoli rilevanti>",
  "key_factors": ["<fattore 1>", "<fattore 2>"]
}
"""

_CLASSIFY_HUMAN = "Descrizione del sistema AI:\n{description}"

_HEURISTIC_RULES: List[Tuple[str, str, float]] = [
    (r"manipol\w+\s+sublimin|social\s+scor\w+\s+govern|sorveglianza\s+di\s+massa|"
     r"riconosci\w+\s+faccial\w+\s+real.?time|previs\w+\s+crimin", "inaccettabile", 0.90),
    (r"selezi\w+.*?\bcv\b|\bcv\b.*?selezi\w+|curriculum|screening\s+candidat|punteggio\s+credit|"
     r"solvibilit|diagnostica\s+medic|diagnosi\s+clinic|infrastruttur\w+\s+critic|"
     r"law\s+enforcement|giustizia|giudiziario|istruzione\s+e\s+valutaz", "alto", 0.80),
    (r"chatbot|assistente\s+ai|customer\s+service|service\s+desk|helpdesk|"
     r"knowledge\s+base|itsm|supporto\s+utenti|deepfake|contenut\w+\s+sintet", "limitato", 0.75),
]

def classify_ai_act(description: str, fast: bool = False) -> Dict[str, Any]:
    if fast or not LANGCHAIN_AVAILABLE or not GOOGLE_API_KEY:
        desc_lower = description.lower()
        level = "minimo"
        confidence = 0.60
        justification = "Classificazione euristica — nessun indicatore di rischio elevato rilevato."
        key_factors = ["nessun pattern ad alto rischio"]

        for pattern, lv, conf in _HEURISTIC_RULES:
            if re.search(pattern, desc_lower):
                level = lv
                confidence = conf
                justification = f"Classificazione euristica: pattern rilevato → livello '{lv}'."
                key_factors = [f"Pattern: '{pattern[:40]}...'"]
                break

        return {
            "level": level,
            "confidence": confidence,
            "justification": justification,
            "key_factors": key_factors,
            "method": "heuristic",
            "details": AI_ACT_LEVELS[level],
        }

    _rate_limit()
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=GOOGLE_API_KEY, temperature=0.0)
    messages = [
        SystemMessage(content=_CLASSIFY_SYSTEM),
        HumanMessage(content=_CLASSIFY_HUMAN.format(description=description)),
    ]
    resp = llm.invoke(messages)
    raw = resp.content.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        data = json.loads(raw)
        level = data.get("level", "minimo")
        if level not in AI_ACT_LEVELS:
            level = "minimo"
        return {
            "level": level,
            "confidence": float(data.get("confidence", 0.7)),
            "justification": data.get("justification", ""),
            "key_factors": data.get("key_factors", []),
            "method": "llm",
            "details": AI_ACT_LEVELS[level],
        }
    except Exception as e:
        return {
            "level": "minimo",
            "confidence": 0.0,
            "justification": f"Errore parsing: {e}",
            "key_factors": [],
            "method": "llm_error",
            "details": AI_ACT_LEVELS["minimo"],
        }

_DISCLOSURE_PATTERNS = [
    r"sei\s+un\s+(assistente\s+)?ai",
    r"sono\s+un\s+(assistente\s+)?ai",
    r"questa\s+è\s+una\s+conversazione\s+con\s+(un\s+)?ai",
    r"stai\s+parlando\s+con\s+(un\s+)?ai",
    r"assistente\s+virtuale",
    r"sistema\s+di\s+intelligenza\s+artificiale",
    r"i\s+am\s+an?\s+ai",
    r"you\s+are\s+(talking\s+to\s+)?an?\s+ai",
    r"powered\s+by\s+ai",
    r"this\s+is\s+an?\s+ai",
    r"art\.?\s*50",
    r"disclosure",
    r"nota:\s*(questo|questa)\s*(è|sistema)",
]

_RISK_PATTERNS = [
    (r"sei\s+umano|sei\s+una\s+persona|non\s+sei\s+(un\s+)?ai",
     "Istruzione che nega la natura AI del sistema — violazione Art. 50"),
    (r"non\s+(dire|rivelare|comunicare)\s+che\s+sei\s+(un\s+)?ai",
     "Istruzione esplicita a nascondere la natura AI — violazione Art. 50"),
    (r"fingiti?\s+(una\s+persona|umano)|simula\s+di\s+essere\s+umano",
     "Istruzione a fingersi umano — violazione Art. 50"),
    (r"ignora\s+(le\s+)?(tue\s+)?(istruzioni|linee\s+guida)|dimentica\s+le\s+regole",
     "Istruzione jailbreak — violazione policy sicurezza"),
    (r"password\s*[:=]\s*['\"]?\w|api[_\-]?key\s*[:=]\s*['\"]?\w|sk-[a-zA-Z0-9]{10,}",
     "Presenza di credenziali o chiavi API nel system prompt — rischio sicurezza"),
]

def check_prompt_compliance(system_prompt: str) -> Dict[str, Any]:
    prompt_lower = system_prompt.lower()
    issues = []
    warnings = []
    recommendations = []

    has_disclosure = any(re.search(p, prompt_lower) for p in _DISCLOSURE_PATTERNS)
    if not has_disclosure:
        issues.append(
            "MANCANTE: Il prompt non contiene una disclosure esplicita della natura AI "
            "(Art. 50 AI Act). Aggiungere una clausola che informi l'utente."
        )
        recommendations.append(
            "Aggiungere in apertura: 'Sei un assistente AI. L'utente è informato di stare "
            "interagendo con un sistema di intelligenza artificiale.'"
        )

    for pattern, description in _RISK_PATTERNS:
        if re.search(pattern, prompt_lower):
            issues.append(f"VIOLAZIONE: {description}")

    word_count = len(system_prompt.split())
    if word_count < 20:
        warnings.append(f"System prompt molto breve ({word_count} parole).")
    if word_count > 2000:
        warnings.append(f"System prompt molto lungo ({word_count} parole).")

    has_role = bool(re.search(r"sei\s+un|il\s+tuo\s+ruolo|il\s+tuo\s+scopo|"
                               r"you\s+are|your\s+role|your\s+purpose", prompt_lower))
    if not has_role:
        warnings.append("Il prompt non definisce esplicitamente il ruolo/scopo dell'assistente.")
        recommendations.append("Aggiungere: 'Il tuo ruolo è [descrizione del ruolo e dello scopo].'")

    score = 100
    score -= len(issues) * 25
    score -= len(warnings) * 5
    score = max(0, min(100, score))

    return {
        "compliant": len(issues) == 0,
        "has_disclosure": has_disclosure,
        "issues": issues,
        "warnings": warnings,
        "score": score,
        "recommendations": recommendations,
        "word_count": word_count,
    }

_ART50_DISCLOSURE_IT = """\
[DISCLOSURE AI ACT ART. 50]
Sei un assistente AI. L'utente con cui stai interagendo è informato di stare
comunicando con un sistema di intelligenza artificiale e non con un essere umano.
"""

_ART50_DISCLOSURE_EN = """\
[AI ACT ART. 50 DISCLOSURE]
You are an AI assistant. The user interacting with you is informed that they
are communicating with an artificial intelligence system, not a human being.
"""

_PATCH_SYSTEM = """\
Sei un esperto di AI governance e conformità all'AI Act EU.
Il tuo compito è migliorare un system prompt per renderlo conforme all'Art. 50
senza alterare le istruzioni operative esistenti.
Rispondi con il prompt corretto e nulla altro.
"""

def patch_prompt(system_prompt: str, fast: bool = False) -> Dict[str, Any]:
    check = check_prompt_compliance(system_prompt)

    if check["compliant"]:
        return {
            "patched": False,
            "original": system_prompt,
            "result": system_prompt,
            "changes": [],
            "compliance_before": check,
            "compliance_after": check,
            "method": "no_change_needed",
        }

    if fast or not LANGCHAIN_AVAILABLE or not GOOGLE_API_KEY:
        lang_hint = "en" if re.search(r"\bthe\b|\byour\b|\byou\b", system_prompt[:200].lower()) else "it"
        disclosure = _ART50_DISCLOSURE_EN if lang_hint == "en" else _ART50_DISCLOSURE_IT
        patched_prompt = disclosure + "\n" + system_prompt
        check_after = check_prompt_compliance(patched_prompt)
        return {
            "patched": True,
            "original": system_prompt,
            "result": patched_prompt,
            "changes": ["Aggiunta disclosure Art. 50 in apertura (template standard)"],
            "compliance_before": check,
            "compliance_after": check_after,
            "method": "template",
        }

    _rate_limit()
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=GOOGLE_API_KEY, temperature=0.2)
    messages = [
        SystemMessage(content=_PATCH_SYSTEM),
        HumanMessage(content=f"System prompt da correggere:\n\n{system_prompt}"),
    ]
    resp = llm.invoke(messages)
    patched_prompt = resp.content.strip()
    check_after = check_prompt_compliance(patched_prompt)
    return {
        "patched": True,
        "original": system_prompt,
        "result": patched_prompt,
        "changes": ["Disclosure Art. 50 integrata contestualmente via LLM"],
        "compliance_before": check,
        "compliance_after": check_after,
        "method": "llm",
    }

SAMPLE_RACI: Dict[str, Any] = {
    "project": "Assistente AI ITSM — Governance RACI",
    "roles": ["AI Team", "Business Owner", "CISO", "Legal", "IT Ops"],
    "activities": [
        ["Definizione requisiti sistema",        "R", "A", "C", "C", "I"],
        ["Classificazione AI Act",               "R", "C", "C", "A", "I"],
        ["Data governance e qualità dataset",    "R", "A", "C", "C", "I"],
        ["Training e fine-tuning modello",       "A", "C", "I", "I", "R"],
        ["Valutazione bias e fairness",          "R", "C", "A", "C", "I"],
        ["Security review e penetration test",  "C", "I", "A", "I", "R"],
        ["Approvazione deployment produzione",   "C", "A", "C", "I", "R"],
        ["Monitoring operativo post-deploy",     "C", "I", "I", "I", "A"],
        ["Incident response AI",                 "R", "A", "A", "I", "R"],
        ["Revisione policy e compliance",        "I", "C", "C", "A", "I"],
        ["Comunicazione utenti (Art. 50)",       "R", "A", "C", "R", "I"],
        ["Audit annuale sistema",                "C", "I", "C", "A", "R"],
    ],
}

RACI_VALID_VALUES = {"R", "A", "C", "I", "-", ""}

def validate_raci(raci: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if raci is None:
        raci = SAMPLE_RACI

    roles = raci.get("roles", [])
    activities = raci.get("activities", [])
    issues = []
    warnings = []
    fixed_matrix = []

    for i, row in enumerate(activities):
        if not row:
            issues.append(f"Riga {i+1}: vuota")
            continue

        activity_name = row[0] if isinstance(row[0], str) else f"Attività {i+1}"
        assignments = [str(v).strip().upper() for v in row[1:]]

        for j, v in enumerate(assignments):
            if v not in RACI_VALID_VALUES:
                issues.append(f"'{activity_name}': valore non valido '{v}'")

        a_count = assignments.count("A")
        if a_count == 0:
            issues.append(f"'{activity_name}': nessun Accountable (A)")
        elif a_count > 1:
            a_roles = [roles[j] for j, v in enumerate(assignments) if v == "A" and j < len(roles)]
            issues.append(f"'{activity_name}': {a_count} Accountable (A) trovati {a_roles} — deve essere esattamente 1.")

        r_count = assignments.count("R")
        if r_count == 0:
            warnings.append(f"'{activity_name}': nessun Responsible (R)")

        active = [v for v in assignments if v not in ("I", "-", "")]
        if not active:
            warnings.append(f"'{activity_name}': tutti i ruoli sono Informed (I)")

        if a_count != 1:
            fixed_row = [activity_name]
            for j, v in enumerate(assignments):
                if a_count == 0 and j == 0:
                    fixed_row.append("A")
                elif a_count > 1 and v == "A" and j > assignments.index("A"):
                    fixed_row.append("R")
                else:
                    fixed_row.append(v)
            fixed_matrix.append({"activity": activity_name, "original": row[1:], "suggested": fixed_row[1:]})

    stats = {
        "total_activities": len(activities),
        "activities_with_a_issues": sum(1 for i in issues if "Accountable" in i or "A)" in i),
        "activities_with_r_warnings": len(warnings),
        "total_issues": len(issues),
        "total_warnings": len(warnings),
    }

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "stats": stats,
        "fixed_suggestions": fixed_matrix,
    }

ONEPAGER_TEMPLATE = {
    "nome_sistema": "Assistente AI ITSM — Knowledge Base",
    "owner": "IT Operations / Business Owner: [Nome]",
    "data_assessment": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    "ai_act_classification": {"level": "", "justification": "", "articles": []},
    "description": (
        "Assistente AI per la gestione delle richieste ITSM. Risponde a domande su "
        "policy, SLA e procedure dal knowledge base aziendale. Interagisce direttamente "
        "con i dipendenti tramite portale web e Slack."
    ),
    "use_case": "Supporto tier-1 automatizzato, deviazione dei ticket verso self-service",
    "data_used": [
        "Knowledge base ITSM interno (non dati personali)",
        "Log delle sessioni di chat (anonimizzati dopo 30 giorni)",
    ],
    "obligations": [],
    "human_oversight": "Supervisione umana attiva: l'utente può sempre escalare a un agente umano.",
    "disclosure_compliant": False,
    "system_prompt_sample": "",
    "monitoring": [
        "Logging di tutte le sessioni con ID anonimo",
        "Revisione settimanale delle metriche di qualità",
        "Alert automatico se pass_rate < 80%",
        "Audit trimestrale con revisione manuale 50 conversazioni campione",
    ],
    "risk_mitigations": [
        "Guardrail: blocco domande fuori scope",
        "Guardrail: no condivisione dati sensibili o credenziali",
        "Fallback a agente umano per richieste complesse",
        "No decisioni automatiche vincolanti (solo advisory)",
    ],
    "review_schedule": "Trimestrale + ad ogni aggiornamento del modello",
    "contacts": {
        "ai_owner": "AI Team Lead — [email]",
        "dpo": "Data Protection Officer — [email]",
        "ciso": "Chief Information Security Officer — [email]",
    },
}

_ONEPAGER_SYSTEM = """\
Sei un esperto di AI governance per sistemi enterprise.
Compila il seguente template di governance one-pager per un sistema AI aziendale.
Rispondi SOLO con JSON valido completando tutti i campi.
"""

def generate_onepager(fast: bool = False) -> Dict[str, Any]:
    onepager = dict(ONEPAGER_TEMPLATE)

    classification = classify_ai_act(onepager["description"], fast=fast)
    onepager["ai_act_classification"] = {
        "level": classification["level"],
        "label": AI_ACT_LEVELS[classification["level"]]["label"],
        "justification": classification["justification"],
        "articles": AI_ACT_LEVELS[classification["level"]]["articles"],
    }
    onepager["obligations"] = AI_ACT_LEVELS[classification["level"]]["obligations"]

    sample_prompt = (
        "Sei un assistente AI per il supporto IT aziendale (ITSM). "
        "Rispondi solo a domande relative a policy IT, SLA, procedure e knowledge base aziendale. "
        "Non fornire informazioni su sistemi esterni, credenziali o dati sensibili. "
        "Se la domanda è fuori scope, informa l'utente e suggerisci di contattare il Service Desk."
    )
    onepager["system_prompt_sample"] = sample_prompt

    compliance = check_prompt_compliance(sample_prompt)
    onepager["disclosure_compliant"] = compliance["compliant"]

    if not compliance["compliant"]:
        patch = patch_prompt(sample_prompt, fast=fast)
        onepager["system_prompt_sample"] = patch["result"]
        onepager["disclosure_compliant"] = patch["compliance_after"]["compliant"]
        onepager["_patch_applied"] = True
        onepager["_patch_method"] = patch["method"]

    if not fast and LANGCHAIN_AVAILABLE and GOOGLE_API_KEY:
        _rate_limit()
        llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=GOOGLE_API_KEY, temperature=0.3)
        messages = [
            SystemMessage(content=_ONEPAGER_SYSTEM),
            HumanMessage(content=(
                f"Ecco il template:\n{json.dumps(onepager, ensure_ascii=False, indent=2)}\n\n"
                "Completa i campi mancanti. Rispondi SOLO con il JSON aggiornato."
            )),
        ]
        resp = llm.invoke(messages)
        raw = resp.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        try:
            enriched = json.loads(raw)
            onepager.update(enriched)
            onepager["_enriched_by_llm"] = True
        except Exception:
            onepager["_enrichment_failed"] = True

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = RUNS_DIR / f"onepager_{run_id}.json"
    out_path.write_text(json.dumps(onepager, ensure_ascii=False, indent=2), encoding="utf-8")
    onepager["_saved_to"] = str(out_path)
    return onepager

def _print_classify_result(result: Dict[str, Any]) -> None:
    level = result["level"]
    details = result["details"]
    print(f"\n{'═'*62}")
    print(f"  CLASSIFICAZIONE AI ACT")
    print(f"{'═'*62}")
    print(f"  Livello:      {details['color']} {details['label']}")
    print(f"  Confidenza:   {result['confidence']*100:.0f}%")
    print(f"  Metodo:       {result['method']}")
    print(f"  Giustificazione:")
    for line in result["justification"].split(". "):
        if line.strip():
            print(f"    • {line.strip()}")
    if result.get("key_factors"):
        print(f"  Fattori chiave:")
        for f in result["key_factors"]:
            print(f"    - {f}")
    print(f"\n  Articoli rilevanti: {', '.join(details['articles'])}")
    print(f"\n  Obblighi principali:")
    for ob in details["obligations"]:
        print(f"    ✓ {ob}")
    print()

def _print_check_result(result: Dict[str, Any]) -> None:
    status = "✅ COMPLIANT" if result["compliant"] else "❌ NON COMPLIANT"
    print(f"\n{'═'*62}")
    print(f"  SYSTEM PROMPT COMPLIANCE CHECK")
    print(f"{'═'*62}")
    print(f"  Stato:        {status}")
    print(f"  Score:        {result['score']}/100")
    print(f"  Disclosure:   {'✓ presente' if result['has_disclosure'] else '✗ MANCANTE'}")
    print(f"  Lunghezza:    {result['word_count']} parole")
    if result["issues"]:
        print(f"\n  Problemi critici ({len(result['issues'])}):")
        for i in result["issues"]:
            print(f"    ❌ {i}")
    if result["warnings"]:
        print(f"\n  Warning ({len(result['warnings'])}):")
        for w in result["warnings"]:
            print(f"    ⚠️  {w}")
    if result["recommendations"]:
        print(f"\n  Raccomandazioni:")
        for r in result["recommendations"]:
            print(f"    → {r}")
    print()

def main() -> None:
    parser = argparse.ArgumentParser(description="Lab Pomeriggio Giorno 5 — Governance AI Act")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("ai-act-levels")
    p_cl = sub.add_parser("classify")
    p_cl.add_argument("description")
    p_cl.add_argument("--fast", action="store_true")
    p_chk = sub.add_parser("check-prompt")
    p_chk.add_argument("prompt")
    p_patch = sub.add_parser("patch-prompt")
    p_patch.add_argument("prompt")
    p_patch.add_argument("--fast", action="store_true")
    sub.add_parser("validate-raci")
    p_op = sub.add_parser("gen-onepager")
    p_op.add_argument("--fast", action="store_true")
    sub.add_parser("examples")

    args = parser.parse_args()

    if args.cmd == "ai-act-levels":
        print(f"\n{'═'*62}")
        print("  AI ACT EU — 4 LIVELLI DI RISCHIO")
        print(f"{'═'*62}")
        for level_key, level_data in AI_ACT_LEVELS.items():
            print(f"\n  {level_data['color']} {level_data['label']}")
            print(f"  {level_data['description']}")
            print(f"  Articoli: {', '.join(level_data['articles'])}")
            print(f"  Esempi:")
            for ex in level_data["examples"][:3]:
                print(f"    • {ex}")
            print(f"  Obblighi:")
            for ob in level_data["obligations"][:2]:
                print(f"    ✓ {ob}")
        print()

    elif args.cmd == "classify":
        result = classify_ai_act(args.description, fast=args.fast)
        _print_classify_result(result)

    elif args.cmd == "check-prompt":
        result = check_prompt_compliance(args.prompt)
        _print_check_result(result)

    elif args.cmd == "patch-prompt":
        print(f"\n  Original:\n  {args.prompt}\n")
        result = patch_prompt(args.prompt, fast=args.fast)
        if not result["patched"]:
            print("  ✅ System prompt già conforme — nessuna modifica necessaria.\n")
        else:
            print(f"{'═'*62}")
            print("  PATCHING RESULT")
            print(f"{'═'*62}")
            print(f"  Metodo: {result['method']}")
            print(f"  Modifiche: {result['changes']}")
            print(f"\n  Compliance PRIMA: score={result['compliance_before']['score']} | compliant={result['compliance_before']['compliant']}")
            print(f"  Compliance DOPO:  score={result['compliance_after']['score']} | compliant={result['compliance_after']['compliant']}")
            residual_issues = result['compliance_after'].get('issues', [])
            if residual_issues:
                print(f"\n  ⚠️  Problemi residui (richiedono correzione manuale):")
                for issue in residual_issues:
                    print(f"    ❌ {issue}")
            print(f"\n  Prompt corretto:\n")
            print(f"  {'─'*58}")
            for line in result["result"].split("\n"):
                print(f"  {line}")
            print(f"  {'─'*58}\n")

    elif args.cmd == "validate-raci":
        result = validate_raci()
        print(f"\n{'═'*62}")
        print(f"  RACI VALIDATOR — {SAMPLE_RACI['project']}")
        print(f"{'═'*62}")
        roles = SAMPLE_RACI["roles"]
        header = f"  {'Attività':<40} " + "  ".join(f"{r[:10]:<10}" for r in roles)
        print(f"\n{header}")
        print(f"  {'─'*58}")
        for row in SAMPLE_RACI["activities"]:
            name = row[0][:38]
            vals = "  ".join(f"{str(v):<10}" for v in row[1:])
            print(f"  {name:<40} {vals}")
        print(f"\n  {'─'*58}")
        print(f"  Stato:    {'✅ VALIDA' if result['valid'] else '❌ INVALIDA'}")
        print(f"  Issues:   {result['stats']['total_issues']}")
        print(f"  Warnings: {result['stats']['total_warnings']}")
        if result["issues"]:
            print(f"\n  ❌ Problemi:")
            for i in result["issues"]:
                print(f"     {i}")
        if result["warnings"]:
            print(f"\n  ⚠️  Warning:")
            for w in result["warnings"]:
                print(f"     {w}")
        if result["fixed_suggestions"]:
            print(f"\n  💡 Suggerimenti correzione:")
            for fix in result["fixed_suggestions"]:
                print(f"     '{fix['activity']}': {fix['original']} → {fix['suggested']}")
        print()

    elif args.cmd == "gen-onepager":
        print(f"\n  Generazione governance one-pager … (fast={args.fast})")
        onepager = generate_onepager(fast=args.fast)
        print(f"\n{'═'*62}")
        print("  GOVERNANCE ONE-PAGER")
        print(f"{'═'*62}")
        print(f"  Sistema:      {onepager['nome_sistema']}")
        print(f"  Owner:        {onepager['owner']}")
        print(f"  Data:         {onepager['data_assessment']}")
        cls = onepager.get("ai_act_classification", {})
        lvl = cls.get("level", "?")
        print(f"\n  AI Act:       {AI_ACT_LEVELS.get(lvl, {}).get('color', '')} {cls.get('label', lvl)}")
        print(f"  Articoli:     {', '.join(cls.get('articles', []))}")
        print(f"  Disclosure:   {'✅ Compliant' if onepager.get('disclosure_compliant') else '❌ Non compliant'}")
        print(f"\n  Obblighi ({len(onepager.get('obligations', []))}):")
        for ob in onepager.get("obligations", [])[:4]:
            print(f"    • {ob[:70]}")
        saved = onepager.get("_saved_to", "")
        if saved:
            print(f"\n  📄 Salvato: {saved}")
        print()

    elif args.cmd == "examples":
        print("Vedere docstring per esempi completi.")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()