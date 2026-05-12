"""
bonus_rag.py

RAG bonus Giorno 1:
- corpus eterogeneo pubblico
- chunking più grande
- metadata filtering
- retrieval cross-domain
- prompt con controllo fonti
- logging in bonus_qa_log.md

Comandi:

    python bonus_rag.py ingest
    python bonus_rag.py retrieve "What is prompt injection and how can it affect an AI agent?" --domain security
    python bonus_rag.py ask "Compare NIST AI RMF and OWASP guidance for GenAI risk." --domain all --log
    python bonus_rag.py batch-test
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings


load_dotenv()

BASE_DIR = Path(__file__).parent.resolve()
EXTRACTED_DIR = BASE_DIR / "corpus" / "extracted"

CHROMA_PATH = Path(os.getenv("BONUS_CHROMA_PATH", "./chroma_bonus"))
if not CHROMA_PATH.is_absolute():
    CHROMA_PATH = BASE_DIR / CHROMA_PATH

COLLECTION_NAME = os.getenv("BONUS_COLLECTION_NAME", "enterprise_ai_governance_rag")

TOP_K = int(os.getenv("BONUS_TOP_K", "5"))
CHUNK_SIZE = int(os.getenv("BONUS_CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("BONUS_CHUNK_OVERLAP", "85"))

LLM_MODE = os.getenv("LLM_MODE", "mock").lower().strip()
QA_LOG_PATH = BASE_DIR / "bonus_qa_log.md"

EMBEDDING_DIM = 512


# ---------------------------------------------------------------------
# Embedding locale robusto e didattico
# ---------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-ZÀ-ÿ0-9]+", text.lower())


def hash_embedding(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    vec = [0.0] * dim

    for token in tokenize(text):
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        idx = int(digest[:8], 16) % dim
        sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
        vec[idx] += sign

    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return vec

    return [x / norm for x in vec]


class HashEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        return [hash_embedding(text) for text in input]


# ---------------------------------------------------------------------
# Parsing metadati e chunking
# ---------------------------------------------------------------------

@dataclass
class SourceDoc:
    source_id: str
    title: str
    domain: str
    url: str
    text: str


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict


def parse_extracted_file(path: Path) -> SourceDoc:
    raw = path.read_text(encoding="utf-8", errors="ignore")

    source_id = extract_header_value(raw, "SOURCE_ID") or path.stem
    title = extract_header_value(raw, "TITLE") or path.stem
    domain = extract_header_value(raw, "DOMAIN") or "unknown"
    url = extract_header_value(raw, "URL") or ""

    # Rimuove header iniziale.
    body = re.sub(
        r"(?s)^SOURCE_ID:.*?\n\n",
        "",
        raw,
        count=1,
    ).strip()

    return SourceDoc(
        source_id=source_id,
        title=title,
        domain=domain,
        url=url,
        text=body,
    )


def extract_header_value(raw: str, key: str) -> str | None:
    pattern = rf"^{re.escape(key)}:\s*(.+)$"
    match = re.search(pattern, raw, flags=re.MULTILINE)
    if match:
        return match.group(1).strip()
    return None


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()

    chunks = []
    start = 0

    while start < len(cleaned):
        end = start + chunk_size
        chunk = cleaned[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(cleaned):
            break

        start = end - chunk_overlap

    return chunks


def load_source_docs() -> list[SourceDoc]:
    if not EXTRACTED_DIR.exists():
        raise RuntimeError(
            "Cartella corpus/extracted non trovata. "
            "Esegui prima: python download_corpus.py"
        )

    files = sorted(EXTRACTED_DIR.glob("*.txt"))

    if not files:
        raise RuntimeError(
            "Nessun file .txt trovato in corpus/extracted. "
            "Esegui prima: python download_corpus.py"
        )

    return [parse_extracted_file(path) for path in files]


def build_chunks() -> list[Chunk]:
    source_docs = load_source_docs()
    chunks = []

    for source in source_docs:
        parts = split_text(source.text, CHUNK_SIZE, CHUNK_OVERLAP)

        for idx, part in enumerate(parts):
            print("_______________________\n")
            print(part, "\n")
            chunk_id = f"{source.source_id}::chunk_{idx}"
            chunks.append(
                Chunk(
                    id=chunk_id,
                    text=part,
                    metadata={
                        "source_id": source.source_id,
                        "title": source.title,
                        "domain": source.domain,
                        "url": source.url,
                        "chunk_index": idx,
                    },
                )
            )

    return chunks


# ---------------------------------------------------------------------
# Chroma
# ---------------------------------------------------------------------

def get_collection(reset: bool = False):
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=HashEmbeddingFunction(),
    )


def ingest() -> None:
    chunks = build_chunks()
    collection = get_collection(reset=True)

    collection.add(
        ids=[c.id for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[c.metadata for c in chunks],
    )

    print(f"Indicizzati {len(chunks)} chunk.")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Chroma path: {CHROMA_PATH}")


def retrieve(question: str, domain: str | None = None, k: int = TOP_K) -> list[dict]:
    collection = get_collection(reset=False)

    where = None
    if domain and domain != "all":
        where = {"domain": domain}

    result = collection.query(
        query_texts=[question],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    docs = result["documents"][0]
    metas = result["metadatas"][0]
    distances = result["distances"][0]

    rows = []

    for doc, meta, distance in zip(docs, metas, distances):
        rows.append(
            {
                "text": doc,
                "metadata": meta,
                "distance": distance,
            }
        )

    return rows


def print_results(rows: list[dict]) -> None:
    for i, row in enumerate(rows, start=1):
        meta = row["metadata"]

        print("=" * 100)
        print(f"RESULT {i}")
        print(f"distance    : {row['distance']:.4f}")
        print(f"domain      : {meta.get('domain')}")
        print(f"source_id   : {meta.get('source_id')}")
        print(f"title       : {meta.get('title')}")
        print(f"chunk_index : {meta.get('chunk_index')}")
        print(f"url         : {meta.get('url')}")
        print("-" * 100)
        print(row["text"][:1500])


# ---------------------------------------------------------------------
# Prompt e LLM
# ---------------------------------------------------------------------

def build_prompt(question: str, rows: list[dict]) -> str:
    context_blocks = []

    for i, row in enumerate(rows, start=1):
        meta = row["metadata"]
        context_blocks.append(
            f"[SOURCE {i}]\n"
            f"title={meta.get('title')}\n"
            f"domain={meta.get('domain')}\n"
            f"url={meta.get('url')}\n"
            f"chunk={meta.get('chunk_index')}\n"
            f"{row['text']}"
        )

    context = "\n\n".join(context_blocks)

    return f"""
Sei un assistente enterprise specializzato in GenAI governance, security e reliability.

Rispondi usando SOLO il contesto fornito.

Regole:
- Se il contesto non basta, dichiaralo esplicitamente. 
- Distingui chiaramente tra requisiti normativi, raccomandazioni e best practice.
- Se la domanda confronta più fonti, cita quali fonti supportano quale parte della risposta.
- Non inventare obblighi legali.
- Non inventare policy aziendali.
- Rispondi in italiano.
- Chiudi sempre con una sezione "Fonti usate".

<context>
{context}
</context>

Domanda:
{question}

Risposta:
""".strip()


def call_mock_llm(prompt: str) -> str:
    return (
        "MOCK RESPONSE\n\n"
        "La RAG bonus ha costruito un prompt con più fonti eterogenee. "
        "In modalità reale questo prompt verrebbe inviato a Gemini su Vertex AI.\n\n"
        "Compito dello studente: controllare se le fonti recuperate sono adeguate "
        "alla domanda e annotare nel log eventuali errori di retrieval.\n\n"
        "Fonti usate: vedere SOURCE nel prompt."
    )


def call_vertex_gemini(prompt: str) -> str:
    from google.oauth2 import service_account
    import vertexai
    from vertexai.generative_models import GenerativeModel

    service_account_file = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "./service_account.json")
    project = os.getenv("GOOGLE_CLOUD_PROJECT", "hclsw-gcp-wrkld-auto")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-east1")
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    service_account_path = Path(service_account_file)
    if not service_account_path.is_absolute():
        service_account_path = BASE_DIR / service_account_path

    credentials = service_account.Credentials.from_service_account_file(
        str(service_account_path)
    )

    vertexai.init(
        project=project,
        location=location,
        credentials=credentials,
    )

    model = GenerativeModel(model_name)
    response = model.generate_content(prompt)

    return getattr(response, "text", str(response))


def call_llm(prompt: str) -> str:
    if LLM_MODE == "mock":
        return call_mock_llm(prompt)

    if LLM_MODE == "vertex":
        return call_vertex_gemini(prompt)

    raise ValueError("LLM_MODE deve essere mock oppure vertex")


def retrieve_balanced(question: str, domains: list[str], k_per_domain: int) -> list[dict]:
    """
    Retrieval bilanciato: recupera k_per_domain chunk da ciascun dominio
    specificato, poi li unisce in un'unica lista.

    Risolve il problema delle query comparative (es. "Confronta NIST e OWASP"):
    con retrieval standard il vettore della query cade più vicino a un solo
    dominio e l'altro è sotto-rappresentato nel contesto.

    Strategia: k query separate con filtro per dominio, risultati concatenati.
    I chunk vengono ordinati per distanza all'interno di ogni gruppo, ma i
    gruppi stessi alternano le fonti in modo che l'LLM le veda entrambe.
    """
    all_rows: list[dict] = []
    for domain in domains:
        rows = retrieve(question, domain=domain, k=k_per_domain)
        for row in rows:
            # Segna da quale sotto-query proviene, utile per debug.
            row["retrieved_via_domain"] = domain
        all_rows.extend(rows)
    return all_rows


def ask(question: str, domain: str | None, k: int, log: bool) -> str:
    rows = retrieve(question, domain=domain, k=k)
    prompt = build_prompt(question, rows)
    answer = call_llm(prompt)

    if log:
        append_log(question, domain, k, rows, answer)

    return answer


def ask_compare(question: str, domains: list[str], k_per_domain: int, log: bool) -> str:
    """
    Versione compare-aware di ask():
    usa retrieve_balanced per garantire k chunk per ogni dominio.
    """
    rows = retrieve_balanced(question, domains=domains, k_per_domain=k_per_domain)
    prompt = build_prompt(question, rows)
    answer = call_llm(prompt)

    if log:
        domain_label = "+".join(domains)
        append_log(question, domain_label, k_per_domain * len(domains), rows, answer)

    return answer


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

def append_log(question: str, domain: str | None, k: int, rows: list[dict], answer: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    sources = []
    for row in rows:
        meta = row["metadata"]
        sources.append(
            f"- {meta.get('title')} | domain={meta.get('domain')} | "
            f"chunk={meta.get('chunk_index')} | distance={row['distance']:.4f}"
        )

    entry = f"""
## {timestamp}

### Question
{question}

### Retrieval settings
- domain: {domain or "none"}
- top_k: {k}

### Retrieved sources
{chr(10).join(sources)}

### Answer
{answer}

### Human evaluation
- corretto/parziale/sbagliato:
- il retrieval era pertinente?
- mancava una fonte importante?
- serviva un filtro metadata?
- note:

---
""".strip()

    with QA_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(entry + "\n\n")

    print(f"Log scritto in: {QA_LOG_PATH}")


# ---------------------------------------------------------------------
# Batch test
# ---------------------------------------------------------------------

BONUS_QUESTIONS = [
    ("governance", "Quali sono le funzioni principali del NIST AI RMF e come possono essere usate per governare una applicazione GenAI?"),
    ("security", "Che cos'è il prompt injection secondo OWASP e perché è rilevante per una RAG?"),
    ("responsible_ai", "Quali limiti dei modelli generativi vengono evidenziati nella documentazione Vertex AI Responsible AI?"),
    ("reliability", "Che relazione c'è tra SLO, monitoring e incident management in un servizio enterprise?"),
    ("all", "Confronta NIST AI RMF e OWASP LLM Top 10: uno è più orientato alla governance o alla sicurezza applicativa?"),
    ("all", "Quali controlli minimi proporresti per un agent GenAI aziendale che usa documenti interni e tool operativi?"),
    ("all", "Quali rischi emergono se un agent ha troppa autonomia senza human-in-the-loop?"),
    ("all", "Quali fonti parlano di allucinazioni, sicurezza o affidabilità dei sistemi AI?"),
]


def batch_test() -> None:
    for domain, question in BONUS_QUESTIONS:
        print("\n" + "=" * 100)
        print(f"DOMAIN  : {domain}")
        print(f"QUESTION: {question}")
        print("=" * 100)

        answer_text = ask(
            question=question,
            domain=domain,
            k=TOP_K,
            log=True,
        )

        print(answer_text)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bonus Enterprise AI Governance RAG")

    sub = parser.add_subparsers(required=True)

    p_ingest = sub.add_parser("ingest")
    p_ingest.set_defaults(func=lambda args: ingest())

    p_retrieve = sub.add_parser("retrieve")
    p_retrieve.add_argument("question")
    p_retrieve.add_argument("--domain", default="all")
    p_retrieve.add_argument("--k", type=int, default=TOP_K)

    def retrieve_cmd(args):
        rows = retrieve(args.question, domain=args.domain, k=args.k)
        print_results(rows)

    p_retrieve.set_defaults(func=retrieve_cmd)

    p_ask = sub.add_parser("ask")
    p_ask.add_argument("question")
    p_ask.add_argument("--domain", default="all")
    p_ask.add_argument("--k", type=int, default=TOP_K)
    p_ask.add_argument("--log", action="store_true")

    def ask_cmd(args):
        rows = retrieve(args.question, domain=args.domain, k=args.k)

        print("\nRETRIEVED SOURCES")
        print_results(rows)

        print("\nANSWER")
        answer_text = ask(
            question=args.question,
            domain=args.domain,
            k=args.k,
            log=args.log,
        )
        print(answer_text)

    p_ask.set_defaults(func=ask_cmd)

    p_batch = sub.add_parser("batch-test")
    p_batch.set_defaults(func=lambda args: batch_test())

    # --- ask-compare: retrieval bilanciato per dominio ---
    # Esempio:
    #   python bonus_rag.py ask-compare "Compare NIST AI RMF and OWASP" \
    #       --domains governance security --k-per-domain 3 --log
    p_compare = sub.add_parser(
        "ask-compare",
        help="Retrieval bilanciato: k chunk per ogni dominio specificato.",
    )
    p_compare.add_argument("question")
    p_compare.add_argument(
        "--domains",
        nargs="+",
        default=["governance", "security"],
        help="Lista di domini da cui recuperare chunk in modo bilanciato.",
    )
    p_compare.add_argument("--k-per-domain", type=int, default=3)
    p_compare.add_argument("--log", action="store_true")

    def ask_compare_cmd(args):
        rows = retrieve_balanced(
            args.question,
            domains=args.domains,
            k_per_domain=args.k_per_domain,
        )

        print(f"\nRETRIEVED SOURCES (bilanciato: {args.k_per_domain} per dominio)")
        print_results(rows)

        print("\nANSWER")
        answer_text = ask_compare(
            question=args.question,
            domains=args.domains,
            k_per_domain=args.k_per_domain,
            log=args.log,
        )
        print(answer_text)

    p_compare.set_defaults(func=ask_compare_cmd)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()