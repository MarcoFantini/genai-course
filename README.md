# Progetto FastAPI + Uvicorn

Questo repository contiene un'applicazione FastAPI e la configurazione per avviare Uvicorn insieme a una semplice applicazione HTML.

## Requisiti

- Python 3.10+ (o versione compatibile)
- FastAPI
- Uvicorn

Installa i pacchetti necessari:

```bash
pip install fastapi uvicorn
```

## Avviare l'app FastAPI

Per eseguire l'app FastAPI con Uvicorn, usa il comando:

```bash
uvicorn main:app --reload
```

Dove `main` è il nome del file Python che contiene l'istanza FastAPI e `app` è il nome dell'oggetto FastAPI.

Con `--reload`, l'app viene riavviata automaticamente a ogni modifica del codice.

## Applicazione HTML

Se l'applicazione include una pagina HTML, potrebbe essere servita come risposta da FastAPI. Ad esempio, un endpoint potrebbe restituire un file HTML o un template.

### Esempio di endpoint HTML

```python
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>Applicazione FastAPI</title>
        </head>
        <body>
            <h1>Benvenuto in FastAPI</h1>
            <p>Questa è la pagina HTML servita dall'app.</p>
        </body>
    </html>
    """
```

## Esempio di avvio completo

1. Crea un file `main.py` con l'app FastAPI.
2. Esegui il comando:

```bash
uvicorn main:app --reload
```

3. Apri il browser e visita `http://127.0.0.1:8000/`.

## Note

- Se il nome del file o dell'istanza FastAPI cambia, aggiorna il comando Uvicorn di conseguenza.
- Per un'app in produzione, rimuovi `--reload` e utilizza `uvicorn day2_agents.api:app --reload --port 8000`.
