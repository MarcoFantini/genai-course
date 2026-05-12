#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

# prepare agents
python day2_agents/itsm_agent.py setup-rag-data --ingest
python day2_agents/itsm_agent.py examples
# serve api.py 
uvicorn day2_agents.api:app --reload --port 8000

open ./index.html
