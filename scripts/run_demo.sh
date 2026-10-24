#!/usr/bin/env bash
# Start the API in the background, run the incident demo against it, then stop the API.
#   bash scripts/run_demo.sh
set -u
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate

uvicorn api.main:app --port 8000 --log-level warning &
SERVER=$!
trap 'kill $SERVER 2>/dev/null' EXIT

for _ in $(seq 1 60); do
  curl -sf http://localhost:8000/health >/dev/null && break
  sleep 0.5
done
if ! curl -sf http://localhost:8000/health >/dev/null; then
  echo "The API did not start. Run: uvicorn api.main:app --port 8000   and read the error."
  exit 1
fi

echo "--- one routing decision ---"
curl -s localhost:8000/route -H 'content-type: application/json' \
  -d '{"amount_inr": 2500, "method": "card", "bank": "HDFC", "merchant_category": "retail"}' \
  | python -m json.tool
echo
echo "--- incident demo ---"
python scripts/demo_live.py
