# Deploy the demo on Render

This deploys the existing FastAPI API with interactive documentation at `/docs`.
The repository currently has no dashboard application; `dashboards/theme.py`
contains theme definitions only.

1. Push `render.yaml`, `requirements-deploy.txt`, and `scripts/build_render.sh`
   to your GitHub repository.
2. Sign in at https://dashboard.render.com and select **New > Blueprint**.
3. Connect `kanishk6524/payment-routing-engine`, select the branch containing
   `render.yaml`, review the **Free** service, and deploy.
4. Wait for the build and health check to succeed. Open the service's assigned
   `https://<service-name>.onrender.com/docs` URL to try `POST /route`.

The build installs the API and ML dependencies, generates 300,000 simulated
transactions, and trains XGBoost. It then retains only the latest 75 minutes of
history for startup to reduce runtime memory. Build artifacts include the model
and CSVs, so they do not need to be committed to Git. Each deployment retrains
the model; the existing README's evaluation numbers describe its original run.

If creating a Web Service manually instead of using a Blueprint, use:

| Setting | Value |
| --- | --- |
| Runtime | Python |
| Instance type | Free |
| Build command | `bash scripts/build_render.sh` |
| Start command | `python -m uvicorn api.main:app --host 0.0.0.0 --port $PORT --workers 1` |
| Health check | `/health` |
| `PYTHON_VERSION` | `3.11.11` |
| `ROUTER_SOURCE` | `csv` |
| `ROUTER_HEALTH` | `memory` |
| `EVENT_BUS` | `none` |
| `OMP_NUM_THREADS` | `1` |
| `OPENBLAS_NUM_THREADS` | `1` |

Verify the live API (replace the hostname with the one Render assigns):

```bash
curl https://<service-name>.onrender.com/health
curl https://<service-name>.onrender.com/route \
  -H 'Content-Type: application/json' \
  -d '{"amount_inr":2500,"method":"card","bank":"HDFC","merchant_category":"retail"}'
```

This is a public simulation demo. Outcomes, gateway switches, and circuit-breaker
state are stored in one process and reset on restart. PostgreSQL, Redis, Kafka,
and Grafana are not provisioned. The gateway control endpoints are public, so
other visitors can change the demo's gateway state. Use authentication and
persistent backing services before adapting it for real payments.

Render's free services sleep after 15 minutes of inactivity and can take about
a minute to wake. See the current [free plan limits](https://render.com/docs/free)
and [FastAPI deployment guide](https://render.com/docs/deploy-fastapi).
