"""HTTP service for the routing engine.

    uvicorn api.main:app --port 8000
    ROUTER_SOURCE=csv uvicorn api.main:app --port 8000     # no database needed

Interactive docs: http://localhost:8000/docs
"""
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from pipeline.bus import ConsumerThread, make_bus

from .events import (
    DECISIONS_TOPIC, OUTCOMES_TOPIC, apply_outcome_event, decision_event, outcome_event,
)
from .health import utc_now
from .service import NoGatewayAvailable, Router

METHODS = ("upi", "card", "netbanking")


class RouteRequest(BaseModel):
    amount_inr: float = Field(gt=0, le=1_000_000, examples=[2500])
    method: str = Field(examples=["card"])
    bank: Optional[str] = Field(default=None, examples=["HDFC"])
    merchant_category: Optional[str] = Field(default=None, examples=["retail"])
    txn_id: Optional[str] = None
    exclude: list[str] = Field(default_factory=list, description="gateways already tried (failover retry)")


class OutcomeRequest(BaseModel):
    gateway_id: str
    success: bool
    latency_ms: int = Field(ge=0, le=120_000)
    failure_code: Optional[str] = Field(default=None, examples=["GATEWAY_ERROR"])


log = logging.getLogger("router.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.router = Router.from_artifacts(
        source=os.getenv("ROUTER_SOURCE", "db"), data_dir=os.getenv("ROUTER_DATA", "data"),
        model_path=os.getenv("ROUTER_MODEL", "ml/artifacts/model.joblib"))
    app.state.started = time.time()
    app.state.instance = uuid.uuid4().hex[:8]
    app.state.bus = make_bus()
    app.state.consumer = None
    if app.state.bus.name == "kafka":  # outcomes reported by other services arrive here
        app.state.consumer = ConsumerThread(
            OUTCOMES_TOPIC, f"router-api-{app.state.instance}",
            lambda ev: apply_outcome_event(app.state.router, ev, app.state.instance))
        app.state.consumer.start()
    log.info("health store: %s, event bus: %s", app.state.router.health_backend, app.state.bus.name)
    yield
    if app.state.consumer:
        app.state.consumer.stop()
    app.state.bus.flush(3)


app = FastAPI(title="Payment Routing Engine", version="0.1.0", lifespan=lifespan)


def _router() -> Router:
    return app.state.router


@app.get("/health")
def health():
    r = _router()
    return {"status": "ok", "model_version": r.bundle["version"], "model_kind": r.bundle["kind"],
            "gateways": r.gateway_ids, "disabled": sorted(r.tracker.disabled),
            "uptime_s": round(time.time() - app.state.started),
            "breaker_cooldown_s": r.breakers.cfg.cooldown_s,
            "health_store": r.health_backend, "event_bus": app.state.bus.name}


@app.get("/gateways")
def gateways():
    r = _router()
    recent = r.tracker.summary()
    breaker = r.breakers.snapshot(utc_now())
    return [{"gateway_id": g, "enabled": g not in r.tracker.disabled, "last_15_min": recent.get(g),
             "breaker": breaker.get(g, {"state": "closed", "probe_successes": 0})["state"]}
            for g in r.gateway_ids]


@app.post("/route")
def route(req: RouteRequest):
    if req.method not in METHODS:
        raise HTTPException(422, f"method must be one of {METHODS}")
    try:
        txn = req.model_dump()
        decision = _router().route(txn, exclude=req.exclude)
    except NoGatewayAvailable as exc:
        raise HTTPException(503, str(exc))
    app.state.bus.publish(DECISIONS_TOPIC, decision["txn_id"], decision_event(decision, txn, req.exclude))
    return decision


@app.post("/outcome", status_code=202)
def outcome(req: OutcomeRequest):
    try:
        change = _router().record_outcome(req.gateway_id, req.success, req.latency_ms, req.failure_code)
    except KeyError:
        raise HTTPException(404, f"unknown gateway {req.gateway_id}")
    app.state.bus.publish(OUTCOMES_TOPIC, req.gateway_id, outcome_event(
        req.gateway_id, req.success, req.latency_ms, req.failure_code, origin=app.state.instance))
    return {"recorded": True, "breaker_change": change}


@app.post("/gateways/{gateway_id}/disable")
def disable(gateway_id: str):
    r = _router()
    if gateway_id not in r.gateway_ids:
        raise HTTPException(404, f"unknown gateway {gateway_id}")
    r.tracker.disabled.add(gateway_id)
    return {"gateway_id": gateway_id, "enabled": False}


@app.post("/gateways/{gateway_id}/enable")
def enable(gateway_id: str):
    r = _router()
    r.tracker.disabled.discard(gateway_id)
    return {"gateway_id": gateway_id, "enabled": True}
