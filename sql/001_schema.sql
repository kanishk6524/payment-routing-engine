-- Stage 3: schema for the payment routing engine.
-- Idempotent: safe to run repeatedly (CREATE ... IF NOT EXISTS).
--
-- Design (3NF):
--   lookup tables      payment_methods, banks, merchant_categories
--   gateways           one row per gateway; gateway_methods = which methods it supports
--   transactions       the payment request (what the router sees)
--   payment_attempts   one row per attempt through a gateway; retries and failover are
--                      extra attempts for the same txn_id (txn_id is the idempotency key)
--   routing_decisions  what the router chose and why (scores + SHAP explanation as JSON)
--   gateway_metrics    time-bucketed gateway stats for dashboards
--   sim_outages        simulator ground truth, for evaluation only

-- ---------------------------------------------------------------- lookups
CREATE TABLE IF NOT EXISTS payment_methods (
    method text PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS banks (
    bank text PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS merchant_categories (
    category text PRIMARY KEY
);

-- ---------------------------------------------------------------- gateways
CREATE TABLE IF NOT EXISTS gateways (
    gateway_id text PRIMARY KEY,
    name       text         NOT NULL,
    fee_pct    numeric(6,4) NOT NULL CHECK (fee_pct >= 0),            -- 0.0200 = 2%
    fee_fixed  numeric(8,2) NOT NULL DEFAULT 0 CHECK (fee_fixed >= 0) -- flat INR per txn
);

CREATE TABLE IF NOT EXISTS gateway_methods (
    gateway_id text NOT NULL REFERENCES gateways (gateway_id),
    method     text NOT NULL REFERENCES payment_methods (method),
    PRIMARY KEY (gateway_id, method)
);

-- ---------------------------------------------------------------- transactions
CREATE TABLE IF NOT EXISTS transactions (
    txn_id            text PRIMARY KEY,
    created_at        timestamp     NOT NULL,   -- simulated local time, no time zone
    amount_inr        numeric(12,2) NOT NULL CHECK (amount_inr > 0),
    method            text          NOT NULL REFERENCES payment_methods (method),
    bank              text          NOT NULL REFERENCES banks (bank),
    merchant_category text          NOT NULL REFERENCES merchant_categories (category)
);

CREATE TABLE IF NOT EXISTS payment_attempts (
    txn_id       text        NOT NULL REFERENCES transactions (txn_id),
    attempt      smallint    NOT NULL DEFAULT 1 CHECK (attempt >= 1),
    gateway_id   text        NOT NULL REFERENCES gateways (gateway_id),
    attempted_at timestamp   NOT NULL,
    success      boolean     NOT NULL,
    latency_ms   integer     NOT NULL CHECK (latency_ms >= 0),
    failure_code text        CHECK (failure_code IN ('BANK_DECLINE', 'GATEWAY_ERROR', 'TIMEOUT')),
    fee_inr      numeric(10,2) NOT NULL CHECK (fee_inr >= 0),  -- fee if the attempt succeeds
    PRIMARY KEY (txn_id, attempt),
    CHECK (success = (failure_code IS NULL))
);

-- ---------------------------------------------------------------- routing
CREATE TABLE IF NOT EXISTS routing_decisions (
    decision_id    bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    txn_id         text      NOT NULL REFERENCES transactions (txn_id),
    attempt        smallint  NOT NULL DEFAULT 1 CHECK (attempt >= 1),
    decided_at     timestamp NOT NULL DEFAULT now(),
    policy         text      NOT NULL,          -- e.g. 'xgboost', 'baseline_cheapest'
    model_version  text,
    chosen_gateway text      NOT NULL REFERENCES gateways (gateway_id),
    scores         jsonb     NOT NULL DEFAULT '{}'::jsonb,       -- {"gw_a": 0.91, ...}
    explanation    jsonb     NOT NULL DEFAULT '{}'::jsonb,       -- top SHAP factors
    UNIQUE (txn_id, attempt, policy)
);

CREATE TABLE IF NOT EXISTS gateway_metrics (
    gateway_id     text      NOT NULL REFERENCES gateways (gateway_id),
    bucket_start   timestamp NOT NULL,
    bucket_minutes smallint  NOT NULL DEFAULT 5 CHECK (bucket_minutes > 0),
    attempts       integer   NOT NULL CHECK (attempts >= 0),
    successes      integer   NOT NULL CHECK (successes >= 0),
    avg_latency_ms numeric(10,2),
    p95_latency_ms numeric(10,2),
    fees_inr       numeric(14,2) NOT NULL DEFAULT 0,             -- fees on successful attempts
    PRIMARY KEY (gateway_id, bucket_minutes, bucket_start),
    CHECK (successes <= attempts)
);

-- ---------------------------------------------------------------- simulator truth
CREATE TABLE IF NOT EXISTS sim_outages (
    outage_id    smallint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    gateway_id   text          NOT NULL REFERENCES gateways (gateway_id),
    start_ts     timestamp     NOT NULL,
    end_ts       timestamp     NOT NULL,
    success_prob numeric(4,3)  NOT NULL CHECK (success_prob BETWEEN 0 AND 1),
    kind         text          NOT NULL CHECK (kind IN ('outage', 'degradation')),
    CHECK (end_ts > start_ts),
    UNIQUE (gateway_id, start_ts)
);
COMMENT ON TABLE sim_outages IS
    'Simulator ground truth for evaluation only. Never use as a model feature.';

-- ---------------------------------------------------------------- indexes
CREATE INDEX IF NOT EXISTS idx_transactions_created_at   ON transactions (created_at);
CREATE INDEX IF NOT EXISTS idx_attempts_gateway_time     ON payment_attempts (gateway_id, attempted_at);
CREATE INDEX IF NOT EXISTS idx_attempts_time             ON payment_attempts (attempted_at);
CREATE INDEX IF NOT EXISTS idx_decisions_gateway_time    ON routing_decisions (chosen_gateway, decided_at);
CREATE INDEX IF NOT EXISTS idx_gateway_metrics_time      ON gateway_metrics (bucket_start);
