-- Denormalized view used for analysis, feature engineering and dashboards.
CREATE OR REPLACE VIEW v_attempt_facts AS
SELECT
    a.txn_id,
    a.attempt,
    t.created_at AS ts,
    t.amount_inr,
    t.method,
    t.bank,
    t.merchant_category,
    a.gateway_id,
    a.success,
    a.latency_ms,
    a.failure_code,
    a.fee_inr,
    CASE WHEN a.success THEN a.fee_inr ELSE 0 END AS fee_charged_inr
FROM payment_attempts AS a
JOIN transactions AS t USING (txn_id);

-- Same rows minus the simulator's known incident windows (sim_outages). For exploratory
-- analysis only, so that outages do not hide the steady-state patterns. Not a feature source.
CREATE OR REPLACE VIEW v_attempt_facts_no_incidents AS
SELECT f.*
FROM v_attempt_facts AS f
WHERE NOT EXISTS (
    SELECT 1 FROM sim_outages AS o
    WHERE o.gateway_id = f.gateway_id AND f.ts >= o.start_ts AND f.ts < o.end_ts
);
