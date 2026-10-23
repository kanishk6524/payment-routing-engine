-- Loads simulator output from staging tables into the normalized schema.
-- The staging tables (stg_transactions, stg_gateways, stg_outages) are temporary and
-- are filled by scripts/load_data.py. Every INSERT is idempotent, so re-running is safe.

-- lookups, taken from the data itself
INSERT INTO payment_methods (method)
SELECT DISTINCT method FROM stg_transactions
UNION
SELECT DISTINCT m FROM stg_gateways, unnest(string_to_array(supported_methods, '|')) AS m
ON CONFLICT DO NOTHING;

INSERT INTO banks (bank)
SELECT DISTINCT bank FROM stg_transactions
ON CONFLICT DO NOTHING;

INSERT INTO merchant_categories (category)
SELECT DISTINCT merchant_category FROM stg_transactions
ON CONFLICT DO NOTHING;

-- gateways
INSERT INTO gateways (gateway_id, name, fee_pct, fee_fixed)
SELECT gateway_id, name, fee_pct, fee_fixed FROM stg_gateways
ON CONFLICT (gateway_id) DO UPDATE
    SET name = EXCLUDED.name,
        fee_pct = EXCLUDED.fee_pct,
        fee_fixed = EXCLUDED.fee_fixed;

INSERT INTO gateway_methods (gateway_id, method)
SELECT g.gateway_id, m
FROM stg_gateways AS g, unnest(string_to_array(g.supported_methods, '|')) AS m
ON CONFLICT DO NOTHING;

INSERT INTO sim_outages (gateway_id, start_ts, end_ts, success_prob, kind)
SELECT gateway_id, start_ts, end_ts, success_prob, kind FROM stg_outages
ON CONFLICT (gateway_id, start_ts) DO NOTHING;

-- transactions and their attempts
INSERT INTO transactions (txn_id, created_at, amount_inr, method, bank, merchant_category)
SELECT txn_id, ts, amount_inr, method, bank, merchant_category FROM stg_transactions
ON CONFLICT (txn_id) DO NOTHING;

INSERT INTO payment_attempts
    (txn_id, attempt, gateway_id, attempted_at, success, latency_ms, failure_code, fee_inr)
SELECT txn_id, attempt, gateway_id, ts, success, latency_ms, failure_code, fee_inr
FROM stg_transactions
ON CONFLICT (txn_id, attempt) DO NOTHING;
