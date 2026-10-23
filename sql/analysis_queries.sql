-- Stage 4 exploratory analysis. Each query is introduced by a "-- name:" line.
-- Run them all:
--   docker compose exec -T postgres psql -U router -d payment_router < sql/analysis_queries.sql
--
-- Segment queries use v_attempt_facts_no_incidents so that the four simulated incidents do not
-- mask steady-state patterns. Amount buckets match ml.features.AMOUNT_BINS:
-- 0:<100  1:100-500  2:500-1k  3:1k-2.5k  4:2.5k-5k  5:5k-10k  6:10k-25k  7:25k-50k  8:50k+

-- name: overview_by_gateway
SELECT gateway_id,
       count(*)                                                          AS attempts,
       round(avg(success::int)::numeric, 4)                              AS success_rate,
       round(avg(latency_ms))                                            AS avg_latency_ms,
       round((percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms))::numeric) AS p95_latency_ms,
       round(avg(fee_inr), 2)                                            AS avg_fee_inr
FROM v_attempt_facts
GROUP BY gateway_id
ORDER BY gateway_id;

-- name: failure_codes_by_gateway
SELECT gateway_id, failure_code, count(*) AS failures,
       round(100.0 * count(*) / sum(count(*)) OVER (PARTITION BY gateway_id), 1) AS pct_of_gateway_failures
FROM v_attempt_facts
WHERE NOT success
GROUP BY gateway_id, failure_code
ORDER BY gateway_id, failures DESC;

-- name: success_by_gateway_method
SELECT gateway_id, method, count(*) AS attempts,
       round(avg(success::int)::numeric, 4) AS success_rate
FROM v_attempt_facts_no_incidents
GROUP BY gateway_id, method
ORDER BY gateway_id, method;

-- name: success_by_gateway_hour
SELECT gateway_id, extract(hour FROM ts)::int AS hour, count(*) AS attempts,
       round(avg(success::int)::numeric, 4) AS success_rate
FROM v_attempt_facts_no_incidents
GROUP BY gateway_id, extract(hour FROM ts)::int
ORDER BY gateway_id, hour;

-- name: success_by_gateway_amount_bucket
SELECT gateway_id,
       width_bucket(amount_inr, ARRAY[100, 500, 1000, 2500, 5000, 10000, 25000, 50000]::numeric[]) AS amount_bucket,
       count(*) AS attempts,
       round(avg(success::int)::numeric, 4) AS success_rate
FROM v_attempt_facts_no_incidents
GROUP BY gateway_id, amount_bucket
ORDER BY gateway_id, amount_bucket;

-- name: success_by_gateway_bank
SELECT gateway_id, bank, count(*) AS attempts,
       round(avg(success::int)::numeric, 4) AS success_rate
FROM v_attempt_facts_no_incidents
GROUP BY gateway_id, bank
ORDER BY gateway_id, bank;

-- name: success_by_gateway_category
SELECT gateway_id, merchant_category, count(*) AS attempts,
       round(avg(success::int)::numeric, 4) AS success_rate
FROM v_attempt_facts_no_incidents
GROUP BY gateway_id, merchant_category
ORDER BY gateway_id, merchant_category;

-- name: card_success_by_gateway_and_amount
SELECT gateway_id,
       CASE WHEN amount_inr >= 20000 THEN '20k and above' ELSE 'under 20k' END AS card_amount,
       count(*) AS attempts,
       round(avg(success::int)::numeric, 4) AS success_rate
FROM v_attempt_facts_no_incidents
WHERE method = 'card'
GROUP BY gateway_id, card_amount
ORDER BY gateway_id, card_amount;

-- name: incident_impact
WITH normal AS (
    SELECT gateway_id, avg(success::int) AS rate
    FROM v_attempt_facts_no_incidents
    GROUP BY gateway_id
)
SELECT o.gateway_id, o.kind, o.start_ts, o.end_ts,
       w.attempts                                   AS attempts_in_window,
       round(w.rate::numeric, 4)                    AS success_in_window,
       round(n.rate::numeric, 4)                    AS success_normally
FROM sim_outages AS o
CROSS JOIN LATERAL (
    SELECT count(*) AS attempts, avg(success::int) AS rate
    FROM v_attempt_facts AS f
    WHERE f.gateway_id = o.gateway_id AND f.ts >= o.start_ts AND f.ts < o.end_ts
) AS w
JOIN normal AS n ON n.gateway_id = o.gateway_id
ORDER BY o.start_ts;

-- name: daily_volume_and_success
SELECT ts::date AS day, count(*) AS attempts,
       round(avg(success::int)::numeric, 4) AS success_rate,
       round(sum(fee_charged_inr)) AS fees_charged_inr
FROM v_attempt_facts
GROUP BY ts::date
ORDER BY day;
