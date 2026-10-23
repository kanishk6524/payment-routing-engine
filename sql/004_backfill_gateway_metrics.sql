-- Backfill 15-minute gateway metrics from historical attempts (needs PostgreSQL 14+ for date_bin).
-- In production the Kafka consumer (Stage 8) keeps this table up to date.
INSERT INTO gateway_metrics
    (gateway_id, bucket_start, bucket_minutes, attempts, successes,
     avg_latency_ms, p95_latency_ms, fees_inr)
SELECT
    gateway_id,
    date_bin(INTERVAL '15 minutes', attempted_at, TIMESTAMP '2000-01-01') AS bucket_start,
    15,
    count(*),
    count(*) FILTER (WHERE success),
    round(avg(latency_ms)::numeric, 2),
    round((percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms))::numeric, 2),
    COALESCE(sum(fee_inr) FILTER (WHERE success), 0)
FROM payment_attempts
GROUP BY gateway_id, date_bin(INTERVAL '15 minutes', attempted_at, TIMESTAMP '2000-01-01')
ON CONFLICT (gateway_id, bucket_minutes, bucket_start) DO UPDATE
    SET attempts       = EXCLUDED.attempts,
        successes      = EXCLUDED.successes,
        avg_latency_ms = EXCLUDED.avg_latency_ms,
        p95_latency_ms = EXCLUDED.p95_latency_ms,
        fees_inr       = EXCLUDED.fees_inr;
