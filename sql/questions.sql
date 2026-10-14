-- business questions for the retention team.
-- each block has a name (the output file) and the question it answers,
-- so the dashboard can show the exact query next to its result.
-- standard SQL, tested in DuckDB; runs on Postgres, Snowflake or BigQuery
-- with little or no change


-- name: churn_by_contract
-- question: How does churn differ by contract type, and how much monthly revenue walked out?
SELECT
    a.Contract                                          AS contract,
    COUNT(*)                                            AS customers,
    ROUND(AVG(a.churn) * 100, 1)                        AS churn_rate_pct,
    ROUND(SUM(CASE WHEN a.churn = 1 THEN a.MonthlyCharges ELSE 0 END), 0)
                                                        AS monthly_revenue_lost
FROM accounts a
GROUP BY a.Contract
ORDER BY churn_rate_pct DESC;


-- name: churn_by_tenure
-- question: When in the customer lifecycle do people leave?
SELECT
    a.tenure_group,
    COUNT(*)                         AS customers,
    ROUND(AVG(a.churn) * 100, 1)     AS churn_rate_pct
FROM accounts a
GROUP BY a.tenure_group
-- sort by the shortest tenure in each group so the buckets stay in time order
ORDER BY MIN(a.tenure);


-- name: churn_by_payment
-- question: Does how a customer pays predict whether they stay?
SELECT
    a.PaymentMethod                  AS payment_method,
    a.auto_pay,
    COUNT(*)                         AS customers,
    ROUND(AVG(a.churn) * 100, 1)     AS churn_rate_pct
FROM accounts a
GROUP BY a.PaymentMethod, a.auto_pay
ORDER BY churn_rate_pct DESC;


-- name: fiber_tech_support
-- question: Are fiber customers without tech support a problem group?
SELECT
    s.InternetService                AS internet,
    s.TechSupport                    AS tech_support,
    COUNT(*)                         AS customers,
    ROUND(AVG(a.churn) * 100, 1)     AS churn_rate_pct,
    ROUND(AVG(a.MonthlyCharges), 2)  AS avg_monthly_bill
FROM services s
JOIN accounts a ON a.customerID = s.customerID
WHERE s.InternetService <> 'No'
GROUP BY s.InternetService, s.TechSupport
ORDER BY churn_rate_pct DESC;


-- name: add_ons_vs_churn
-- question: Do customers with more add-on services stay longer?
SELECT
    s.num_add_ons,
    COUNT(*)                         AS customers,
    ROUND(AVG(a.churn) * 100, 1)     AS churn_rate_pct
FROM services s
JOIN accounts a ON a.customerID = s.customerID
-- add-ons need internet, so leave out phone-only customers
WHERE s.InternetService <> 'No'
GROUP BY s.num_add_ons
ORDER BY s.num_add_ons;


-- name: seniors_and_households
-- question: Which household types are most likely to leave?
SELECT
    c.SeniorCitizen                  AS senior,
    c.Partner                        AS has_partner,
    c.Dependents                     AS has_dependents,
    COUNT(*)                         AS customers,
    ROUND(AVG(a.churn) * 100, 1)     AS churn_rate_pct
FROM customers c
JOIN accounts a ON a.customerID = c.customerID
GROUP BY c.SeniorCitizen, c.Partner, c.Dependents
HAVING COUNT(*) >= 100
ORDER BY churn_rate_pct DESC;


-- name: score_check
-- question: On customers the model never trained on, do the risk tiers really separate leavers?
SELECT
    cs.risk_tier,
    COUNT(*)                         AS held_out_customers,
    ROUND(AVG(a.churn) * 100, 1)     AS actual_churn_pct,
    ROUND(AVG(cs.churn_score), 3)    AS avg_predicted_score
FROM churn_scores cs
JOIN accounts a ON a.customerID = cs.customerID
WHERE cs.in_test = 1
GROUP BY cs.risk_tier
ORDER BY avg_predicted_score DESC;


-- name: revenue_at_risk
-- question: Among customers who are still active, how much monthly revenue sits in each risk tier?
-- the raw scores run high on purpose (the model is weighted to catch leavers),
-- so they'd overstate the loss. instead each tier is priced at the churn rate
-- it actually showed on held-out customers
WITH tier_rate AS (
    SELECT cs.risk_tier, AVG(a.churn) AS held_out_churn_rate
    FROM churn_scores cs
    JOIN accounts a ON a.customerID = cs.customerID
    WHERE cs.in_test = 1
    GROUP BY cs.risk_tier
)
SELECT
    cs.risk_tier,
    COUNT(*)                                        AS active_customers,
    ROUND(SUM(a.MonthlyCharges), 0)                 AS monthly_revenue,
    ROUND(MAX(t.held_out_churn_rate) * 100, 1)      AS tier_churn_rate_pct,
    ROUND(SUM(a.MonthlyCharges) * MAX(t.held_out_churn_rate), 0) AS expected_monthly_loss
FROM churn_scores cs
JOIN accounts a  ON a.customerID = cs.customerID
JOIN tier_rate t ON t.risk_tier = cs.risk_tier
WHERE a.churn = 0  -- churned customers are already gone, nothing left to save
GROUP BY cs.risk_tier
ORDER BY expected_monthly_loss DESC;


-- name: reasons_on_call_list
-- question: Why are active high-risk customers at risk, and what should we offer them?
SELECT
    cs.main_risk,
    cs.suggested_action,
    COUNT(*)                         AS customers,
    ROUND(SUM(a.MonthlyCharges), 0)  AS monthly_revenue
FROM churn_scores cs
JOIN accounts a ON a.customerID = cs.customerID
WHERE a.churn = 0 AND cs.risk_tier = 'High'
GROUP BY cs.main_risk, cs.suggested_action
ORDER BY customers DESC;


-- name: call_list_top_25
-- question: Who should the retention team call first this week?
SELECT
    a.customerID,
    cs.churn_score,
    a.tenure                         AS months_as_customer,
    a.Contract                       AS contract,
    ROUND(a.MonthlyCharges, 2)       AS monthly_bill,
    cs.main_risk,
    cs.suggested_action
FROM churn_scores cs
JOIN accounts a ON a.customerID = cs.customerID
WHERE a.churn = 0 AND cs.risk_tier = 'High'
-- highest expected loss first: a likely leaver on a big bill beats a likely leaver on a small one
ORDER BY a.MonthlyCharges * cs.churn_score DESC
LIMIT 25;
