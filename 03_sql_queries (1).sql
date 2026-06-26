/* ================================================================================
   CUSTOMER CHURN PREDICTION WITH ML & BI DASHBOARD
   PROJECT 2: SQL QUERIES FOR CHURN ANALYSIS
   ================================================================================ */

-- ==================== CREATE TABLES ====================

CREATE TABLE customers (
    customer_id INT PRIMARY KEY,
    age INT,
    gender VARCHAR(10),
    country VARCHAR(50),
    registration_date DATE,
    monthly_subscription VARCHAR(20),
    subscription_price DECIMAL(6, 2)
);

CREATE TABLE behavior (
    customer_id INT PRIMARY KEY,
    account_age_days INT,
    login_count_last_30 INT,
    login_count_last_90 INT,
    days_since_last_login INT,
    content_views_last_30 INT,
    content_views_last_90 INT,
    support_tickets_count INT,
    support_satisfaction_score DECIMAL(3, 2),
    num_features_used INT,
    monthly_active_days INT,
    avg_session_length_minutes DECIMAL(6, 2),
    profile_completion_percentage INT,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE financial (
    customer_id INT PRIMARY KEY,
    total_months_subscribed INT,
    payments_made INT,
    failed_payment_count INT,
    days_overdue INT,
    total_revenue DECIMAL(10, 2),
    avg_monthly_revenue DECIMAL(10, 2),
    upgrade_count INT,
    downgrade_count INT,
    billing_cycle_day INT,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE churn_labels (
    customer_id INT PRIMARY KEY,
    churn_flag INT,
    churn_probability_score DECIMAL(5, 3),
    likely_churn_date DATE,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE churn_predictions (
    customer_id INT PRIMARY KEY,
    predicted_churn INT,
    churn_probability DECIMAL(5, 4),
    churn_risk_level VARCHAR(20),
    customer_value VARCHAR(20),
    retention_priority VARCHAR(20),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- ==================== ANALYTICAL QUERIES ====================

-- QUERY 1: High-Risk Churn Customers with Business Impact
SELECT 
    c.customer_id,
    c.age,
    c.country,
    c.monthly_subscription,
    f.total_revenue,
    f.avg_monthly_revenue,
    b.days_since_last_login,
    b.login_count_last_30,
    b.content_views_last_30,
    f.failed_payment_count,
    p.churn_probability,
    p.churn_risk_level,
    p.retention_priority,
    CASE 
        WHEN p.churn_probability > 0.8 THEN 'Immediate Action'
        WHEN p.churn_probability > 0.6 THEN 'Urgent Review'
        ELSE 'Monitor'
    END AS action_required
FROM customers c
JOIN behavior b ON c.customer_id = b.customer_id
JOIN financial f ON c.customer_id = f.customer_id
JOIN churn_predictions p ON c.customer_id = p.customer_id
WHERE p.churn_probability > 0.5
ORDER BY f.total_revenue DESC;

-- QUERY 2: Customer Engagement vs Churn Risk
SELECT 
    CASE 
        WHEN b.login_count_last_30 = 0 THEN 'No Login'
        WHEN b.login_count_last_30 <= 5 THEN 'Low Engagement'
        WHEN b.login_count_last_30 <= 15 THEN 'Medium Engagement'
        ELSE 'High Engagement'
    END AS engagement_level,
    COUNT(DISTINCT c.customer_id) AS customer_count,
    ROUND(AVG(p.churn_probability), 4) AS avg_churn_risk,
    COUNT(CASE WHEN p.predicted_churn = 1 THEN 1 END) AS at_risk_count,
    ROUND(100.0 * COUNT(CASE WHEN p.predicted_churn = 1 THEN 1 END) / COUNT(DISTINCT c.customer_id), 2) AS churn_rate_percent,
    ROUND(AVG(f.total_revenue), 2) AS avg_customer_value
FROM customers c
JOIN behavior b ON c.customer_id = b.customer_id
JOIN financial f ON c.customer_id = f.customer_id
JOIN churn_predictions p ON c.customer_id = p.customer_id
GROUP BY 
    CASE 
        WHEN b.login_count_last_30 = 0 THEN 'No Login'
        WHEN b.login_count_last_30 <= 5 THEN 'Low Engagement'
        WHEN b.login_count_last_30 <= 15 THEN 'Medium Engagement'
        ELSE 'High Engagement'
    END
ORDER BY avg_churn_risk DESC;

-- QUERY 3: Payment Health & Churn Correlation
SELECT 
    CASE 
        WHEN f.failed_payment_count = 0 AND f.days_overdue = 0 THEN 'Healthy'
        WHEN f.failed_payment_count <= 1 THEN 'At Risk'
        ELSE 'Critical'
    END AS payment_health,
    COUNT(DISTINCT c.customer_id) AS customer_count,
    ROUND(AVG(p.churn_probability), 4) AS avg_churn_risk,
    COUNT(CASE WHEN p.predicted_churn = 1 THEN 1 END) AS predicted_churners,
    ROUND(AVG(f.total_revenue), 2) AS avg_revenue,
    ROUND(SUM(f.total_revenue), 2) AS total_revenue_at_risk
FROM customers c
JOIN financial f ON c.customer_id = f.customer_id
JOIN churn_predictions p ON c.customer_id = p.customer_id
GROUP BY 
    CASE 
        WHEN f.failed_payment_count = 0 AND f.days_overdue = 0 THEN 'Healthy'
        WHEN f.failed_payment_count <= 1 THEN 'At Risk'
        ELSE 'Critical'
    END
ORDER BY avg_churn_risk DESC;

-- QUERY 4: Subscription Tier Analysis
SELECT 
    c.monthly_subscription,
    COUNT(DISTINCT c.customer_id) AS total_customers,
    ROUND(AVG(p.churn_probability), 4) AS avg_churn_probability,
    COUNT(CASE WHEN p.predicted_churn = 1 THEN 1 END) AS predicted_churners,
    ROUND(100.0 * COUNT(CASE WHEN p.predicted_churn = 1 THEN 1 END) / COUNT(DISTINCT c.customer_id), 2) AS churn_rate_percent,
    ROUND(AVG(f.total_revenue), 2) AS avg_lifetime_value,
    ROUND(SUM(f.total_revenue), 2) AS total_revenue
FROM customers c
JOIN financial f ON c.customer_id = f.customer_id
JOIN churn_predictions p ON c.customer_id = p.customer_id
GROUP BY c.monthly_subscription
ORDER BY churn_rate_percent DESC;

-- QUERY 5: Geographic Churn Analysis
SELECT 
    c.country,
    COUNT(DISTINCT c.customer_id) AS customer_count,
    ROUND(AVG(p.churn_probability), 4) AS avg_churn_risk,
    COUNT(CASE WHEN p.predicted_churn = 1 THEN 1 END) AS at_risk_customers,
    ROUND(100.0 * COUNT(CASE WHEN p.predicted_churn = 1 THEN 1 END) / COUNT(DISTINCT c.customer_id), 2) AS churn_rate,
    ROUND(AVG(f.total_revenue), 2) AS avg_customer_value,
    COUNT(CASE WHEN p.retention_priority = 'CRITICAL' THEN 1 END) AS critical_priority_count
FROM customers c
JOIN financial f ON c.customer_id = f.customer_id
JOIN churn_predictions p ON c.customer_id = p.customer_id
GROUP BY c.country
ORDER BY churn_rate DESC;

-- QUERY 6: Customer Lifetime Value Cohorts
SELECT 
    CASE 
        WHEN f.total_revenue < 500 THEN 'Low Value'
        WHEN f.total_revenue < 1500 THEN 'Medium Value'
        ELSE 'High Value'
    END AS customer_value_segment,
    COUNT(DISTINCT c.customer_id) AS segment_size,
    ROUND(AVG(f.total_revenue), 2) AS avg_lifetime_value,
    ROUND(AVG(p.churn_probability), 4) AS avg_churn_risk,
    COUNT(CASE WHEN p.predicted_churn = 1 THEN 1 END) AS predicted_churners,
    ROUND(100.0 * COUNT(CASE WHEN p.predicted_churn = 1 THEN 1 END) / COUNT(DISTINCT c.customer_id), 2) AS churn_rate,
    ROUND(SUM(f.total_revenue), 2) AS total_revenue_at_risk
FROM customers c
JOIN financial f ON c.customer_id = f.customer_id
JOIN churn_predictions p ON c.customer_id = p.customer_id
GROUP BY 
    CASE 
        WHEN f.total_revenue < 500 THEN 'Low Value'
        WHEN f.total_revenue < 1500 THEN 'Medium Value'
        ELSE 'High Value'
    END
ORDER BY churn_rate DESC;

-- QUERY 7: Support & Satisfaction Analysis
SELECT 
    CASE 
        WHEN b.support_satisfaction_score IS NULL THEN 'No Rating'
        WHEN b.support_satisfaction_score >= 4 THEN 'Satisfied'
        WHEN b.support_satisfaction_score >= 3 THEN 'Neutral'
        ELSE 'Dissatisfied'
    END AS satisfaction_level,
    COUNT(DISTINCT c.customer_id) AS customer_count,
    ROUND(AVG(b.support_tickets_count), 2) AS avg_tickets,
    ROUND(AVG(p.churn_probability), 4) AS avg_churn_risk,
    COUNT(CASE WHEN p.predicted_churn = 1 THEN 1 END) AS at_risk_count,
    ROUND(100.0 * COUNT(CASE WHEN p.predicted_churn = 1 THEN 1 END) / COUNT(DISTINCT c.customer_id), 2) AS churn_rate
FROM customers c
JOIN behavior b ON c.customer_id = b.customer_id
JOIN churn_predictions p ON c.customer_id = p.customer_id
GROUP BY 
    CASE 
        WHEN b.support_satisfaction_score IS NULL THEN 'No Rating'
        WHEN b.support_satisfaction_score >= 4 THEN 'Satisfied'
        WHEN b.support_satisfaction_score >= 3 THEN 'Neutral'
        ELSE 'Dissatisfied'
    END
ORDER BY churn_rate DESC;

-- QUERY 8: Account Age & Churn Pattern
SELECT 
    CASE 
        WHEN f.total_months_subscribed <= 3 THEN '0-3 Months'
        WHEN f.total_months_subscribed <= 6 THEN '4-6 Months'
        WHEN f.total_months_subscribed <= 12 THEN '7-12 Months'
        ELSE '12+ Months'
    END AS account_age_bucket,
    COUNT(DISTINCT c.customer_id) AS customer_count,
    ROUND(AVG(f.total_months_subscribed), 1) AS avg_months,
    ROUND(AVG(p.churn_probability), 4) AS avg_churn_risk,
    COUNT(CASE WHEN p.predicted_churn = 1 THEN 1 END) AS predicted_churners,
    ROUND(100.0 * COUNT(CASE WHEN p.predicted_churn = 1 THEN 1 END) / COUNT(DISTINCT c.customer_id), 2) AS churn_rate
FROM customers c
JOIN financial f ON c.customer_id = f.customer_id
JOIN churn_predictions p ON c.customer_id = p.customer_id
GROUP BY 
    CASE 
        WHEN f.total_months_subscribed <= 3 THEN '0-3 Months'
        WHEN f.total_months_subscribed <= 6 THEN '4-6 Months'
        WHEN f.total_months_subscribed <= 12 THEN '7-12 Months'
        ELSE '12+ Months'
    END
ORDER BY 
    CASE 
        WHEN account_age_bucket = '0-3 Months' THEN 1
        WHEN account_age_bucket = '4-6 Months' THEN 2
        WHEN account_age_bucket = '7-12 Months' THEN 3
        ELSE 4
    END;

-- QUERY 9: Feature Usage Impact
SELECT 
    CASE 
        WHEN b.num_features_used <= 2 THEN 'Low Feature Adoption'
        WHEN b.num_features_used <= 5 THEN 'Medium Feature Adoption'
        ELSE 'High Feature Adoption'
    END AS feature_adoption,
    COUNT(DISTINCT c.customer_id) AS customer_count,
    ROUND(AVG(b.num_features_used), 2) AS avg_features_used,
    ROUND(AVG(p.churn_probability), 4) AS avg_churn_risk,
    COUNT(CASE WHEN p.predicted_churn = 1 THEN 1 END) AS predicted_churners,
    ROUND(100.0 * COUNT(CASE WHEN p.predicted_churn = 1 THEN 1 END) / COUNT(DISTINCT c.customer_id), 2) AS churn_rate,
    ROUND(AVG(f.total_revenue), 2) AS avg_customer_value
FROM customers c
JOIN behavior b ON c.customer_id = b.customer_id
JOIN financial f ON c.customer_id = f.customer_id
JOIN churn_predictions p ON c.customer_id = p.customer_id
GROUP BY 
    CASE 
        WHEN b.num_features_used <= 2 THEN 'Low Feature Adoption'
        WHEN b.num_features_used <= 5 THEN 'Medium Feature Adoption'
        ELSE 'High Feature Adoption'
    END
ORDER BY churn_rate DESC;

-- QUERY 10: Retention Action Plan - CRITICAL Priority
SELECT TOP 100
    c.customer_id,
    c.country,
    c.monthly_subscription,
    f.total_revenue,
    p.churn_probability,
    CASE 
        WHEN f.failed_payment_count > 0 THEN 'Payment Issue'
        WHEN b.days_since_last_login > 30 THEN 'Inactivity'
        WHEN b.support_satisfaction_score < 3 THEN 'Support Issue'
        WHEN f.downgrade_count > 0 THEN 'Downgrade Signal'
        ELSE 'Low Engagement'
    END AS primary_issue,
    CASE 
        WHEN f.failed_payment_count > 0 THEN 'Payment Recovery'
        WHEN b.days_since_last_login > 30 THEN 'Re-engagement Campaign'
        WHEN b.support_satisfaction_score < 3 THEN 'Support Quality Intervention'
        WHEN f.downgrade_count > 0 THEN 'Premium Feature Showcase'
        ELSE 'Engagement Incentive'
    END AS recommended_action
FROM customers c
JOIN behavior b ON c.customer_id = b.customer_id
JOIN financial f ON c.customer_id = f.customer_id
JOIN churn_predictions p ON c.customer_id = p.customer_id
WHERE p.retention_priority = 'CRITICAL'
ORDER BY f.total_revenue DESC;

-- ==================== INDEX CREATION ====================

CREATE INDEX idx_churn_risk ON churn_predictions(churn_probability DESC);
CREATE INDEX idx_customer_value ON financial(total_revenue DESC);
CREATE INDEX idx_engagement ON behavior(login_count_last_30 DESC);
CREATE INDEX idx_retention ON churn_predictions(retention_priority);

-- ==================== END OF QUERIES ====================
