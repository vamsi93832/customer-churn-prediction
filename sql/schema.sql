-- the IBM file is one wide table. splitting it the way a real telecom would
-- store it (who the customer is / their account / what they subscribe to /
-- the model's score) makes the questions read like real joins.
-- 03_run_sql.py builds these from data/telco_clean.csv and
-- outputs/churn_predictions.csv, then runs sql/questions.sql against them

CREATE TABLE customers AS
SELECT customerID, gender, SeniorCitizen, Partner, Dependents
FROM clean;

CREATE TABLE accounts AS
SELECT customerID, tenure, tenure_group, Contract, PaperlessBilling, PaymentMethod,
       auto_pay, MonthlyCharges, TotalCharges,
       churn  -- 1 = left within the last month of the snapshot
FROM clean;

CREATE TABLE services AS
SELECT customerID, PhoneService, MultipleLines, InternetService, OnlineSecurity,
       OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies,
       num_add_ons
FROM clean;

CREATE TABLE churn_scores AS
SELECT customerID, churn_score, risk_tier, main_risk, suggested_action,
       in_test  -- 1 = held out from training, the only rows used to judge accuracy
FROM predictions;
