# Customer Churn Prediction

**Which subscribers are about to cancel, why, and who should the retention team call first?**

This project builds a churn model for a subscription business, from raw customer tables through to a ranked call list. It covers feature engineering in Python, a Random Forest classifier in scikit-learn, and ten SQL queries that turn the predictions into retention decisions.

The data is synthetic. I generated 10,000 customers across profile, behavior, payment and churn tables, with churn driven by a known set of risk factors. That makes it a good test of whether the model can find the real drivers, since I know what they are.

---

## Key findings

The overall churn rate is **19.4%**.

1. **Inactivity is the clearest warning sign.** Customers who hadn't logged in for 30+ days churned at 36.5%, more than twice the 15.1% rate for active customers.
2. **Repeated payment failures double the risk.** Customers with 2+ failed payments churned at 40.1%, against 17.2% for everyone else.
3. **Bad support experiences drive people away.** Customers who rated support below 2 out of 5 churned at 38.5%, against 15.5% for the rest.
4. **Plan tier doesn't matter.** Basic, Standard and Premium all churn at 18–21%, so pricing isn't the lever here.
5. **The model ranks these same factors highest.** Support satisfaction, days since last login and failed payments came out as the top three features. It found the drivers without being told what they were.

## How well the model works

All numbers below are on 2,000 held-out customers the model never saw in training.

| Metric | Value |
|---|---|
| ROC AUC | 0.72 |
| Recall | 62% of churners caught |
| Precision | 34% of flagged customers actually churned |

The risk tiers separate customers well. The High tier churns at **42.6%**, against **9.4%** for the Low tier. Calling only the High tier (19% of customers) reaches **42%** of everyone who would have churned.

Precision is modest because the model is set to favor catching churners over avoiding false alarms. For a retention team that's usually the right trade, since a wasted call costs far less than a lost customer.

## Recommendation

Start retention calls with the **113 CRITICAL customers**: high churn risk and high lifetime value, about $166,000 in combined revenue. Match the outreach to the reason. A payment recovery email for failed payments, a re-engagement offer for inactive accounts, and a follow-up from a support lead for low satisfaction scores. Query 10 in the SQL file produces this list with the reason and action for each customer.

---

## How it works

| Step | File | What it does |
|---|---|---|
| 1 | `01_generate_churn_data_SIMPLE.py` | Builds the five customer tables |
| 2 | `02_train_model_SIMPLE.py` | Engineers features, trains the model, scores every customer |
| 3 | `03_sql_queries.sql` | Schema plus ten business queries on the scored data |

**Feature engineering.** On top of the raw columns, the model uses engagement ratios (logins and content views per day of tenure), feature adoption rate, payment reliability, and three yes/no risk flags for payment trouble, low engagement and inactivity.

**Why Random Forest.** It handles a mix of counts, ratios and flags without much tuning, and its feature importances are easy to explain to a non-technical audience. Class weights are balanced so the model doesn't just predict "no churn" for everyone.

**Honest scoring.** The predictions file includes an `in_test` column. Rows marked 1 were held out from training, and they're the only ones used to measure accuracy.

**SQL portability.** The queries use standard SQL and were tested against both SQLite and DuckDB, so they'll run on Postgres, Snowflake or BigQuery with little or no change.

## Output files

| File | Contents |
|---|---|
| `master_data.csv` | All four input tables joined, one row per customer |
| `churn_predictions.csv` | Churn probability, risk tier, value segment and retention priority for every customer |
| `feature_importance.csv` | How much each feature contributes to the model |

`churn_predictions.csv` is ready to load into Power BI or Tableau.

## Run it yourself

Works in Google Colab or locally.

```bash
git clone https://github.com/vamsi93832/customer-churn-prediction.git
cd customer-churn-prediction
pip install -r requirements.txt

python 01_generate_churn_data_SIMPLE.py
python 02_train_model_SIMPLE.py
```

The data uses a fixed snapshot date and random seed, so every run gives the same results.

---

Built by Vamsi Krishna Chirumalla · [Portfolio](https://vamsikrishnachirumalla.com)
