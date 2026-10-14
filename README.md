# Customer Churn Prediction

**Which telecom customers are about to cancel, why, and who should the retention team call first?**

A telecom company lost 26.5% of its customers in a single month, taking $139,000 of monthly revenue with them. This project finds what drives those cancellations, scores every active customer's risk, and turns the scores into a call list where each customer comes with a reason and a matching offer. It runs end to end in Python, SQL and a Streamlit dashboard.

The data is the IBM Telco Customer Churn dataset: 7,043 customers with their contracts, services, billing and whether they left. IBM built it as sample data for a fictional company, modeled on real telecom patterns.

---

## Key findings

1. **Contracts are the biggest lever.** Month-to-month customers churned at 42.7%. One-year contracts churned at 11.3% and two-year contracts at 2.8%. Month-to-month customers account for $121,000 of the $139,000 in lost monthly revenue.
2. **The first six months decide everything.** New customers churn at 52.9%, and 42% of everyone who left did so within six months. Past four years, churn drops to 9.5%.
3. **Paying by electronic check is a red flag.** Those customers churned at 45.3%, against 15–19% for every other payment method. Customers on auto-pay churn least.
4. **Fiber customers without tech support are the worst-served group.** They churned at 49.4%. Fiber customers who have tech support churned at 22.6%, less than half that.
5. **Every add-on service makes people stickier.** Internet customers with no add-ons churned at 52.2%. With all six add-ons, churn falls to 5.3%.

## The risk score

A logistic regression model scores every customer. On 1,761 customers held out from training:

| Metric | Value |
|---|---|
| ROC AUC | 0.844 |
| Customers flagged High risk | 19% |
| Leavers caught by the High tier | 49% (2.5x better than random) |
| High-tier customers who actually left | 67% |

The tiers separate customers clearly. High-tier customers churned at **67.4%**, Medium at **34.0%**, and Low at **6.8%**.

## Recommendation

Call the **449 active High-risk customers** first, about $37,000 in monthly revenue. Every one of them is on a month-to-month contract, and 62% also pay by e-check and have fiber without tech support. So one bundled offer covers most of the list: a discounted 1-year contract, an auto-pay discount, and a free tech support trial.

Don't extend the same offer to the Medium tier as a blanket campaign. The Medium tier actually carries more expected monthly loss ($33,000 vs $25,000), because it's three times larger. But at the dashboard's default assumptions (a $10/month offer, 20% of leavers saved, 12 months), offering it to all 1,423 of them loses about $51,000, since most would have stayed anyway. The dashboard's revenue tab lets you test that trade-off with your own numbers.

---

## The dashboard

`streamlit run app.py` opens five tabs:

| Tab | Who it's for | What it does |
|---|---|---|
| Retention call list | Retention team | Active customers ranked by expected loss, with the reason and the offer, downloadable as CSV |
| Why customers leave | CX lead | Churn by contract, tenure, payment and add-ons, plus what the model weighs most |
| Revenue at risk | Finance | Expected monthly loss by tier, and an adjustable campaign calculator |
| SQL answers | Anyone | All ten business questions, each with the query that produced it |
| Model quality | Anyone checking the work | Accuracy, lift, tier check, and the three models compared |

## How it was built

| Step | Script | What it does |
|---|---|---|
| 1 | `01_prepare_data.py` | Loads the IBM data, fixes blank charges, adds tenure groups and add-on counts |
| 2 | `02_train_model.py` | Compares three models, scores every customer, builds risk tiers and the call list |
| 3 | `03_run_sql.py` | Builds the tables in `sql/schema.sql` and answers the questions in `sql/questions.sql` |
| 4 | `app.py` | The Streamlit dashboard |

## Decisions worth explaining

**Blank charges are zero, not missing.** Eleven customers have a blank total charge. All of them signed up that month and hadn't been billed yet, so their true value is 0. Dropping them would hide the newest customers, who are the ones most likely to leave.

**Why logistic regression.** I compared it against random forest and gradient boosting using 5-fold cross-validation on the training customers. Random forest scored 0.845 AUC against 0.842, a gap well inside the noise. A retention team has to explain the score to the people they call, so the model with readable drivers wins.

**Fewer billing columns, clearer drivers.** Total charges is roughly tenure times the monthly bill, and the monthly bill is roughly the sum of the services. With all three in the model, total charges came out as a reason customers *leave*, which is backwards. Dropping the two billing columns cost 0.002 AUC and made every driver point the right way.

**Tiers sized to capacity.** The High tier is the top 20% of scores, roughly the number of calls a retention team can make. The cutoffs are set on training customers only, then checked on held-out customers.

**Only active customers get called.** Customers who already left are kept for training and accuracy checks, but they're filtered out of the call list and the revenue-at-risk numbers. There's nothing left to save.

**Honest revenue at risk.** The model is weighted to catch leavers, so its raw scores run high. Revenue at risk is priced at each tier's actual churn rate on held-out customers instead.

**Every offer matches a reason.** Each customer on the call list gets every fixable risk they have (no contract, e-check payments, fiber without support) and one bundled offer that addresses all of them.

## Run it yourself

In Google Colab:

```python
!git clone https://github.com/vamsi93832/customer-churn-prediction.git
%cd customer-churn-prediction
!pip install -q -r requirements-pipeline.txt

!python 01_prepare_data.py
!python 02_train_model.py
!python 03_run_sql.py
```

Locally, run the same three scripts, then `streamlit run app.py`. The raw IBM file ships with the repo, so nothing needs downloading.

## Project structure

```
customer-churn-prediction/
├── 01_prepare_data.py
├── 02_train_model.py
├── 03_run_sql.py
├── app.py                       Streamlit dashboard
├── sql/
│   ├── schema.sql               splits the data into customers, accounts, services, scores
│   └── questions.sql            ten business questions
├── data/Telco-Customer-Churn.csv   raw IBM data
├── outputs/                     predictions, metrics, drivers, SQL answers (the dashboard reads these)
├── requirements.txt             dashboard only
└── requirements-pipeline.txt    adds scikit-learn and DuckDB for the full pipeline
```

## Data source

IBM, [Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d), Apache 2.0 license. A sample dataset for a fictional telecom company, created by IBM for teaching and demos.

---

Built by Vamsi Krishna Chirumalla · [Portfolio](https://vamsikrishnachirumalla.com) · [GitHub](https://github.com/vamsi93832)
