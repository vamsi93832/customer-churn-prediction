# the dashboard
#
# five tabs, one per audience:
#   call list       -> retention team picking who to phone today
#   why they leave  -> the patterns behind churn, for the CX lead
#   revenue at risk -> finance sizing what a retention push is worth
#   sql answers     -> every business question with the query behind it
#   model quality   -> how much to trust the score
#
# reads only from outputs/, which is committed, so the live site never
# has to download data or retrain anything when it wakes up
#
#   streamlit run app.py

import json
import re
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
OUT = ROOT / "outputs"

st.set_page_config(page_title="Customer Churn", layout="wide")

if not (OUT / "churn_predictions.csv").exists():
    st.error("No results yet. Run 01_prepare_data.py, 02_train_model.py and 03_run_sql.py first.")
    st.stop()


@st.cache_data
def load():
    preds = pd.read_csv(OUT / "churn_predictions.csv")
    sql = {p.stem: pd.read_csv(p) for p in (OUT / "sql").glob("*.csv")}
    return (preds, sql, json.loads((OUT / "model_metrics.json").read_text()),
            pd.read_csv(OUT / "model_comparison.csv"), pd.read_csv(OUT / "churn_drivers.csv"))


def sql_blocks():
    # question text and query straight from the sql file, so what's shown
    # is always the exact query that made the numbers
    text = (ROOT / "sql" / "questions.sql").read_text()
    out = {}
    for block in re.split(r"\n(?=-- name: )", text):
        name = re.search(r"-- name: (\w+)", block)
        if name:
            question = re.search(r"-- question: (.+)", block).group(1).strip()
            out[name.group(1)] = (question, block[block.index("\n", block.index("-- question:")):].strip())
    return out


def rate_chart(df, label, order=None, height=200):
    # one horizontal bar chart style for every "churn rate by X" view
    return (alt.Chart(df).mark_bar(color="#d1495b")
            .encode(y=alt.Y(f"{label}:N", sort=order or "-x", title=None),
                    x=alt.X("churn_rate_pct:Q", title="Churn rate (%)"),
                    tooltip=[label, "customers", "churn_rate_pct"])
            .properties(height=height))


preds, sql, m, comparison, drivers = load()
active = preds[preds["churn"] == 0]  # people who already left can't be saved
test = preds[preds["in_test"] == 1]

st.title("Customer Churn: Who to Call and Why")
st.caption("Built on the IBM Telco Customer Churn dataset (7,043 customers of a fictional telecom "
           "company). For portfolio demonstration.")

call_tab, why_tab, money_tab, sql_tab, model_tab = st.tabs(
    ["Retention call list", "Why customers leave", "Revenue at risk", "SQL answers", "Model quality"])


with call_tab:
    high_active = active[active["risk_tier"] == "High"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Active customers", f"{len(active):,}")
    k2.metric("High risk, still active", f"{len(high_active):,}")
    k3.metric("Their monthly revenue", f"${high_active['MonthlyCharges'].sum():,.0f}")
    k4.metric("Leavers caught by High tier", f"{m['recall']:.0%}",
              help="Share of held-out customers who left that the model put in the High tier")

    c1, c2, c3 = st.columns(3)
    tiers = c1.multiselect("Risk tier", ["High", "Medium", "Low"], default=["High"])
    contracts = c2.multiselect("Contract", sorted(active["Contract"].unique()))
    min_bill = c3.slider("Minimum monthly bill ($)", 0, 120, 0, step=10)

    view = active[active["risk_tier"].isin(tiers) & (active["MonthlyCharges"] >= min_bill)]
    if contracts:
        view = view[view["Contract"].isin(contracts)]
    # likely leavers on big bills first: that's where a call saves the most money
    view = view.assign(priority=view["churn_score"] * view["MonthlyCharges"]).sort_values("priority", ascending=False)

    st.write(f"**{len(view):,} customers** match these filters, highest expected loss first.")
    table = view.rename(columns={
        "customerID": "Customer", "risk_tier": "Tier", "churn_score": "Risk score",
        "tenure": "Months", "MonthlyCharges": "Monthly bill", "main_risk": "Why at risk",
        "suggested_action": "What to offer"})[
        ["Customer", "Tier", "Risk score", "Months", "Contract", "Monthly bill", "Why at risk", "What to offer"]]
    st.dataframe(table.head(500), hide_index=True, width="stretch",
                 column_config={"Risk score": st.column_config.ProgressColumn(format="%.2f", min_value=0.0, max_value=1.0),
                                "Monthly bill": st.column_config.NumberColumn(format="$%.2f")})
    st.download_button("Download call list (CSV)", table.to_csv(index=False),
                       file_name="retention_call_list.csv", mime="text/csv")


with why_tab:
    left, right = st.columns(2)
    with left:
        st.markdown("**By contract**")
        st.altair_chart(rate_chart(sql["churn_by_contract"], "contract", height=130), width="stretch")
        st.markdown("**By how long they've been a customer**")
        order = sql["churn_by_tenure"]["tenure_group"].tolist()
        st.altair_chart(rate_chart(sql["churn_by_tenure"], "tenure_group", order=order), width="stretch")
    with right:
        st.markdown("**By payment method**")
        st.altair_chart(rate_chart(sql["churn_by_payment"], "payment_method", height=170), width="stretch")
        st.markdown("**By number of add-on services** (internet customers)")
        addons = sql["add_ons_vs_churn"].assign(num_add_ons=lambda d: d["num_add_ons"].astype(str))
        st.altair_chart(rate_chart(addons, "num_add_ons", order=addons["num_add_ons"].tolist()), width="stretch")

    st.markdown("**What the model weighs most** (odds of leaving vs a typical customer)")
    # the "no internet service" rows all say the same thing as "InternetService: No",
    # so they're dropped here to keep the chart readable. full list is in outputs/
    d = drivers[~drivers["driver"].str.contains("No internet service|gender")]
    d = pd.concat([d.head(7), d.tail(5)])
    d["direction"] = d["odds_ratio"].map(lambda r: "Raises risk" if r > 1 else "Lowers risk")
    st.altair_chart(
        alt.Chart(d).mark_bar()
        .encode(y=alt.Y("driver:N", sort="-x", title=None),
                x=alt.X("odds_ratio:Q", title="Odds ratio (1.0 = no effect)", scale=alt.Scale(type="log")),
                color=alt.Color("direction:N", scale=alt.Scale(domain=["Raises risk", "Lowers risk"],
                                                               range=["#d1495b", "#1f6f8b"]), title=None),
                tooltip=["driver", alt.Tooltip("odds_ratio:Q", format=".2f")])
        .properties(height=340), width="stretch")
    st.caption("Months as a customer is per 2 years of tenure. Each bar holds everything else equal.")


with money_tab:
    st.subheader("What is a retention push worth?")
    st.dataframe(sql["revenue_at_risk"], hide_index=True, width="stretch",
                 column_config={"monthly_revenue": st.column_config.NumberColumn(format="$%d"),
                                "expected_monthly_loss": st.column_config.NumberColumn(format="$%d")})
    st.caption("Each tier is priced at the churn rate it actually showed on held-out customers, "
               "not the raw model score, which runs high on purpose.")

    st.write("Change the assumptions to size a campaign aimed at one tier.")
    a1, a2, a3, a4 = st.columns(4)
    tier = a1.selectbox("Target tier", ["High", "Medium"])
    save_rate = a2.slider("Leavers you talk into staying", 0, 50, 20, format="%d%%")
    offer_cost = a3.number_input("Offer cost per customer per month ($)", value=10, step=5)
    months = a4.slider("Months a saved customer stays", 1, 24, 12)

    row = sql["revenue_at_risk"].set_index("risk_tier").loc[tier]
    targeted = active[active["risk_tier"] == tier]
    would_leave = row["active_customers"] * row["tier_churn_rate_pct"] / 100
    saved = would_leave * save_rate / 100
    revenue_kept = saved * targeted["MonthlyCharges"].mean() * months
    # the discount goes to everyone you offer it to, not just the ones who would have left
    cost = len(targeted) * offer_cost * months

    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Customers targeted", f"{len(targeted):,}")
    f2.metric("Leavers saved", f"{saved:,.0f}")
    f3.metric("Revenue kept", f"${revenue_kept:,.0f}")
    f4.metric("Net after offer cost", f"${revenue_kept - cost:,.0f}")
    st.caption("A rough sizing, not a forecast. The offer cost applies to everyone targeted, "
               "including customers who would have stayed anyway.")


with sql_tab:
    blocks = sql_blocks()
    names = [n for n in blocks if n in sql]
    pick = st.selectbox("Question", names, format_func=lambda n: blocks[n][0])
    st.dataframe(sql[pick], hide_index=True, width="stretch")
    with st.expander("Show the SQL"):
        st.code(blocks[pick][1], language="sql")


with model_tab:
    st.subheader("How good is the risk score?")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC AUC", f"{m['roc_auc']:.3f}", help="Measured on 25% of customers held out from training")
    c2.metric("Customers flagged High", f"{m['flagged_share']:.0%}")
    c3.metric("Precision", f"{m['precision']:.0%}", help="Share of High-tier customers who actually left")
    c4.metric("Lift vs random", f"{m['lift_vs_random']}x")

    left, right = st.columns(2)
    with left:
        st.markdown("**Do the tiers separate leavers?** (held-out customers)")
        check = sql["score_check"]
        st.altair_chart(
            alt.Chart(check).mark_bar()
            .encode(x=alt.X("risk_tier:N", sort=["Low", "Medium", "High"], title=None),
                    y=alt.Y("actual_churn_pct:Q", title="Actually left (%)"),
                    color=alt.Color("risk_tier:N", legend=None, scale=alt.Scale(
                        domain=["Low", "Medium", "High"], range=["#1f6f8b", "#edae49", "#d1495b"])),
                    tooltip=["risk_tier", "held_out_customers", "actual_churn_pct"])
            .properties(height=240), width="stretch")
    with right:
        st.markdown("**Three models compared** (5-fold cross-validation on training customers)")
        st.dataframe(comparison.rename(columns={"model": "Model", "cv_auc": "CV AUC",
                                                "cv_auc_std": "+/-", "test_auc": "Held-out AUC"}),
                     hide_index=True, width="stretch")
        st.markdown(f"""
- **{m['model']}** was chosen. The tree models scored within 0.01 AUC, and a model the
  retention team can explain is worth more than a fraction of a point.
- Flagging the top **{m['flagged_share']:.0%}** catches **{m['recall']:.0%}** of leavers,
  **{m['lift_vs_random']}x** better than calling the same number at random.
- About **{m['precision']:.0%}** of High-tier customers actually left, against a base rate of
  {m['base_rate']:.0%}.
""")
