# Step 2 - train the churn model and build the retention call list
#
# compares three models with cross-validation on the training customers only,
# scores the held-out customers once at the end, then ranks everyone into
# risk tiers sized to what a retention team can actually call.
#
#   python 02_train_model.py

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).parent
CLEAN = ROOT / "data" / "telco_clean.csv"
OUT = ROOT / "outputs"
RANDOM_STATE = 42

# only tenure on the numeric side, on purpose. TotalCharges is basically
# tenure x monthly bill, and the monthly bill is basically the sum of the
# services below. with all three in, logistic regression split the credit
# between them and "total billed so far" came out as a reason people LEAVE,
# which is backwards and would confuse anyone reading the drivers.
# dropping them cost about 0.002 AUC and every driver now reads the right way
NUMERIC = ["tenure"]
CATEGORICAL = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
]

# the high tier is the top 20% of scores because that's roughly how many
# calls a retention team can make in a month. it's a capacity decision,
# not a statistical one. change it if the real team is bigger or smaller
HIGH_SHARE = 0.20
MEDIUM_SHARE = 0.30


def prep():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), CATEGORICAL),
    ])


def candidates():
    # logistic regression is the one I'd want to ship if it holds up, since the
    # retention team can see exactly why someone was flagged. the two tree
    # models are here to check whether that simplicity costs real accuracy
    return {
        "Logistic regression": Pipeline([("prep", prep()), ("model", LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE))]),
        "Random forest": Pipeline([("prep", prep()), ("model", RandomForestClassifier(
            n_estimators=300, min_samples_leaf=10, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1))]),
        "Gradient boosting": Pipeline([("prep", prep()), ("model", HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=300, max_leaf_nodes=15, class_weight="balanced",
            random_state=RANDOM_STATE))]),
    }


def readable(feature):
    # "cat__Contract_Two year" -> "Contract: Two year". nobody on a retention
    # team should have to decode sklearn's column names
    name = feature.split("__", 1)[1]
    nice = {"tenure": "Months as a customer (per 2 years)"}
    if name in nice:
        return nice[name]
    for col in CATEGORICAL:
        if name.startswith(col + "_"):
            return f"{col}: {name[len(col) + 1:]}"
    return name


def drivers(pipe):
    names = pipe.named_steps["prep"].get_feature_names_out()
    coefs = pipe.named_steps["model"].coef_[0]
    out = pd.DataFrame({"driver": [readable(n) for n in names], "coefficient": coefs})
    # odds ratio is the easier thing to explain: 2.0 means "twice the odds of leaving"
    out["odds_ratio"] = np.exp(out["coefficient"])
    return out.sort_values("coefficient", ascending=False)


def main_risk(row):
    # every fixable risk a customer has, plus one offer that covers them.
    # the first version stopped at the first match, and since nearly every
    # high-risk customer is month-to-month, the whole call list said
    # "offer a contract" and the e-check and fiber problems never showed up
    risks, offers = [], []
    if row["Contract"] == "Month-to-month":
        if row["tenure"] <= 6:
            risks.append("New, no contract")
            offers.append("welcome call + discounted 1-year contract")
        else:
            risks.append("No contract")
            offers.append("discounted 1-year contract")
    if row["PaymentMethod"] == "Electronic check":
        risks.append("Pays by e-check")
        offers.append("auto-pay discount")
    if row["InternetService"] == "Fiber optic" and row["TechSupport"] == "No":
        risks.append("Fiber, no tech support")
        offers.append("free tech support trial")
    if not risks:
        return "No single clear risk", "General check-in call"
    offer = " + ".join(offers)
    return " + ".join(risks), offer[0].upper() + offer[1:]


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(CLEAN)
    X, y = df[NUMERIC + CATEGORICAL], df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)

    # model choice happens on the training customers only. the test set
    # gets used exactly once, at the end, so its score is an honest one
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    comparison = []
    for name, pipe in candidates().items():
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=folds, scoring="roc_auc")
        comparison.append({"model": name, "cv_auc": round(cv_auc.mean(), 4),
                           "cv_auc_std": round(cv_auc.std(), 4)})
        print(f"{name:20s} cross-validated AUC {cv_auc.mean():.3f} (+/- {cv_auc.std():.3f})")
    comparison = pd.DataFrame(comparison)

    # if logistic regression is within 0.01 AUC of the best, take it. a score
    # the team can explain beats a fraction of a point they can't
    best = comparison.loc[comparison["cv_auc"].idxmax()]
    lr_auc = comparison.set_index("model").loc["Logistic regression", "cv_auc"]
    chosen = "Logistic regression" if best["cv_auc"] - lr_auc <= 0.01 else best["model"]
    print(f"\nchosen model: {chosen}")

    pipe = candidates()[chosen].fit(X_train, y_train)

    # tier cutoffs come from the training scores, so the test customers are
    # judged against a line they had no part in drawing
    train_scores = pipe.predict_proba(X_train)[:, 1]
    high_cut = float(np.quantile(train_scores, 1 - HIGH_SHARE))
    med_cut = float(np.quantile(train_scores, 1 - HIGH_SHARE - MEDIUM_SHARE))

    test_scores = pipe.predict_proba(X_test)[:, 1]
    flagged = test_scores >= high_cut
    recall = recall_score(y_test, flagged)
    test_auc = roc_auc_score(y_test, test_scores)
    comparison["test_auc"] = np.nan
    comparison.loc[comparison["model"] == chosen, "test_auc"] = round(test_auc, 4)
    comparison.to_csv(OUT / "model_comparison.csv", index=False)

    metrics = {
        "model": chosen,
        "test_customers": int(len(y_test)),
        "base_rate": round(float(y_test.mean()), 4),
        "roc_auc": round(float(test_auc), 4),
        "flagged_share": round(float(flagged.mean()), 4),
        "recall": round(float(recall), 4),
        "precision": round(float(precision_score(y_test, flagged)), 4),
        # flagging 20% at random catches about 20% of churners.
        # lift is how many times better the model does than that
        "lift_vs_random": round(float(recall / flagged.mean()), 2),
        "high_cutoff": round(high_cut, 4),
        "medium_cutoff": round(med_cut, 4),
    }
    (OUT / "model_metrics.json").write_text(json.dumps(metrics, indent=2))

    if chosen == "Logistic regression":
        drivers(pipe).to_csv(OUT / "churn_drivers.csv", index=False)
    else:
        # still save the logistic regression drivers so the "why" is available
        lr = candidates()["Logistic regression"].fit(X_train, y_train)
        drivers(lr).to_csv(OUT / "churn_drivers.csv", index=False)

    # score every customer for the call list, and keep track of who was held out
    df["churn_score"] = pipe.predict_proba(X)[:, 1].round(4)
    df["risk_tier"] = np.where(df["churn_score"] >= high_cut, "High",
                      np.where(df["churn_score"] >= med_cut, "Medium", "Low"))
    df["in_test"] = df.index.isin(X_test.index).astype(int)
    df[["main_risk", "suggested_action"]] = df.apply(main_risk, axis=1, result_type="expand")

    keep = ["customerID", "risk_tier", "churn_score", "main_risk", "suggested_action",
            "tenure", "tenure_group", "Contract", "PaymentMethod", "auto_pay", "InternetService",
            "TechSupport", "OnlineSecurity", "num_add_ons", "SeniorCitizen", "PaperlessBilling",
            "MonthlyCharges", "TotalCharges", "churn", "in_test"]
    df[keep].to_csv(OUT / "churn_predictions.csv", index=False)

    test = df[df["in_test"] == 1]
    print("\nheld-out results:")
    print(json.dumps(metrics, indent=2))
    print("\nactual churn rate by tier (held-out customers):")
    print(test.groupby("risk_tier")["churn"].agg(customers="count", churn_rate="mean")
              .reindex(["High", "Medium", "Low"]).round(3))
    print("\nsaved outputs/churn_predictions.csv, model_metrics.json, model_comparison.csv, churn_drivers.csv")
