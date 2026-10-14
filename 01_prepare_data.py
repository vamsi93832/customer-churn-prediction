# Step 1 - load and clean the IBM Telco churn data
#
# reads data/Telco-Customer-Churn.csv (downloads it first if it's missing),
# fixes the one real data problem, adds a few features, and writes
# data/telco_clean.csv for the model and the SQL step.
#
#   python 01_prepare_data.py

from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "Telco-Customer-Churn.csv"
CLEAN = ROOT / "data" / "telco_clean.csv"
OUT = ROOT / "outputs"

# IBM's public copy, Apache 2.0 licensed. the repo ships a copy too, so this
# only runs if someone deletes it
SOURCE_URL = ("https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
              "master/data/Telco-Customer-Churn.csv")

ADD_ON_SERVICES = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                   "TechSupport", "StreamingTV", "StreamingMovies"]


def load_raw():
    if not RAW.exists():
        print("no local copy, downloading from IBM's GitHub")
        RAW.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(SOURCE_URL, RAW)
    return pd.read_csv(RAW)


def clean(df):
    df = df.copy()

    # TotalCharges comes in as text because 11 rows are a blank space.
    # every one of them has tenure 0, so they're brand-new customers who
    # haven't been billed yet. 0 is the true value, not a missing one,
    # and dropping them would hide the newest customers from the model
    blank = df["TotalCharges"].str.strip() == ""
    assert (df.loc[blank, "tenure"] == 0).all(), "blank TotalCharges on a non-new customer, check the data"
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].where(~blank, "0"))

    # swap the Yes/No column for 1/0 and drop the original. keeping both meant
    # two columns named Churn and churn, and databases ignore case, so SQL
    # quietly picked up the text one
    df["churn"] = (df["Churn"] == "Yes").astype(int)
    df = df.drop(columns="Churn")
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    # tenure buckets are what a retention team actually talks in.
    # the first six months get their own bucket because that's where most churn happens
    df["tenure_group"] = pd.cut(df["tenure"], bins=[-1, 6, 12, 24, 48, 72],
                                labels=["0-6 months", "7-12 months", "1-2 years", "2-4 years", "4-6 years"]
                                ).astype(str)

    # how many add-ons each customer pays for. more services usually means
    # more reasons to stay, and it's one number instead of six yes/no columns
    df["num_add_ons"] = (df[ADD_ON_SERVICES] == "Yes").sum(axis=1)

    df["auto_pay"] = df["PaymentMethod"].str.contains("automatic").map({True: "Yes", False: "No"})
    return df


def quality_notes(raw, df):
    lines = [
        "# Data quality notes",
        "",
        f"- Rows: {len(raw):,}, one per customer (customerID is unique: {raw['customerID'].is_unique})",
        f"- Missing values: none, apart from {int((raw['TotalCharges'].str.strip() == '').sum())} blank "
        "TotalCharges, all on customers with 0 months of tenure. Set to 0, since they hadn't been billed yet.",
        f"- Churn rate: {df['churn'].mean():.1%}",
        "- Churn here means the customer left within the last month of the snapshot.",
        "- Source: IBM sample data for a fictional telecom company, based on realistic usage patterns.",
    ]
    (OUT / "data_quality.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    raw = load_raw()
    df = clean(raw)
    df.to_csv(CLEAN, index=False)
    quality_notes(raw, df)

    print(f"customers:  {len(df):,}")
    print(f"churn rate: {df['churn'].mean():.1%}")
    print(f"saved {CLEAN.relative_to(ROOT)} and outputs/data_quality.md")
