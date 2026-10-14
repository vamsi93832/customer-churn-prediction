# Step 3 - answer the business questions in SQL
#
# loads the cleaned data and the model's scores into DuckDB, builds the
# tables in sql/schema.sql, runs every question in sql/questions.sql, and
# saves each answer to outputs/sql/ for the dashboard.
#
# DuckDB because it runs real SQL from a single file with no server to set
# up, which makes it easy to run in Colab
#
#   python 03_run_sql.py

import re
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).parent
CLEAN = ROOT / "data" / "telco_clean.csv"
PREDICTIONS = ROOT / "outputs" / "churn_predictions.csv"
SQL_DIR = ROOT / "sql"
OUT = ROOT / "outputs" / "sql"


def questions():
    # each block starts with "-- name:" and "-- question:", so the file stays
    # the one source of truth and the dashboard can show the exact query
    text = (SQL_DIR / "questions.sql").read_text()
    out = []
    for block in re.split(r"\n(?=-- name: )", text):
        name = re.search(r"-- name: (\w+)", block)
        if not name:
            continue
        question = re.search(r"-- question: (.+)", block).group(1).strip()
        body = block[block.index("\n", block.index("-- question:")):].strip()
        out.append((name.group(1), question, body))
    return out


if __name__ == "__main__":
    if not PREDICTIONS.exists():
        raise SystemExit("run 01_prepare_data.py and 02_train_model.py first")
    OUT.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.register("clean", pd.read_csv(CLEAN))
    con.register("predictions", pd.read_csv(PREDICTIONS))
    con.execute((SQL_DIR / "schema.sql").read_text())

    for name, question, sql in questions():
        result = con.execute(sql).df()
        result.to_csv(OUT / f"{name}.csv", index=False)
        print(f"\n{question}\n{result.head(8).to_string(index=False)}")

    print(f"\nsaved {len(questions())} answers to outputs/sql/")
