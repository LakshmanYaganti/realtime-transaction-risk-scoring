"""Train the streaming risk model on the real fraud dataset.

Compares a logistic-regression baseline against XGBoost on a chronological
split, selects the winner by validation PR-AUC, evaluates it once on the
held-out (future) test period, saves the winning bundle to
``models/risk_model.joblib`` (gitignored — rebuild it with this script), and
merges the model metrics into ``reports/metrics.json``.

Usage:
    python -m src.train [--data PATH] [--metrics reports/metrics.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.data import load_data, time_ordered_split
from src.model import RiskModel, build_features

REPO_ROOT = Path(__file__).resolve().parent.parent


def evaluate(model: RiskModel, df) -> dict:
    y = df["Class"].to_numpy()
    scores = model.score_frame(df)
    pred = (scores >= 0.5).astype(int)
    return {
        "threshold": 0.5,
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "pr_auc": float(average_precision_score(y, scores)),
        "confusion_matrix": {
            "tn": int(((pred == 0) & (y == 0)).sum()),
            "fp": int(((pred == 1) & (y == 0)).sum()),
            "fn": int(((pred == 0) & (y == 1)).sum()),
            "tp": int(((pred == 1) & (y == 1)).sum()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    parser.add_argument("--metrics", default=str(REPO_ROOT / "reports" / "metrics.json"))
    parser.add_argument("--model-out", default=str(REPO_ROOT / "models" / "risk_model.joblib"))
    args = parser.parse_args()

    df = load_data(args.data)
    train, val, test = time_ordered_split(df)
    print(f"rows={len(df)} train={len(train)} val={len(val)} test={len(test)}")
    print(
        "fraud: "
        f"train={int(train['Class'].sum())} val={int(val['Class'].sum())} test={int(test['Class'].sum())}"
    )

    X_train, scaler = build_features(train)
    y_train = train["Class"].to_numpy()
    neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())

    candidates: dict[str, RiskModel] = {}

    lr = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    lr.fit(X_train, y_train)
    candidates["logistic_regression"] = RiskModel(lr, scaler, "logistic_regression")

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=neg / max(pos, 1),
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
    )
    xgb.fit(X_train, y_train)
    candidates["xgboost"] = RiskModel(xgb, scaler, "xgboost")

    results: dict[str, dict] = {}
    for name, model in candidates.items():
        results[name] = {"validation": evaluate(model, val), "test": evaluate(model, test)}
        v = results[name]["validation"]
        t = results[name]["test"]
        print(
            f"{name}: val PR-AUC={v['pr_auc']:.4f} | test PR-AUC={t['pr_auc']:.4f} "
            f"ROC-AUC={t['roc_auc']:.4f} P={t['precision']:.3f} R={t['recall']:.3f}"
        )

    winner_name = max(candidates, key=lambda n: results[n]["validation"]["pr_auc"])
    print(f"selected model (best validation PR-AUC): {winner_name}")

    model_path = Path(args.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    candidates[winner_name].save(str(model_path))

    metrics_path = Path(args.metrics)
    metrics: dict = {}
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
    metrics["dataset"] = {
        "rows": int(len(df)),
        "fraud": int(df["Class"].sum()),
        "fraud_rate": float(df["Class"].mean()),
        "split": "chronological 70/15/15 by Time",
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
        "test_rows": int(len(test)),
        "test_fraud": int(test["Class"].sum()),
    }
    metrics["model_comparison"] = results
    metrics["selected_model"] = winner_name
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"metrics -> {metrics_path}; model -> {model_path}")


if __name__ == "__main__":
    main()
