import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


TRAIN_CSV = Path("data/features/HG002.indel.csv")
VAL_CSV = Path("data/features/HG003.indel.csv")
TEST_CSV = Path("data/features/HG004.indel.csv")
MODEL_PATH = Path("models/hr_lgbm.pkl")
META_PATH = Path("models/hr_lgbm.meta.json")
LOG_EVERY_N_ROUNDS = 20

EXCLUDE_COLUMNS = ["sample", "chrom", "pos", "ref", "alt", "label"]

MODEL_PARAMS = {
    "objective": "binary",
    "n_estimators": 400,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "class_weight": "balanced",
    "random_state": 42,
}


def load_split(path):
    df = pd.read_csv(path)
    return df


def label_counts(df):
    counts = df["label"].value_counts().sort_index()
    return {int(label): int(count) for label, count in counts.items()}


def metric_dict(y_true, y_pred):
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def select_threshold(y_true, probabilities):
    best_threshold = None
    best_metrics = None

    for threshold in np.round(np.arange(0.05, 0.951, 0.01), 2):
        y_pred = (probabilities >= threshold).astype(int)
        metrics = metric_dict(y_true, y_pred)
        if best_metrics is None or metrics["f1"] > best_metrics["f1"]:
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_threshold, best_metrics


def print_metric_line(prefix, metrics):
    print(
        f"{prefix} precision/recall/F1: "
        f"{metrics['precision']:.6f} / {metrics['recall']:.6f} / {metrics['f1']:.6f}"
    )


def main():
    print("Loading CSV files...", flush=True)
    train_df = load_split(TRAIN_CSV)
    val_df = load_split(VAL_CSV)
    test_df = load_split(TEST_CSV)

    feature_columns = [col for col in train_df.columns if col not in EXCLUDE_COLUMNS]
    expected_columns = EXCLUDE_COLUMNS + feature_columns

    for split_name, df in [("validation", val_df), ("test", test_df)]:
        missing = [col for col in expected_columns if col not in df.columns]
        extra = [col for col in df.columns if col not in expected_columns]
        if missing or extra:
            raise ValueError(
                f"{split_name} columns do not match training columns: "
                f"missing={missing}, extra={extra}"
            )

    x_train = train_df[feature_columns]
    y_train = train_df["label"].astype(int)
    x_val = val_df[feature_columns]
    y_val = val_df["label"].astype(int)
    x_test = test_df[feature_columns]
    y_test = test_df["label"].astype(int)

    print("Training LightGBM model...", flush=True)
    model = lgb.LGBMClassifier(**MODEL_PARAMS)
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_train, y_train), (x_val, y_val)],
        eval_names=["train", "validation"],
        eval_metric=["binary_logloss", "auc"],
        callbacks=[lgb.log_evaluation(period=LOG_EVERY_N_ROUNDS)],
    )

    print("Searching validation threshold...", flush=True)
    val_prob = model.predict_proba(x_val)[:, 1]
    best_threshold, val_metrics = select_threshold(y_val, val_prob)

    print("Evaluating test set...", flush=True)
    test_prob = model.predict_proba(x_test)[:, 1]
    test_pred = (test_prob >= best_threshold).astype(int)
    test_metrics = metric_dict(y_test, test_pred)

    print("Saving artifacts...", flush=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    meta = {
        "train_csv": str(TRAIN_CSV),
        "validation_csv": str(VAL_CSV),
        "test_csv": str(TEST_CSV),
        "model_path": str(MODEL_PATH),
        "feature_columns": feature_columns,
        "excluded_columns": EXCLUDE_COLUMNS,
        "model_params": MODEL_PARAMS,
        "best_validation_threshold": best_threshold,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "shapes": {
            "train": list(train_df.shape),
            "validation": list(val_df.shape),
            "test": list(test_df.shape),
        },
        "label_counts": {
            "train": label_counts(train_df),
            "validation": label_counts(val_df),
            "test": label_counts(test_df),
        },
        "threshold_search": {
            "start": 0.05,
            "stop": 0.95,
            "step": 0.01,
            "comparison": "probability >= threshold",
        },
        "training_log_period": LOG_EVERY_N_ROUNDS,
        "library_versions": {
            "lightgbm": lgb.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    META_PATH.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print("Shapes:")
    print(f"  train: {train_df.shape}")
    print(f"  val:   {val_df.shape}")
    print(f"  test:  {test_df.shape}")
    print("Label counts:")
    print(f"  train: {label_counts(train_df)}")
    print(f"  val:   {label_counts(val_df)}")
    print(f"  test:  {label_counts(test_df)}")
    print(f"Best validation threshold: {best_threshold:.2f}")
    print_metric_line("Validation", val_metrics)
    print_metric_line("Test", test_metrics)
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metadata: {META_PATH}")


if __name__ == "__main__":
    main()
