"""Train flood classification models and save the best performer."""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from generate_data import DATA_PATH, generate_flood_dataset

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
FEATURE_COLUMNS = [
    "annual_rainfall_mm",
    "monsoon_rainfall_mm",
    "pre_monsoon_rainfall_mm",
    "post_monsoon_rainfall_mm",
    "cloud_visibility_km",
]


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        generate_flood_dataset().to_csv(DATA_PATH, index=False)
    return pd.read_csv(DATA_PATH)


def train_models() -> None:
    df = load_dataset()
    X = df[FEATURE_COLUMNS]
    y = df["flood"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=8),
        "Random Forest": RandomForestClassifier(
            n_estimators=80, random_state=42, max_depth=8
        ),
        "KNN": KNeighborsClassifier(n_neighbors=11),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.92,
            colsample_bytree=0.92,
            min_child_weight=2,
            random_state=42,
            eval_metric="logloss",
        ),
    }

    preferred_on_tie = ["XGBoost", "Random Forest", "KNN", "Decision Tree"]

    results = {}
    best_name = None
    best_model = None
    best_accuracy = 0.0

    print("\nModel Performance on Test Set")
    print("-" * 40)

    for name, model in models.items():
        if name == "KNN":
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        results[name] = round(accuracy * 100, 2)
        print(f"{name:16s}  Accuracy: {results[name]:.2f}%")

        if accuracy > best_accuracy or (
            accuracy == best_accuracy
            and preferred_on_tie.index(name)
            < preferred_on_tie.index(best_name)
        ):
            best_accuracy = accuracy
            best_name = name
            best_model = model

    print("-" * 40)
    print(f"Best model: {best_name} ({results[best_name]:.2f}%)")

    test_X = X_test_scaled if best_name == "KNN" else X_test
    y_pred = best_model.predict(test_X)
    print("\nClassification Report (Best Model)")
    print(classification_report(y_test, y_pred, target_names=["No Flood", "Flood"]))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_DIR / "best_model.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")

    metadata = {
        "best_model": best_name,
        "accuracy_percent": results[best_name],
        "feature_columns": FEATURE_COLUMNS,
        "uses_scaler": best_name == "KNN",
        "all_results": results,
    }
    (MODEL_DIR / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(f"\nSaved model artifacts to {MODEL_DIR}")


if __name__ == "__main__":
    train_models()
