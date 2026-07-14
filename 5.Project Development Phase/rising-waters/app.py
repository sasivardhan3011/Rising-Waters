"""Flask web application for flood risk prediction."""

import json
import os
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, flash, render_template, request

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_model.pkl"
SCALER_PATH = ROOT / "models" / "scaler.pkl"
METADATA_PATH = ROOT / "models" / "model_metadata.json"

app = Flask(__name__)
app.secret_key = "rising-waters-flood-prediction"

model = None
scaler = None
metadata = None
feature_columns = [
    "annual_rainfall_mm",
    "monsoon_rainfall_mm",
    "pre_monsoon_rainfall_mm",
    "post_monsoon_rainfall_mm",
    "cloud_visibility_km",
]


def load_artifacts() -> None:
    global model, scaler, metadata

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run `python train_model.py` first."
        )

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    if METADATA_PATH.exists():
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    else:
        metadata = {"best_model": "Unknown", "accuracy_percent": "N/A"}


def predict_flood(features: dict) -> dict:
    row = np.array([[features[col] for col in feature_columns]])
    uses_scaler = metadata.get("uses_scaler", False) if metadata else False

    if uses_scaler:
        row = scaler.transform(row)

    prediction = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0][1])

    if prediction == 1:
        if probability >= 0.8:
            risk_level = "High"
            advisory = (
                "High flood probability detected. Issue evacuation advisories "
                "and deploy emergency resources immediately."
            )
        else:
            risk_level = "Moderate"
            advisory = (
                "Moderate flood risk. Monitor conditions closely and prepare "
                "contingency plans for affected districts."
            )
    else:
        risk_level = "Low"
        advisory = (
            "Low flood risk based on current readings. Continue routine "
            "monitoring during the monsoon season."
        )

    return {
        "prediction": prediction,
        "probability": round(probability * 100, 2),
        "risk_level": risk_level,
        "advisory": advisory,
    }


try:
    load_artifacts()
except FileNotFoundError:
    pass


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    form_values = {
        "annual_rainfall_mm": "",
        "monsoon_rainfall_mm": "",
        "pre_monsoon_rainfall_mm": "",
        "post_monsoon_rainfall_mm": "",
        "cloud_visibility_km": "",
    }

    if request.method == "POST":
        if model is None:
            flash("Model not loaded. Run `python train_model.py` first.", "error")
            return render_template(
                "index.html",
                result=result,
                form_values=form_values,
                metadata=metadata,
            )

        try:
            features = {}
            for field in feature_columns:
                raw = request.form.get(field, "").strip()
                if not raw:
                    raise ValueError(f"{field.replace('_', ' ')} is required.")
                value = float(raw)
                if value < 0:
                    raise ValueError(f"{field.replace('_', ' ')} cannot be negative.")
                features[field] = value
                form_values[field] = raw

            result = predict_flood(features)
        except ValueError as exc:
            flash(str(exc), "error")

    return render_template(
        "index.html",
        result=result,
        form_values=form_values,
        metadata=metadata,
    )


if __name__ == "__main__":
    load_artifacts()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
