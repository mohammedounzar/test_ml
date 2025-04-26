import numpy as np
import joblib
import tempfile
import os
from sklearn.ensemble import IsolationForest
import cloudinary
import requests

# configure Cloudinary once
def _configure_cloudinary(cloud_name, api_key, api_secret):
    cloudinary.config(
        cloud_name=cloud_name,
        api_key=api_key,
        api_secret=api_secret
    )


def predict_anomalies(
    user_id,
    records,
    cloudinary_cloud_name,
    cloudinary_api_key,
    cloudinary_api_secret,
    public_id="models/global"
):
    """
    Download the single global model from Cloudinary and predict anomalies.

    Parameters:
      - user_id: str, identifier for the user
      - records: list of dicts, each with 'temperature','speed','heart_beat'
      - cloudinary_cloud_name: str, Cloudinary cloud name
      - cloudinary_api_key: str, Cloudinary API key
      - cloudinary_api_secret: str, Cloudinary API secret
      - public_id: same ID used at training time

    Returns:
      dict with:
        - success: bool
        - user_id: str
        - predictions: list of {id, is_anomaly, prediction, timestamp}
        - anomalies: list of record dicts flagged as anomalies
        - error: str (on failure)
    """
    if not records:
        return {"success": False, "user_id": user_id, "error": "No records provided"}

    for r in records:
        if not all(k in r for k in ("temperature","speed","heart_beat")):
            return {"success": False, "user_id": user_id, "error": "Invalid record format"}

    # download model file
    url = f"https://res.cloudinary.com/{cloudinary_cloud_name}/raw/upload/{public_id}.pkl"
    resp = requests.get(url)
    if resp.status_code != 200:
        return {"success": False, "user_id": user_id, "error": "Could not fetch model; re-train required"}

    # write to temp and load
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        tmp.write(resp.content)
        tmp_path = tmp.name

    try:
        model = joblib.load(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    predictions = []
    anomalies = []
    for r in records:
        feats = np.array([r["temperature"],r["speed"],r["heart_beat"]]).reshape(1,-1)
        pred = int(model.predict(feats)[0])  # 1 normal, -1 anomaly
        rec = {
            "id": r.get("id"),
            "is_anomaly": (pred == -1),
            "prediction": pred,
            "timestamp": r.get("timestamp")
        }
        predictions.append(rec)
        if pred == -1:
            anomalies.append(r)

    return {
        "success": True,
        "user_id": user_id,
        "predictions": predictions,
        "anomalies": anomalies
    }