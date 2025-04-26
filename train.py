# train_upload_model.py

import os
import csv
import numpy as np
import joblib
import tempfile
from sklearn.ensemble import IsolationForest
import cloudinary
import cloudinary.uploader

# your train_and_upload_model function, unchanged
def _configure_cloudinary(cloud_name, api_key, api_secret):
    cloudinary.config(
        cloud_name=cloud_name,
        api_key=api_key,
        api_secret=api_secret
    )

def train_and_upload_model(
    records,
    cloudinary_cloud_name,
    cloudinary_api_key,
    cloudinary_api_secret,
    contamination=0.1,
    public_id="models/global"
):
    if len(records) < 10:
        return {"success": False, "error": "Need ≥10 records", "records_used": len(records)}

    X = np.array([[r["temperature"], r["speed"], r["heart_beat"]] for r in records])
    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(X)

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        joblib.dump(model, tmp.name)
        tmp_path = tmp.name

    try:
        _configure_cloudinary(cloudinary_cloud_name, cloudinary_api_key, cloudinary_api_secret)
        res = cloudinary.uploader.upload(
            tmp_path,
            resource_type="raw",
            public_id=public_id,
            overwrite=True
        )
        return {"success": True, "model_url": res["secure_url"], "records_used": len(records)}
    except Exception as e:
        return {"success": False, "error": f"Upload failed: {e}"}
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def load_records_from_csv(csv_path):
    """
    Read the CSV and return list of dicts with proper types.
    """
    records = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append({
                "timestamp": row["timestamp"],
                "heart_beat": int(row["heart_beat"]),
                "temperature": float(row["temperature"]),
                "speed": float(row["speed"])
            })
    return records

if __name__ == "__main__":
    # Usage: python train_upload_model.py <input.csv>
    import sys
    if len(sys.argv) != 2:
        print("Usage: python train_upload_model.py <input.csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    records = load_records_from_csv(csv_path)

    CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME")
    API_KEY    = os.getenv("CLOUDINARY_API_KEY")
    API_SECRET = os.getenv("CLOUDINARY_API_SECRET")

    result = train_and_upload_model(
        records,
        CLOUD_NAME,
        API_KEY,
        API_SECRET,
        contamination=0.1,
        public_id="models/global"
    )

    if result["success"]:
        print("Model uploaded successfully! URL =", result["model_url"])
    else:
        print("Training/upload failed:", result.get("error"))