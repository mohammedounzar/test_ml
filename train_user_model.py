import firebase_admin
from firebase_admin import credentials, firestore
import numpy as np
import tensorflow as tf
import cloudinary
from dotenv import load_dotenv
import cloudinary.uploader
import os
import uuid
from anomaly_utils import train_anomaly_model

# --- Setup Firebase ---
cred = credentials.Certificate("firebase-adminsdk.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# --- Setup Cloudinary ---
load_dotenv()

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

def fetch_user_data(user_id):
    collection = db.collection("users").document(user_id).collection("data").stream()
    records = []
    for doc in collection:
        data = doc.to_dict()
        if all(k in data for k in ['temperature', 'altitude', 'longitude', 'heart_beat']):
            records.append(data)
    return records

def train_model(records):

    model = train_anomaly_model(records)
    return model

def save_and_upload_model(model, user_id):
    filename = f"{user_id}_{uuid.uuid4().hex}.h5"
    model.save(filename)

    upload_result = cloudinary.uploader.upload(
        filename,
        resource_type="raw",
        public_id=f"models/{user_id}",
        overwrite=True
    )

    os.remove(filename)  # clean up local file
    return upload_result['secure_url']

def main(user_id):
    print(f"Training model for user: {user_id}")
    records = fetch_user_data(user_id)
    if len(records) < 3:
        print("Not enough data to train.")
        return

    model = train_model(records)
    url = save_and_upload_model(model, user_id)
    print(f"Model uploaded to Cloudinary: {url}")

if __name__ == "__main__":
    main("user123")