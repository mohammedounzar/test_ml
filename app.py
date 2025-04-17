import firebase_admin
from firebase_admin import credentials, firestore
import numpy as np
import tensorflow as tf
import cloudinary
import cloudinary.uploader
import os
import uuid

# --- Setup Firebase ---
cred = credentials.Certificate("firebase-adminsdk.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# --- Setup Cloudinary ---
cloudinary.config(
    cloud_name='your_cloud_name',
    api_key='your_api_key',
    api_secret='your_api_secret'
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
    X = np.array([[r['temperature'], r['altitude'], r['longitude']] for r in records])
    y = np.array([r['heart_beat'] for r in records])

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(3,)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=5, verbose=0)
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