import requests
import joblib
import numpy as np
import os

def download_and_predict(user_id, input_features):
    url = f"https://res.cloudinary.com/ddnkpgyqv/raw/upload/models/{user_id}.pkl"
    r = requests.get(url)
    with open("temp_model.pkl", "wb") as f:
        f.write(r.content)

    model = joblib.load("temp_model.pkl")
    os.remove("temp_model.pkl")

    features = np.array(input_features).reshape(1, -1)
    prediction = model.predict(features)
    return prediction[0]

# Example usage
result = download_and_predict("user123", [36.8, 5.0, 90])
print("Prediction:", "normal" if result == 1 else "not normal")
