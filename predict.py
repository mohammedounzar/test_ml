import requests
from tensorflow import keras
import numpy as np
import os

def download_and_predict(user_id, input_features):
    url = f"https://res.cloudinary.com/your_cloud_name/raw/upload/models/{user_id}.h5"
    r = requests.get(url)
    with open("temp_model.h5", "wb") as f:
        f.write(r.content)

    model = keras.models.load_model("temp_model.h5")
    os.remove("temp_model.h5")

    features = np.array(input_features).reshape(1, -1)  # e.g. [36.8, 215, -7.55]
    prediction = model.predict(features)
    print(f"Predicted heart rate: {prediction[0][0]:.2f}")

download_and_predict("user123", [36.8, 215, -7.55])
