import pandas as pd
from sklearn.ensemble import IsolationForest

# Create the dataset
data = [
    { "temperature": 36.5, "heart_beat": 85, "speed": 0.0, "activity": "rest", "timestamp": "2025-04-17T10:00:00Z" },
    { "temperature": 36.8, "heart_beat": 88, "speed": 1.03, "activity": "rest", "timestamp": "2025-04-17T10:05:00Z" },
    { "temperature": 37.0, "heart_beat": 92, "speed": 1.03, "activity": "rest", "timestamp": "2025-04-17T10:10:00Z" },
    { "temperature": 36.6, "heart_beat": 84, "speed": 2.47, "activity": "light", "timestamp": "2025-04-17T10:15:00Z" },
    { "temperature": 37.1, "heart_beat": 95, "speed": 2.89, "activity": "light", "timestamp": "2025-04-17T10:20:00Z" },
    { "temperature": 36.4, "heart_beat": 80, "speed": 4.49, "activity": "intense", "timestamp": "2025-04-17T10:25:00Z" },
    { "temperature": 36.9, "heart_beat": 90, "speed": 3.32, "activity": "intense", "timestamp": "2025-04-17T10:30:00Z" },
    { "temperature": 36.7, "heart_beat": 87, "speed": 1.24, "activity": "rest", "timestamp": "2025-04-17T10:35:00Z" },
    { "temperature": 36.3, "heart_beat": 78, "speed": 1.28, "activity": "rest", "timestamp": "2025-04-17T10:40:00Z" },
    { "temperature": 37.2, "heart_beat": 97, "speed": 5.12, "activity": "intense", "timestamp": "2025-04-17T10:45:00Z" },
    { "temperature": 36.2, "heart_beat": 75, "speed": 0.0, "activity": "rest", "timestamp": "2025-04-17T10:50:00Z" },
    { "temperature": 36.7, "heart_beat": 89, "speed": 1.2, "activity": "rest", "timestamp": "2025-04-17T10:55:00Z" },
    { "temperature": 37.0, "heart_beat": 91, "speed": 1.5, "activity": "light", "timestamp": "2025-04-17T11:00:00Z" },
    { "temperature": 36.4, "heart_beat": 84, "speed": 2.1, "activity": "light", "timestamp": "2025-04-17T11:05:00Z" },
    { "temperature": 36.8, "heart_beat": 85, "speed": 3.4, "activity": "intense", "timestamp": "2025-04-17T11:10:00Z" },
    { "temperature": 37.1, "heart_beat": 93, "speed": 4.0, "activity": "intense", "timestamp": "2025-04-17T11:15:00Z" },
    { "temperature": 37.5, "heart_beat": 100, "speed": 3.6, "activity": "intense", "timestamp": "2025-04-17T11:20:00Z" },
    { "temperature": 37.0, "heart_beat": 90, "speed": 1.1, "activity": "rest", "timestamp": "2025-04-17T11:25:00Z" },
    { "temperature": 38.0, "heart_beat": 80, "speed": 0.0, "activity": "rest", "timestamp": "2025-04-17T11:30:00Z" },
    { "temperature": 36.3, "heart_beat": 86, "speed": 1.8, "activity": "light", "timestamp": "2025-04-17T11:35:00Z" },
    { "temperature": 36.9, "heart_beat": 92, "speed": 2.2, "activity": "light", "timestamp": "2025-04-17T11:40:00Z" }
]

# Load data into a pandas DataFrame
df = pd.DataFrame(data)

# Drop timestamp and activity columns as they are not relevant for the model
X = df.drop(['timestamp', 'activity'], axis=1)

# Initialize the Isolation Forest model
iso_forest = IsolationForest(contamination=0.1, random_state=42)

# Fit the model and predict anomalies
df['anomaly'] = iso_forest.fit_predict(X)

# Map the prediction results: -1 = anomaly, 1 = normal
df['anomaly'] = df['anomaly'].map({1: 'normal', -1: 'not normal'})

# Display the result
print(df[['temperature', 'heart_beat', 'speed', 'anomaly']])
