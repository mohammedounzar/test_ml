from sklearn.ensemble import IsolationForest
import pandas as pd

# Load the JSON file
df = pd.read_json("user123_data.json")

# Drop the timestamp column (assumes it exists)
X = df.drop('timestamp', axis=1)

# Apply Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
df['anomaly'] = iso_forest.fit_predict(X)

# Interpret result: -1 = anomaly, 1 = normal
df['anomaly'] = df['anomaly'].map({1: 'normal', -1: 'not normal'})

# Display selected columns
print(df[['temperature', 'altitude', 'longitude', 'heart_beat', 'anomaly']])
