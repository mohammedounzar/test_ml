import pandas as pd
from sklearn.ensemble import IsolationForest

# Load data into a pandas DataFrame
df = pd.read_json('user123_data.json')

# Drop timestamp and activity columns as they are not relevant for the model
X = df.drop(['timestamp'], axis=1)

# Initialize the Isolation Forest model
iso_forest = IsolationForest(contamination=0.1, random_state=42)

# Fit the model and predict anomalies
df['anomaly'] = iso_forest.fit_predict(X)

# Map the prediction results: -1 = anomaly, 1 = normal
df['anomaly'] = df['anomaly'].map({1: 'normal', -1: 'not normal'})

# Display the result
print(df[['temperature', 'heart_beat', 'speed', 'anomaly']])

# Create a new record to predict
new_record = pd.DataFrame({
    'temperature': [38.5],
    'heart_beat': [90],
    'speed': [5]
})

# Predict for the new record
prediction = iso_forest.predict(new_record)
result = 'normal' if prediction[0] == 1 else 'not normal'
print('\nPrediction for new record:')
print(f'Input data: {new_record.iloc[0].to_dict()}')
print(f'Prediction: {result}')