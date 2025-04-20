import pandas as pd
from sklearn.ensemble import IsolationForest
import numpy as np

def train_anomaly_model(records, contamination=0.1, random_state=42):
    """
    Train an Isolation Forest model on the dataset.
    
    Returns:
        model: trained Isolation Forest model
        means: mean values of the features
        stds: standard deviation of the features
    """
    X = np.array([[r['temperature'], r['speed'], r['heart_beat']] for r in records])

    model = IsolationForest(contamination=contamination, random_state=random_state)
    model.fit(X)

    # means = X.mean()
    # stds = X.std()

    return model