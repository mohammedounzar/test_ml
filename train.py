"""
Training script for LSTM-OC-SVM anomaly detection
Based on "Unsupervised and Semi-supervised Anomaly Detection with LSTM Neural Networks"

Usage: python train.py <input.csv>
"""

import os
import sys
import csv
import numpy as np
import tempfile
import joblib
import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv
from lstm_ocsvm_trainer import train_lstm_ocsvm, predict_anomalies


def load_sequences_from_csv(csv_path):
    """
    Load variable-length sequences from CSV file
    CSV format: timestamp,heart_beat,temperature,speed
    Sequences are separated by time gaps or when creating fixed-length windows
    """
    sequences = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        current_seq = []
        last_timestamp = None
        
        # Read all data first
        all_data = []
        for row in reader:
            # Parse timestamp string to get time ordering
            timestamp_str = row["timestamp"]
            features = [
                float(row["temperature"]), 
                float(row["speed"]), 
                float(row["heart_beat"])
            ]
            
            all_data.append({
                'timestamp': timestamp_str,
                'features': features
            })
        
        print(f"Loaded {len(all_data)} data points from CSV")
        
        # Create sequences using sliding window approach
        # This is better for continuous sensor data
        sequence_length = 20  # Fixed sequence length for training
        step_size = 10        # Overlap between sequences
        
        for i in range(0, len(all_data) - sequence_length + 1, step_size):
            sequence = []
            for j in range(sequence_length):
                sequence.append(all_data[i + j]['features'])
            
            if len(sequence) == sequence_length:
                sequences.append(np.array(sequence))
        
        # If we have remaining data, create one more sequence
        if len(all_data) >= sequence_length:
            remaining_start = len(all_data) - sequence_length
            sequence = []
            for j in range(sequence_length):
                sequence.append(all_data[remaining_start + j]['features'])
            sequences.append(np.array(sequence))
    
    print(f"Created {len(sequences)} sequences from training data")
    print(f"Each sequence has {sequence_length} time steps")
    print(f"Feature order: [temperature, speed, heart_beat]")
    
    return sequences


def _configure_cloudinary(cloud_name, api_key, api_secret):
    """Configure Cloudinary for model upload"""
    cloudinary.config(
        cloud_name=cloud_name,
        api_key=api_key,
        api_secret=api_secret
    )


def upload_model_to_cloudinary(model, cloud_name, api_key, api_secret, public_id):
    """
    Upload trained model to Cloudinary
    
    Args:
        model: Trained LSTM-OC-SVM model
        cloud_name, api_key, api_secret: Cloudinary credentials
        public_id: Public ID for the uploaded model
    
    Returns:
        Dictionary with success status and model URL
    """
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        joblib.dump(model, tmp.name)
        tmp_path = tmp.name

    try:
        _configure_cloudinary(cloud_name, api_key, api_secret)
        res = cloudinary.uploader.upload(
            tmp_path,
            resource_type="raw",
            public_id=public_id,
            overwrite=True
        )
        return {"success": True, "model_url": res["secure_url"]}
    except Exception as e:
        return {"success": False, "error": f"Upload failed: {e}"}
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def main():
    """
    Main training function implementing the complete pipeline:
    1. Load variable-length sequences from CSV
    2. Train LSTM-OC-SVM using joint optimization (Algorithm 2)
    3. Upload trained model to Cloudinary
    """
    
    if len(sys.argv) != 2:
        print("Usage: python train.py <input.csv>")
        print("CSV should have columns: timestamp, temperature, speed, heart_beat")
        sys.exit(1)

    csv_path = sys.argv[1]
    
    if not os.path.exists(csv_path):
        print(f"Error: File {csv_path} not found")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("LSTM-OC-SVM Anomaly Detection Training")
    print("Based on Ergen et al. (2017)")
    print("Expected CSV format: timestamp,heart_beat,temperature,speed")
    print("=" * 60)

    # Step 1: Load sequences from CSV
    print("\n1. Loading and processing training data...")
    print("Expected CSV format: timestamp,heart_beat,temperature,speed")
    try:
        sequences = load_sequences_from_csv(csv_path)
        if len(sequences) == 0:
            print("Error: No valid sequences found in CSV")
            sys.exit(1)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        print("Please check that your CSV has the correct format:")
        print("timestamp,heart_beat,temperature,speed")
        print("2025-04-26T14:10:17.868634,62,37.0,0.89")
        sys.exit(1)

    # Step 2: Configure model parameters based on the paper
    print("\n2. Configuring model parameters...")
    
    # Parameters from the paper and experiments section
    input_size = 3          # Features: temperature, speed, heart_beat
    hidden_size = 64        # LSTM hidden dimension (m in paper)
    n_lambda = 0.05         # Regularization parameter λ (from experiments)
    tau = 10.0              # Smoothing parameter τ (high value for good approximation)
    
    # Training parameters
    batch_size = 16         # Batch size for training
    epochs = 100            # Maximum epochs
    learning_rate = 0.001   # Learning rate μ from Algorithm 2
    convergence_threshold = 1e-6  # Convergence criterion ε
    
    print(f"Model config: input_size={input_size}, hidden_size={hidden_size}")
    print(f"OC-SVM config: lambda={n_lambda}, tau={tau}")
    print(f"Training config: batch_size={batch_size}, epochs={epochs}, lr={learning_rate}")

    # Step 3: Train LSTM-OC-SVM using Algorithm 2
    print("\n3. Training LSTM-OC-SVM model...")
    print("Implementing joint optimization of LSTM and OC-SVM parameters...")
    
    try:
        trained_model, trainer = train_lstm_ocsvm(
            sequences=sequences,
            input_size=input_size,
            hidden_size=hidden_size,
            n_lambda=n_lambda,
            tau=tau,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            convergence_threshold=convergence_threshold,
            orthogonal_update_freq=10  # Enforce constraints every 10 iterations
        )
        print("✓ Model training completed successfully!")
        
    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)

    # Step 4: Test the trained model
    print("\n4. Testing trained model...")
    try:
        # Test on a subset of training data to verify model works
        test_sequences = sequences[:min(10, len(sequences))]
        predictions, scores = predict_anomalies(trained_model, test_sequences)
        
        print(f"Test predictions shape: {predictions.shape}")
        print(f"Sample predictions: {predictions[:5]}")
        print(f"Sample scores: {scores[:5]}")
        print("✓ Model prediction test successful!")
        
    except Exception as e:
        print(f"Error during model testing: {e}")
        print("Model training completed but testing failed")

    # Step 5: Save model (locally and/or Cloudinary)
    print("\n5. Saving trained model...")
    
    # Always save locally first
    local_model_path = "trained_lstm_ocsvm_model.pkl"
    try:
        joblib.dump(trained_model, local_model_path)
        print(f"✅ Model saved locally: {local_model_path}")
        
        # Get model size
        model_size = os.path.getsize(local_model_path) / (1024 * 1024)  # MB
        print(f"📁 Model size: {model_size:.2f} MB")
        
    except Exception as e:
        print(f"❌ Failed to save model locally: {e}")
    
    # Check Cloudinary credentials
    load_dotenv()
    
    CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME")
    API_KEY = os.getenv("CLOUDINARY_API_KEY")
    API_SECRET = os.getenv("CLOUDINARY_API_SECRET")
    
    print(f"\n🔍 Checking Cloudinary credentials:")
    print(f"   CLOUDINARY_CLOUD_NAME: {'✅ Set' if CLOUD_NAME else '❌ Missing'}")
    print(f"   CLOUDINARY_API_KEY: {'✅ Set' if API_KEY else '❌ Missing'}")
    print(f"   CLOUDINARY_API_SECRET: {'✅ Set' if API_SECRET else '❌ Missing'}")

    if not all([CLOUD_NAME, API_KEY, API_SECRET]):
        print("\n⚠️  Cloudinary credentials incomplete")
        print("Model trained successfully and saved locally")
        print("\nTo enable Cloudinary upload:")
        print("1. Check that .env file exists in current directory")
        print("2. Verify .env file format:")
        print("   CLOUDINARY_CLOUD_NAME=your_cloud_name")
        print("   CLOUDINARY_API_KEY=your_api_key")
        print("   CLOUDINARY_API_SECRET=your_api_secret")
        print("3. Make sure python-dotenv is installed: pip install python-dotenv")
        print(f"\n🏠 Current working directory: {os.getcwd()}")
        print(f"📂 .env file exists: {os.path.exists('.env')}")
        return

    try:
        print("\n☁️  Uploading to Cloudinary...")
        result = upload_model_to_cloudinary(
            trained_model, 
            CLOUD_NAME, 
            API_KEY, 
            API_SECRET, 
            public_id="models/lstm_ocsvm_joint"
        )

        if result["success"]:
            print("✅ Model uploaded successfully to Cloudinary!")
            print(f"🌐 Model URL: {result['model_url']}")
            print(f"🆔 Public ID: models/lstm_ocsvm_joint")
        else:
            print(f"❌ Cloudinary upload failed: {result.get('error')}")
            print(f"✅ Model is still available locally at: {local_model_path}")
            
    except Exception as e:
        print(f"❌ Cloudinary upload error: {e}")
        print(f"✅ Model is still available locally at: {local_model_path}")

    print("\n" + "=" * 60)
    print("Training pipeline completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()