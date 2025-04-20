# Health Monitoring API

A Flask-based RESTful API for mobile health monitoring applications. This API supports training user-specific machine learning models for anomaly detection in health data, making predictions, and managing anomalous records.

## Features

- **User-specific model training** using Isolation Forest for anomaly detection
- **Model storage and retrieval** via Cloudinary
- **Data storage** in Firebase Firestore
- **Real-time predictions** for health monitoring data
- **Anomaly tracking and management**

## Prerequisites

- Python 3.8+
- Firebase account with Firestore enabled
- Cloudinary account
- Firebase Admin SDK credentials

## Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/health-monitoring-api.git
cd health-monitoring-api
```

2. **Create and activate a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Set up environment variables**

Copy the example environment file and fill in your Cloudinary credentials.

```bash
cp .env.example .env
```

Edit the `.env` file with your Cloudinary information:

```
CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_api_key
CLOUDINARY_API_SECRET=your_api_secret
```

5. **Add Firebase credentials**

Place your Firebase Admin SDK service account credentials in a file named `firebase-adminsdk.json` in the project root directory.

6. **Run the application**

```bash
python app.py
```

The API will be available at `http://localhost:5000`.

## Deployment with Docker

You can also use Docker to deploy the application:

```bash
docker build -t health-monitoring-api .
docker run -p 5000:5000 -v $(pwd)/firebase-adminsdk.json:/app/firebase-adminsdk.json --env-file .env health-monitoring-api
```

## API Endpoints

### Train Model

**Endpoint:** `POST /api/train`

Trains a model for a specific user using data from Firebase or provided in the request.

**Request Body:**

```json
{
  "user_id": "user123",
  "use_firebase": true,  // Optional, defaults to true
  "training_data": [     // Optional if use_firebase is true
    {
      "temperature": 36.8,
      "speed": 5.0,
      "heart_beat": 90
    },
    // More records...
  ]
}
```

**Response:**

```json
{
  "success": true,
  "message": "Model trained successfully for user user123",
  "model_url": "https://res.cloudinary.com/yourcloud/raw/upload/models/user123.pkl",
  "records_used": 42
}
```

### Make Predictions

**Endpoint:** `POST /api/predict`

Makes predictions for new data records, using the user's trained model.

**Request Body:**

```json
{
  "user_id": "user123",
  "data": [
    {
      "id": "record1",  // Optional, will be generated if not provided
      "temperature": 37.2,
      "speed": 8.5,
      "heart_beat": 120,
      "timestamp": "2023-01-01T12:00:00Z",  // Optional
      "additional_field": "value"  // Optional additional fields
    },
    // More records...
  ]
}
```

**Response:**

```json
{
  "success": true,
  "user_id": "user123",
  "predictions": [
    {
      "id": "record1",
      "is_anomaly": true,
      "prediction": -1,
      "timestamp": "2023-01-01T12:00:00Z"
    },
    // More predictions...
  ],
  "records_processed": 2,
  "anomalies_detected": 1,
  "anomalies": [
    {
      "id": "record1",
      "temperature": 37.2,
      "speed": 8.5,
      "heart_beat": 120,
      "timestamp": "2023-01-01T12:00:00Z"
    }
  ]
}
```

### Get Anomalies

**Endpoint:** `GET /api/anomalies/{user_id}`

Retrieves anomalies for a specific user.

**Query Parameters:**
- `reviewed` - Filter by review status (`true` or `false`)
- `limit` - Maximum number of anomalies to return (default: 100)
- `offset` - Number of anomalies to skip for pagination (default: 0)

**Response:**

```json
{
  "success": true,
  "user_id": "user123",
  "anomalies": [
    {
      "id": "record1",
      "temperature": 37.2,
      "speed": 8.5,
      "heart_beat": 120,
      "timestamp": "2023-01-01T12:00:00Z",
      "is_reviewed": false,
      "data_id": "record1"
    },
    // More anomalies...
  ],
  "count": 5,
  "limit": 100,
  "offset": 0
}
```

### Get Anomaly Details

**Endpoint:** `GET /api/anomalies/{user_id}/{anomaly_id}`

Retrieves details for a specific anomaly.

**Response:**

```json
{
  "success": true,
  "anomaly": {
    "id": "record1",
    "temperature": 37.2,
    "speed": 8.5,
    "heart_beat": 120,
    "timestamp": "2023-01-01T12:00:00Z",
    "is_reviewed": false,
    "data_id": "record1"
  }
}
```

### Update Anomaly Status

**Endpoint:** `PATCH /api/anomalies/{user_id}/{anomaly_id}`

Updates the review status of an anomaly.

**Request Body:**

```json
{
  "is_reviewed": true
}
```

**Response:**

```json
{
  "success": true,
  "message": "Anomaly review status updated"
}
```

### Get Model Information

**Endpoint:** `GET /api/model-info/{user_id}`

Retrieves information about a user's trained model.

**Response:**

```json
{
  "success": true,
  "exists": true,
  "model_info": {
    "trained_at": "2023-01-01T12:00:00Z",
    "records_count": 42,
    "model_url": "https://res.cloudinary.com/yourcloud/raw/upload/models/user123.pkl"
  }
}
```

## Error Handling

All endpoints return appropriate HTTP status codes:

- `200` - Success
- `400` - Bad request (e.g., missing required fields)
- `404` - Resource not found
- `500` - Server error

Error responses include a descriptive message:

```json
{
  "error": "Failed to load model for user user123. Model may not exist or training may be required."
}
```

## Integration with Android

### Android Studio Integration

Here's a simple example of how to integrate with this API from an Android application using Retrofit:

```kotlin
// Define the API interface
interface HealthMonitoringApi {
    @POST("api/train")
    suspend fun trainModel(@Body trainRequest: TrainRequest): Response<TrainResponse>
    
    @POST("api/predict")
    suspend fun predict(@Body predictRequest: PredictRequest): Response<PredictResponse>
    
    @GET("api/anomalies/{userId}")
    suspend fun getAnomalies(
        @Path("userId") userId: String,
        @Query("reviewed") reviewed: Boolean? = null
    ): Response<AnomaliesResponse>
}

// Setup Retrofit
val retrofit = Retrofit.Builder()
    .baseUrl("https://your-api-url.com/")
    .addConverterFactory(GsonConverterFactory.create())
    .build()

val api = retrofit.create(HealthMonitoringApi::class.java)

// Example usage
lifecycleScope.launch {
    try {
        // Train model
        val trainResponse = api.trainModel(TrainRequest(userId = "user123"))
        
        // Make predictions
        val predictResponse = api.predict(PredictRequest(
            userId = "user123",
            data = listOf(
                HealthData(
                    temperature = 37.2f,
                    speed = 8.5f,
                    heartBeat = 120
                )
            )
        ))
        
        // Get anomalies
        val anomaliesResponse = api.getAnomalies(userId = "user123")
    } catch (e: Exception) {
        Log.e("API", "Error: ${e.message}")
    }
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.