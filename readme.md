# Anomaly Detection API

This is a RESTful API built with Flask for detecting anomalies in user data. It supports receiving data via POST requests and retrieving anomalies via GET requests.

## Setup

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Make sure your Firebase credentials file (`firebase-adminsdk.json`) is in the project root.

3. For model storage, set up Cloudinary credentials in a `.env` file:

```
CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_api_key
CLOUDINARY_API_SECRET=your_api_secret
```

4. Run the Flask application:

```bash
python app.py
```

## API Endpoints

### Submit User Data

```
POST /api/data
```

Submit user data which will be stored and analyzed for anomalies.

**Request Body:**

```json
{
  "user_id": "user123",
  "temperature": 36.8,
  "speed": 5.0,
  "heart_beat": 90,
  "altitude": 100,
  "longitude": 12.345,
  "latitude": 45.678
}
```

**Required fields:** `user_id`, `temperature`, `speed`, `heart_beat`

**Response:**

```json
{
  "success": true,
  "message": "Data received and stored",
  "anomaly_detected": true,
  "data_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

### Get All Anomalies for a User

```
GET /api/anomalies/{user_id}
```

Retrieve all anomalies detected for a specific user.

**Optional Query Parameters:**
- `reviewed=true|false` - Filter by review status

**Response:**

```json
{
  "success": true,
  "user_id": "user123",
  "anomalies": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "temperature": 39.5,
      "speed": 12.3,
      "heart_beat": 150,
      "timestamp": "2025-04-20T12:00:00.000Z",
      "is_reviewed": false
    },
    ...
  ],
  "count": 3
}
```

### Get Specific Anomaly Details

```
GET /api/anomalies/{user_id}/{anomaly_id}
```

Retrieve details for a specific anomaly.

**Response:**

```json
{
  "success": true,
  "anomaly": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "temperature": 39.5,
    "speed": 12.3,
    "heart_beat": 150,
    "timestamp": "2025-04-20T12:00:00.000Z",
    "is_reviewed": false
  }
}
```

### Update Anomaly Review Status

```
PATCH /api/anomalies/{user_id}/{anomaly_id}
```

Mark an anomaly as reviewed or unreviewed.

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

## Error Handling

All endpoints return appropriate HTTP status codes:

- `200` - Success
- `201` - Resource created successfully
- `400` - Bad request (e.g., missing required fields)
- `404` - Resource not found
- `500` - Server error

Error responses include a descriptive message:

```json
{
  "error": "Missing required field: heart_beat"
}
```

## How It Works

1. When data is submitted via POST, the API:
   - Stores the data in Firestore
   - Uses an Isolation Forest model to detect anomalies
   - Flags and stores any anomalous data in a separate collection

2. The GET endpoints allow you to:
   - Retrieve all anomalies for a user
   - Get details for a specific anomaly
   - Mark anomalies as reviewed

3. Models are:
   - Trained on user data
   - Stored in Cloudinary for future use
   - Automatically retrained when necessary