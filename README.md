# House Prediction API

A FastAPI-based machine learning API for predicting California house prices using a trained Random Forest Regressor model.

## Overview

This project takes California housing features as input and returns house price predictions through a FAST API. It also supports CSV file uploads for batch predictions.

The API is built with **FastAPI**, uses a saved **joblib** model, and provides both single prediction and bulk prediction endpoints.

## Features

- Predict house prices from structured JSON input.
- Predict prices for multiple records using CSV upload.
- Health check endpoint to verify API and model status.
- Input validation using Pydantic.
- Fast, interactive API documentation with Swagger UI.
- Batch prediction output downloadable as CSV.

## Project Structure

```bash
HOUSE-PREDICTION-API/
├── requirements.txt
├── explore.py
├── main.py
├── train.py
└── README.md
```

## Tech Stack

- **Python**
- **FastAPI**
- **Pandas**
- **Scikit-learn**
- **Joblib**
- **Pydantic**
- **Uvicorn**

## Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/house-prediction-api.git
cd house-prediction-api
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

**Windows:**
```bash
.venv\Scripts\activate
```

**Mac/Linux:**
```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the FastAPI app

```bash
uvicorn main:app --reload
```

## API Endpoints

### 1. Root Endpoint

**GET** `/`

Returns a welcome message.

**Response Example**
```json
{
  "message": "Welcome to the House Price Prediction API"
}
```

---

### 2. Health Check

**GET** `/health`

Checks whether the API and model are working properly.

**Response Example**
```json
{
  "status": "healthy",
  "model": "RandomForestRegressor",
  "features": ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"],
  "avg_error": "$32,000"
}
```

---

### 3. Single House Price Prediction

**POST** `/predict`

Predicts the house price for a single input record.

#### Request Body
```json
{
  "MedInc": 8.3252,
  "HouseAge": 41,
  "AveRooms": 6.9841,
  "AveBedrms": 1.0238,
  "Population": 322,
  "AveOccup": 2.5556,
  "Latitude": 37.88,
  "Longitude": -122.23
}
```

#### Response Example
```json
{
  "predicted_price": "$452,000.00",
  "predicted_price_short": "$4.52 hundred thousand dollars",
  "confidence_interval": "$420,000 - $484,000"
}
```

#### Notes
- The API validates all fields before prediction.
- Latitude must be between `32` and `42`.
- Longitude must be between `-125` and `-114`.
- Prediction is returned in a human-readable currency format.

---

### 4. Batch Prediction Using CSV

**POST** `/predict-file`

Uploads a CSV file and returns predictions for all rows.

#### Requirements
- File must be a `.csv`
- CSV must contain all required feature columns:
  - `MedInc`
  - `HouseAge`
  - `AveRooms`
  - `AveBedrms`
  - `Population`
  - `AveOccup`
  - `Latitude`
  - `Longitude`

#### Response
- A CSV file is returned with a new column:
  - `PredictedPrice`

#### Example Usage with cURL
```bash
curl -X POST "http://127.0.0.1:8000/predict-file" \
  -F "file=@sample_data.csv" \
  --output predictions.csv
```

## Input Features

| Feature | Description |
|--------|-------------|
| MedInc | Median income in block group |
| HouseAge | House age in years |
| AveRooms | Average number of rooms |
| AveBedrms | Average number of bedrooms |
| Population | Population in block group |
| AveOccup | Average number of occupants |
| Latitude | Latitude coordinate |
| Longitude | Longitude coordinate |

## Example Request with Python

```python
import requests

url = "http://127.0.0.1:8000/predict"
data = {
    "MedInc": 8.3252,
    "HouseAge": 41,
    "AveRooms": 6.9841,
    "AveBedrms": 1.0238,
    "Population": 322,
    "AveOccup": 2.5556,
    "Latitude": 37.88,
    "Longitude": -122.23
}

response = requests.post(url, json=data)
print(response.json())
```

## API Docs

After running the app, open:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

## Model Files

The project uses the following saved files:

- `house_price_model.joblib` — trained ML model
- `feature_names.joblib` — expected feature columns

## Notes

- The `/predict` endpoint multiplies the model output by `100000` before formatting the price.
- The `/predict-file` endpoint appends predictions to the uploaded dataset and returns a downloadable CSV.
- `__pycache__` and `.venv` should not be pushed to GitHub.

## .gitignore

Use this `.gitignore` file:

```gitignore
__pycache__/
*.pyc
.venv/
env/
venv/
.ipynb_checkpoints/
```

## requirements.txt

```txt
fastapi
uvicorn
pandas
scikit-learn
joblib
python-multipart
```

## Author

**Hardik Sikka**

## License

This project is open-source and available under the MIT License.
