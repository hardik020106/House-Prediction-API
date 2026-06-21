import io
import joblib
import pandas as pd
from fastapi import FastAPI,HTTPException,UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

app = FastAPI()

model = joblib.load('house_price_model.joblib')
feature_names = joblib.load('feature_names.joblib')

class HouseData(BaseModel):
    MedInc: float = Field(gt=0, description="Median income in block group")
    HouseAge: float = Field(ge=0, description="House age in years")
    AveRooms: float = Field(gt=0, description="Average number of rooms")
    AveBedrms: float = Field(gt=0, description="Average number of bedrooms")
    Population: float = Field(gt=0, description="Population in block group")
    AveOccup: float = Field(gt=0, description="Average number of occupants")
    Latitude: float = Field(ge=32, le=42, description="Latitude coordinate")
    Longitude: float = Field(ge=-125, le=-114, description="Longitude coordinate")

@app.get("/")
async def root():
    return {"message": "Welcome to the House Price Prediction API"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model": "RandomForestRegressor",
        "features": feature_names,
        "avg_error": "$32,000"

    }

@app.post("/predict")
async def predict(data: HouseData):
    input_data = pd.DataFrame([{
        "MedInc": data.MedInc,
        "HouseAge": data.HouseAge,
        "AveRooms": data.AveRooms,
        "AveBedrms": data.AveBedrms,
        "Population": data.Population,
        "AveOccup": data.AveOccup,
        "Latitude": data.Latitude,
        "Longitude": data.Longitude
    }])
    
    try:
        prediction = model.predict(input_data)[0]
        return {"predicted_price": f"${(prediction * 100000):,.2f}",
                "predicted_price_short": f"${(prediction):,.2f} hundred thousand dollars",
                "confidence_interval": f"${(prediction * 100000) - 32000:,.0f} - ${(prediction * 100000 + 32000):,.0f}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/predict-file")
async def predict_file(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a CSV file.")
    
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))  # ✅ Fixed
        
        if not all(feature in df.columns for feature in feature_names):
            missing = [feature for feature in feature_names if feature not in df.columns]
            raise HTTPException(status_code=400, detail=f"Missing features: {', '.join(missing)}")
        
        predictions = model.predict(df[feature_names])
        df['PredictedPrice'] = predictions * 100000
        output = df.to_csv(index=False)
        
        return StreamingResponse(
            io.StringIO(output),  # ✅ This works - StreamingResponse accepts StringIO
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=predictions.csv"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))