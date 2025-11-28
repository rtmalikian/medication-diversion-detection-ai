from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime
import os

from models.diversion_detection import DiversionDetectionModel
from utils.logging import logger

router = APIRouter()

class ModelInfo(BaseModel):
    model_name: str
    version: str
    features: int
    created_at: datetime
    accuracy: Optional[float] = None
    auc_score: Optional[float] = None

class ModelTrainingRequest(BaseModel):
    data_source: str
    test_size: float = 0.2
    validation_size: float = 0.2
    hyperparameters: Dict[str, Any] = {}

class ModelTrainingResponse(BaseModel):
    model_info: ModelInfo
    training_time: float
    metrics: Dict[str, float]

class ModelPredictionRequest(BaseModel):
    features: Dict[str, float]

class ModelPredictionResponse(BaseModel):
    prediction: float
    probability: float
    explanation: Optional[Dict[str, Any]] = None

@router.get("/model-info", response_model=ModelInfo)
async def get_model_info():
    """
    Get information about the current ML model
    """
    try:
        logger.info("Fetching model information")
        model = DiversionDetectionModel()
        model.load_model()
        return model.get_model_info()
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/train-model", response_model=ModelTrainingResponse)
async def train_model(request: ModelTrainingRequest):
    """
    Train a new model with the provided parameters
    """
    try:
        logger.info("Starting model training")
        model = DiversionDetectionModel()
        
        # Train the model
        start_time = datetime.now()
        model_info, metrics = model.train_new_model(
            data_source=request.data_source,
            test_size=request.test_size,
            validation_size=request.validation_size,
            hyperparameters=request.hyperparameters
        )
        end_time = datetime.now()
        
        training_time = (end_time - start_time).total_seconds()
        
        response = ModelTrainingResponse(
            model_info=model_info,
            training_time=training_time,
            metrics=metrics
        )
        
        logger.info("Model training completed successfully")
        return response
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict", response_model=ModelPredictionResponse)
async def predict(request: ModelPredictionRequest):
    """
    Make a prediction using the current model
    """
    try:
        logger.info("Making prediction")
        model = DiversionDetectionModel()
        model.load_model()
        
        prediction, probability = model.predict_single(request.features)
        
        response = ModelPredictionResponse(
            prediction=prediction,
            probability=probability
        )
        
        logger.info(f"Prediction completed: {prediction} with probability {probability}")
        return response
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload-model")
async def upload_model(file: UploadFile = File(...)):
    """
    Upload a pre-trained model file
    """
    try:
        logger.info(f"Uploading model file: {file.filename}")
        
        # Save uploaded file to models directory
        upload_path = f"models/{file.filename}"
        with open(upload_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Load the new model
        model = DiversionDetectionModel()
        model.load_model(model_path=upload_path)
        
        return {
            "message": f"Model {file.filename} uploaded and loaded successfully",
            "filename": file.filename
        }
    except Exception as e:
        logger.error(f"Error uploading model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/model-performance")
async def get_model_performance():
    """
    Get the current model's performance metrics
    """
    try:
        logger.info("Fetching model performance")
        model = DiversionDetectionModel()
        model.load_model()
        
        performance = model.evaluate_model()
        return performance
    except Exception as e:
        logger.error(f"Error getting model performance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))