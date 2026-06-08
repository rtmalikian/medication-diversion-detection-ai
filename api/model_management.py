from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime
import os
import re
import secrets
import logging

from models.diversion_detection import DiversionDetectionModel

logger = logging.getLogger(__name__)

router = APIRouter()

# Security constants for model uploads
ALLOWED_MODEL_EXTENSIONS = {".pkl", ".joblib"}
MAX_UPLOAD_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB
UPLOAD_DIR = "models"


def _sanitize_filename(filename: str) -> str:
    """
    Sanitize an uploaded filename to prevent path traversal and injection.

    - Strips directory components (prevents ../ traversal)
    - Allows only alphanumeric, hyphens, underscores, dots
    - Rejects hidden files (starting with .)
    - Rejects empty or excessively long names
    """
    if not filename:
        raise ValueError("Filename cannot be empty")

    # Strip all directory components — only keep the basename
    basename = os.path.basename(filename)

    if not basename or basename.startswith("."):
        raise ValueError("Invalid filename: hidden files are not allowed")

    if len(basename) > 255:
        raise ValueError("Filename too long (max 255 characters)")

    # Allow only safe characters: alphanumeric, hyphen, underscore, dot
    if not re.match(r"^[a-zA-Z0-9_\-]+(\.[a-zA-Z0-9]+)+$", basename):
        raise ValueError(
            "Filename must contain only letters, numbers, hyphens, underscores, "
            "and a valid extension (e.g., model_v1.pkl)"
        )

    return basename

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
    Upload a pre-trained model file.

    Security measures applied:
    - Filename sanitization to prevent path traversal (e.g., ../../etc/passwd)
    - File extension allowlist (.pkl, .joblib only)
    - Maximum upload size limit (100 MB)
    - Randomized storage filename to prevent overwrites and naming conflicts

    ⚠️ WARNING: Model files are loaded via joblib (pickle), which can execute
    arbitrary code. Only upload model files from trusted sources. In production,
    consider using a safe serialization format (e.g., ONNX) or enforcing
    cryptographic signatures on uploaded models.
    """
    try:
        # Validate filename (prevents path traversal)
        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename is required")
        try:
            safe_name = _sanitize_filename(file.filename)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Validate file extension
        ext = os.path.splitext(safe_name)[1].lower()
        if ext not in ALLOWED_MODEL_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"File extension '{ext}' not allowed. "
                       f"Accepted: {', '.join(sorted(ALLOWED_MODEL_EXTENSIONS))}"
            )

        # Read content with size limit
        content = await file.read()
        if len(content) > MAX_UPLOAD_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds maximum upload size of "
                       f"{MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MB"
            )

        # Use a randomized filename to prevent overwrites and naming attacks
        random_prefix = secrets.token_hex(8)
        stored_name = f"{random_prefix}_{safe_name}"
        upload_path = os.path.join(UPLOAD_DIR, stored_name)

        # Ensure the upload directory exists
        os.makedirs(UPLOAD_DIR, exist_ok=True)

        # Write file
        with open(upload_path, "wb") as f:
            f.write(content)

        logger.info(
            "Model file uploaded: original='%s', stored='%s', size=%d bytes",
            file.filename, stored_name, len(content)
        )

        # Validate the uploaded model can be loaded
        try:
            model = DiversionDetectionModel()
            model.load_model(model_path=upload_path)
            logger.info("Uploaded model loaded and validated successfully")
        except Exception as load_err:
            # Clean up invalid file
            if os.path.exists(upload_path):
                os.remove(upload_path)
            logger.warning("Uploaded model failed validation: %s", load_err)
            raise HTTPException(
                status_code=422,
                detail="Uploaded file is not a valid model file and could not be loaded"
            )

        return {
            "message": f"Model uploaded and loaded successfully",
            "filename": stored_name
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error uploading model: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error during model upload")

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