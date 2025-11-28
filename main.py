from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from typing import Optional, List
from datetime import datetime

# Import API routes
from api.risk_assessment import router as risk_assessment_router
from api.patient_data import router as patient_data_router
from api.model_management import router as model_management_router

# Initialize FastAPI app
app = FastAPI(
    title="Clinical Risk Modeling Engine",
    description="Medication Diversion Detection System",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(risk_assessment_router, prefix="/api/v1", tags=["Risk Assessment"])
app.include_router(patient_data_router, prefix="/api/v1", tags=["Patient Data"])
app.include_router(model_management_router, prefix="/api/v1", tags=["Model Management"])

@app.get("/")
async def root():
    return {
        "message": "Clinical Risk Modeling Engine for Medication Diversion Detection",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)