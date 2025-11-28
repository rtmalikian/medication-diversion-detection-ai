from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    # Database settings
    database_url: str = "sqlite:///./diversion_detection.db"
    
    # Model settings
    model_path: str = "models/diversion_model.pkl"
    scaler_path: str = "models/scaler.pkl"
    
    # API settings
    api_key: Optional[str] = None
    debug: bool = False
    
    # Feature settings
    max_patients: int = 3000
    risk_threshold: float = 0.7  # Risk score threshold for alerts
    
    class Config:
        env_file = ".env"


def get_settings():
    return Settings()


settings = get_settings()