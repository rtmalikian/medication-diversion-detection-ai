import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import joblib
import logging
from datetime import datetime
import os
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

from data.processor import data_processor
from config import settings

# Configure logging
logger = logging.getLogger(__name__)

class DiversionDetectionModel:
    """
    Gradient Boosting model for detecting medication diversion
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_trained = False
        self.model_path = settings.model_path
        self.scaler_path = settings.scaler_path
    
    def load_model(self, model_path: Optional[str] = None):
        """
        Load a pre-trained model from disk
        """
        try:
            path = model_path or self.model_path
            if os.path.exists(path):
                self.model = joblib.load(path)
                self.is_trained = True
                logger.info(f"Model loaded from {path}")
                
                # Load scaler if it exists
                if os.path.exists(self.scaler_path):
                    self.scaler = joblib.load(self.scaler_path)
                    logger.info(f"Scaler loaded from {self.scaler_path}")
            else:
                logger.info("No pre-trained model found, initializing new model")
                self._initialize_default_model()
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self._initialize_default_model()
    
    def _initialize_default_model(self):
        """
        Initialize a default model
        """
        try:
            # Using Gradient Boosting as the primary algorithm
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
            self.is_trained = False
            logger.info("Default Gradient Boosting model initialized")
        except Exception as e:
            logger.error(f"Error initializing default model: {str(e)}")
            raise
    
    def train_new_model(self, 
                        data_source: str, 
                        test_size: float = 0.2, 
                        validation_size: float = 0.2,
                        hyperparameters: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Train a new model with the provided data
        """
        try:
            logger.info("Starting model training")
            
            # Get data - in a real implementation, this would load from the specified source
            # For now, we'll use synthetic data
            from data.loader import data_loader
            data = data_loader.load_synthetic_data(size=1000)
            
            # Process data into features
            X, y = self._prepare_training_data(data)
            
            # Split data into train/validation/test sets
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            # Further split training data to get validation set
            val_size = validation_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size, random_state=42, stratify=y_temp
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Set hyperparameters
            if hyperparameters is None:
                hyperparameters = {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 3,
                    'random_state': 42
                }
            
            # Initialize model with hyperparameters
            self.model = GradientBoostingClassifier(**hyperparameters)
            
            # Train the model
            self.model.fit(X_train_scaled, y_train)
            self.is_trained = True
            
            # Evaluate the model
            val_predictions = self.model.predict(X_val_scaled)
            val_probabilities = self.model.predict_proba(X_val_scaled)[:, 1]
            
            test_predictions = self.model.predict(X_test_scaled)
            test_probabilities = self.model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            val_auc = roc_auc_score(y_val, val_probabilities)
            test_auc = roc_auc_score(y_test, test_probabilities)
            
            # Create model info
            model_info = {
                'model_name': 'GradientBoosting_DiversionDetection',
                'version': '1.0.0',
                'features': X.shape[1],
                'created_at': datetime.now(),
                'accuracy': self.model.score(X_test_scaled, y_test),
                'auc_score': test_auc
            }
            
            # Calculate additional metrics
            metrics = {
                'val_auc': val_auc,
                'test_auc': test_auc,
                'val_accuracy': self.model.score(X_val_scaled, y_val),
                'test_accuracy': self.model.score(X_test_scaled, y_test),
                'training_samples': len(X_train),
                'validation_samples': len(X_val),
                'test_samples': len(X_test)
            }
            
            # Save the trained model
            self.save_model()
            
            logger.info(f"Model training completed. Test AUC: {test_auc:.4f}")
            return model_info, metrics
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
    
    def _prepare_training_data(self, patient_data: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Prepare training data from patient records
        """
        try:
            logger.info("Preparing training data")
            
            # Process each patient record into features
            processed_features = []
            labels = []  # This would come from training labels in a real implementation
            
            for patient in patient_data:
                # Process patient data
                features_df = data_processor.process_patient_data(patient)
                processed_features.append(features_df.values[0])  # Get the feature array
                
                # In a real implementation, labels would come from training data
                # For now, we'll synthesize labels based on risk indicators
                label = self._synthesize_label(patient)
                labels.append(label)
            
            # Convert to numpy arrays
            X = np.array(processed_features)
            y = np.array(labels)
            
            # Store feature names for later use
            self.feature_names = data_processor.feature_columns
            
            logger.info(f"Training data prepared: {X.shape[0]} samples, {X.shape[1]} features")
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing training data: {str(e)}")
            raise
    
    def _synthesize_label(self, patient: Dict[str, Any]) -> int:
        """
        Synthesize a label for training (0 = no diversion, 1 = diversion risk)
        This is a simplified approach - in practice, this would come from ground truth data
        """
        try:
            # Create synthetic labels based on risk factors
            risk_score = 0
            
            # Add points for various risk factors
            if patient.get('clinical_indicators', {}).get('history_of_diversion', False):
                risk_score += 3
            if patient.get('clinical_indicators', {}).get('substance_abuse_history', False):
                risk_score += 2
            if patient.get('opioid_prescriptions', 0) > 3:
                risk_score += 2
            if patient.get('benzodiazepine_prescriptions', 0) > 2:
                risk_score += 1
            if patient.get('early_refill_events', 0) > 1:
                risk_score += 1
            if patient.get('num_prescribers', 0) > 3:
                risk_score += 1
            if patient.get('num_pharmacies', 0) > 3:
                risk_score += 1
            if patient.get('concurrent_prescriptions', 0) > 2:
                risk_score += 2
            if patient.get('out_of_region_prescriptions', 0) > 1:
                risk_score += 1
            
            # Return 1 if risk score exceeds threshold, 0 otherwise
            return 1 if risk_score >= 4 else 0
            
        except Exception:
            return 0
    
    def predict_single(self, features: Dict[str, float]) -> Tuple[int, float]:
        """
        Make a prediction for a single patient
        """
        try:
            if not self.is_trained:
                logger.warning("Model not trained, initializing default model")
                self._initialize_default_model()
            
            # Convert features to the expected format
            feature_array = np.array([[features.get(col, 0) for col in self.feature_names]])
            
            # Scale features
            feature_scaled = self.scaler.transform(feature_array)
            
            # Make prediction
            prediction = self.model.predict(feature_scaled)[0]
            probability = self.model.predict_proba(feature_scaled)[0][1]  # Probability of positive class
            
            logger.info(f"Prediction: {prediction}, Probability: {probability:.4f}")
            return int(prediction), float(probability)
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise
    
    def predict_batch(self, features_list: List[Dict[str, float]]) -> List[Tuple[int, float]]:
        """
        Make predictions for multiple patients
        """
        try:
            if not self.is_trained:
                logger.warning("Model not trained, initializing default model")
                self._initialize_default_model()
            
            # Convert features list to array
            feature_array = np.array([
                [features.get(col, 0) for col in self.feature_names] 
                for features in features_list
            ])
            
            # Scale features
            feature_scaled = self.scaler.transform(feature_array)
            
            # Make predictions
            predictions = self.model.predict(feature_scaled)
            probabilities = self.model.predict_proba(feature_scaled)[:, 1]
            
            # Combine predictions and probabilities
            results = [(int(pred), float(prob)) for pred, prob in zip(predictions, probabilities)]
            
            logger.info(f"Batch prediction completed for {len(results)} samples")
            return results
            
        except Exception as e:
            logger.error(f"Error making batch predictions: {str(e)}")
            raise
    
    def evaluate_model(self) -> Dict[str, Any]:
        """
        Evaluate the current model's performance
        """
        try:
            if not self.is_trained:
                logger.warning("Model not trained, cannot evaluate")
                return {"error": "Model not trained"}
            
            # In a real implementation, we would evaluate on test data
            # For now, we'll return the model's feature importances
            feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
            
            evaluation = {
                "feature_importance": feature_importance,
                "n_features": len(self.feature_names),
                "model_type": type(self.model).__name__,
                "is_trained": self.is_trained
            }
            
            logger.info("Model evaluation completed")
            return evaluation
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise
    
    def save_model(self, model_path: Optional[str] = None, scaler_path: Optional[str] = None):
        """
        Save the trained model to disk
        """
        try:
            model_path = model_path or self.model_path
            scaler_path = scaler_path or self.scaler_path
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Save model
            joblib.dump(self.model, model_path)
            logger.info(f"Model saved to {model_path}")
            
            # Save scaler
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Scaler saved to {scaler_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model
        """
        try:
            model_info = {
                'model_name': type(self.model).__name__ if self.model else 'Uninitialized',
                'version': '1.0.0',
                'features': len(self.feature_names) if self.feature_names else 0,
                'created_at': datetime.now(),
                'is_trained': self.is_trained
            }
            
            if self.is_trained:
                model_info['feature_names'] = self.feature_names
                model_info['n_features'] = len(self.feature_names)
            
            return model_info
            
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            raise
    
    def hyperparameter_tuning(self, X_train, y_train, cv=5):
        """
        Perform hyperparameter tuning using GridSearchCV
        """
        try:
            logger.info("Starting hyperparameter tuning")
            
            # Define parameter grid for Gradient Boosting
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            }
            
            # Create a new model for tuning
            gb_model = GradientBoostingClassifier(random_state=42)
            
            # Perform grid search
            grid_search = GridSearchCV(
                estimator=gb_model,
                param_grid=param_grid,
                scoring='roc_auc',
                cv=cv,
                n_jobs=-1,
                verbose=1
            )
            
            # Fit the grid search
            grid_search.fit(X_train, y_train)
            
            # Update the model with the best parameters
            self.model = grid_search.best_estimator_
            self.is_trained = True
            
            logger.info(f"Best parameters: {grid_search.best_params_}")
            logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
            
            return grid_search.best_params_, grid_search.best_score_
            
        except Exception as e:
            logger.error(f"Error in hyperparameter tuning: {str(e)}")
            raise
    
    def get_feature_names(self) -> List[str]:
        """
        Get the list of feature names used by the model
        """
        return self.feature_names or []


# Global instance
diversion_model = DiversionDetectionModel()