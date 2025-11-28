import shap
import lime
import lime.lime_tabular
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Callable
import logging
from sklearn.ensemble import GradientBoostingClassifier
import joblib

from models.diversion_detection import diversion_model
from data.processor import data_processor

# Configure logging
logger = logging.getLogger(__name__)

class XAIExplainer:
    """
    Explainable AI (XAI) service using SHAP and LIME for model interpretability
    """
    
    def __init__(self):
        self.shap_explainer = None
        self.lime_explainer = None
        self.feature_names = None
    
    def initialize_explainers(self, model=None, background_data=None):
        """
        Initialize SHAP and LIME explainers
        """
        try:
            logger.info("Initializing XAI explainers")
            
            # Use provided model or load the default one
            if model is None:
                model = diversion_model.model
            
            # Get feature names
            self.feature_names = diversion_model.feature_names
            
            # Initialize SHAP explainer
            if background_data is not None:
                # Use the model's training data as background
                self.shap_explainer = shap.TreeExplainer(model)
            else:
                # For gradient boosting, we can still initialize the explainer
                self.shap_explainer = shap.TreeExplainer(model)
            
            # Initialize LIME explainer
            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=np.array([]),  # Will be provided during explanation
                feature_names=self.feature_names,
                class_names=['No Diversion', 'Diversion Risk'],
                mode='classification'
            )
            
            logger.info("XAI explainers initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing XAI explainers: {str(e)}")
            raise
    
    def explain_with_shap(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate SHAP explanation for a patient
        """
        try:
            logger.info(f"Generating SHAP explanation for patient {patient_data.get('patient_id', 'unknown')}")
            
            # Process patient data into features
            features_df = data_processor.process_patient_data(patient_data)
            feature_array = features_df.values
            
            # Calculate SHAP values
            shap_values = self.shap_explainer.shap_values(feature_array)
            
            # Handle both classes for binary classification
            if isinstance(shap_values, list):
                # For binary classification, use the positive class
                shap_values_for_explanation = shap_values[1]  # Positive class
            else:
                shap_values_for_explanation = shap_values
            
            # Create explanation dictionary
            shap_explanation = {
                'feature_importance': dict(zip(self.feature_names, shap_values_for_explanation[0])),
                'base_value': float(self.shap_explainer.expected_value) if hasattr(self.shap_explainer, 'expected_value') else 0.0,
                'explanation_type': 'SHAP',
                'feature_names': self.feature_names
            }
            
            # Sort features by absolute SHAP value to show most important first
            sorted_features = sorted(
                shap_explanation['feature_importance'].items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            shap_explanation['sorted_feature_importance'] = sorted_features
            
            logger.info(f"SHAP explanation generated for patient {patient_data.get('patient_id', 'unknown')}")
            return shap_explanation
            
        except Exception as e:
            logger.error(f"Error generating SHAP explanation: {str(e)}")
            raise
    
    def explain_with_lime(self, patient_data: Dict[str, Any], model_predict_fn: Callable = None) -> Dict[str, Any]:
        """
        Generate LIME explanation for a patient
        """
        try:
            logger.info(f"Generating LIME explanation for patient {patient_data.get('patient_id', 'unknown')}")
            
            # Process patient data into features
            features_df = data_processor.process_patient_data(patient_data)
            feature_array = features_df.values[0]  # Get single row as array
            
            # Use the model's predict_proba function if not provided
            if model_predict_fn is None:
                def predict_fn(x):
                    # Reshape if needed
                    if len(x.shape) == 1:
                        x = x.reshape(1, -1)
                    return diversion_model.model.predict_proba(x)
                
                model_predict_fn = predict_fn
            
            # Create LIME explainer for this specific instance
            lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=np.array([feature_array]),  # Use the patient's data as reference
                feature_names=self.feature_names,
                class_names=['No Diversion', 'Diversion Risk'],
                mode='classification'
            )
            
            # Generate LIME explanation
            exp = lime_explainer.explain_instance(
                data_row=feature_array,
                predict_fn=model_predict_fn,
                num_features=len(self.feature_names)
            )
            
            # Extract explanation data
            lime_explanation = {
                'feature_importance': dict(exp.as_list()),
                'explanation_type': 'LIME',
                'feature_names': self.feature_names
            }
            
            # Sort features by absolute importance
            sorted_features = sorted(
                lime_explanation['feature_importance'].items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            lime_explanation['sorted_feature_importance'] = sorted_features
            
            logger.info(f"LIME explanation generated for patient {patient_data.get('patient_id', 'unknown')}")
            return lime_explanation
            
        except Exception as e:
            logger.error(f"Error generating LIME explanation: {str(e)}")
            raise
    
    def generate_combined_explanation(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate combined explanation using both SHAP and LIME
        """
        try:
            logger.info(f"Generating combined XAI explanation for patient {patient_data.get('patient_id', 'unknown')}")
            
            # Generate both explanations
            shap_exp = self.explain_with_shap(patient_data)
            lime_exp = self.explain_with_lime(patient_data)
            
            # Combine explanations
            combined_explanation = {
                'patient_id': patient_data.get('patient_id'),
                'shap_explanation': shap_exp,
                'lime_explanation': lime_exp,
                'agreement_score': self._calculate_agreement_score(shap_exp, lime_exp),
                'combined_insights': self._combine_insights(shap_exp, lime_exp)
            }
            
            logger.info(f"Combined XAI explanation generated for patient {patient_data.get('patient_id', 'unknown')}")
            return combined_explanation
            
        except Exception as e:
            logger.error(f"Error generating combined explanation: {str(e)}")
            raise
    
    def _calculate_agreement_score(self, shap_exp: Dict[str, Any], lime_exp: Dict[str, Any]) -> float:
        """
        Calculate agreement score between SHAP and LIME explanations
        """
        try:
            # Extract feature importances
            shap_importance = shap_exp['feature_importance']
            lime_importance = lime_exp['feature_importance']
            
            # Calculate correlation between the two methods
            common_features = set(shap_importance.keys()).intersection(set(lime_importance.keys()))
            
            if not common_features:
                return 0.0
            
            # Calculate agreement based on rank correlation
            shap_values = [shap_importance[f] for f in common_features]
            lime_values = [lime_importance[f] for f in common_features]
            
            # Simple agreement calculation (could be more sophisticated)
            agreement = 0
            total_features = len(common_features)
            
            for feature in common_features:
                shap_val = shap_importance[feature]
                lime_val = lime_importance[feature]
                
                # Check if both methods agree on direction (positive/negative impact)
                if (shap_val >= 0) == (lime_val >= 0):
                    agreement += 1
            
            agreement_score = agreement / total_features if total_features > 0 else 0.0
            return agreement_score
            
        except Exception:
            return 0.0
    
    def _combine_insights(self, shap_exp: Dict[str, Any], lime_exp: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combine insights from SHAP and LIME explanations
        """
        try:
            insights = {
                'top_features_shap': shap_exp.get('sorted_feature_importance', [])[:5],
                'top_features_lime': lime_exp.get('sorted_feature_importance', [])[:5],
                'common_top_features': [],
                'explanation_summary': ''
            }
            
            # Find common top features
            shap_top = [f[0] for f in shap_exp.get('sorted_feature_importance', [])[:5]]
            lime_top = [f[0] for f in lime_exp.get('sorted_feature_importance', [])[:5]]
            
            insights['common_top_features'] = list(set(shap_top).intersection(set(lime_top)))
            
            # Create summary
            summary_parts = []
            if insights['common_top_features']:
                summary_parts.append(f"Both SHAP and LIME identify these features as important: {', '.join(insights['common_top_features'][:3])}")
            else:
                summary_parts.append("SHAP and LIME highlight different important features.")
            
            insights['explanation_summary'] = " ".join(summary_parts)
            
            return insights
            
        except Exception as e:
            logger.error(f"Error combining insights: {str(e)}")
            return {'error': str(e)}


class XAIService:
    """
    Main XAI service for the clinical decision support tool
    """
    
    def __init__(self):
        self.explainer = XAIExplainer()
    
    def initialize(self):
        """
        Initialize the XAI service
        """
        try:
            logger.info("Initializing XAI Service")
            self.explainer.initialize_explainers()
        except Exception as e:
            logger.error(f"Error initializing XAI service: {str(e)}")
            raise
    
    def generate_explanation(self, patient_data: Dict[str, Any], method: str = 'combined') -> Dict[str, Any]:
        """
        Generate explanation for a patient using specified method
        """
        try:
            if method == 'shap':
                return self.explainer.explain_with_shap(patient_data)
            elif method == 'lime':
                return self.explainer.explain_with_lime(patient_data)
            elif method == 'combined':
                return self.explainer.generate_combined_explanation(patient_data)
            else:
                raise ValueError(f"Unknown explanation method: {method}")
                
        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            raise


# Global instance
xai_service = XAIService()