import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import joblib
from pathlib import Path

class ModelTrainer:
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestClassifier(random_state=42),
            'XGBoost': xgb.XGBClassifier(random_state=42),
            'LightGBM': lgb.LGBMClassifier(random_state=42)
        }
        self.current_model = None
        self.model_name = None
    
    def train_model(self, X_train, y_train, model_name='Random Forest', model_params=None):
        """Entraîne le modèle spécifié avec les paramètres donnés."""
        self.model_name = model_name
        
        # Création d'une nouvelle instance du modèle avec les paramètres spécifiés
        if model_params is None:
            model_params = {}
        
        base_model = self.models[model_name]
        self.current_model = type(base_model)(**{**base_model.get_params(), **model_params, 'random_state': 42})
        
        # Entraînement du modèle
        self.current_model.fit(X_train, y_train)
        return self.current_model
    
    def evaluate_model(self, X_test, y_test):
        """Évalue le modèle sur les données de test."""
        if self.current_model is None:
            raise ValueError("Aucun modèle n'a été entraîné.")
        
        y_pred = self.current_model.predict(X_test)
        y_pred_proba = self.current_model.predict_proba(X_test)[:, 1]
        
        results = {
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'roc_auc_score': roc_auc_score(y_test, y_pred_proba)
        }
        
        return results
    
    def predict(self, X):
        """Fait des prédictions sur de nouvelles données."""
        if self.current_model is None:
            raise ValueError("Aucun modèle n'a été entraîné.")
        
        return self.current_model.predict(X)
    
    def predict_proba(self, X):
        """Retourne les probabilités de prédiction."""
        if self.current_model is None:
            raise ValueError("Aucun modèle n'a été entraîné.")
        
        return self.current_model.predict_proba(X)[:, 1]
    
    def save_model(self, model_path):
        """Sauvegarde le modèle entraîné."""
        if self.current_model is None:
            raise ValueError("Aucun modèle n'a été entraîné.")
        
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.current_model, model_path)
    
    def load_model(self, model_path):
        """Charge un modèle sauvegardé."""
        self.current_model = joblib.load(model_path)
        return self.current_model 