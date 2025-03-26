import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.smote = SMOTE(random_state=42)
    
    def load_data(self, file_path):
        """Charge les données depuis un fichier CSV."""
        return pd.read_csv(file_path)
    
    def preprocess_data(self, df, target_column='Class'):
        """Prétraite les données pour l'entraînement."""
        # Séparation features et target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Normalisation des features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        return X_scaled, y
    
    def apply_smote(self, X, y):
        """Applique SMOTE pour rééquilibrer les données."""
        X_resampled, y_resampled = self.smote.fit_resample(X, y)
        return X_resampled, y_resampled
    
    def split_data(self, X, y, test_size=0.2):
        """Divise les données en ensembles d'entraînement et de test."""
        return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    def prepare_single_prediction(self, transaction_data):
        """Prépare une seule transaction pour la prédiction."""
        # Conversion en DataFrame si nécessaire
        if isinstance(transaction_data, dict):
            transaction_data = pd.DataFrame([transaction_data])
        
        # Application du même scaling que pour l'entraînement
        scaled_data = self.scaler.transform(transaction_data)
        return pd.DataFrame(scaled_data, columns=transaction_data.columns) 