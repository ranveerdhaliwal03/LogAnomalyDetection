"""
HDFS Block Anomaly Detection Pipeline

This module provides functionality for detecting anomalous HDFS log block sequences
using TF-IDF vectorization of template sequences and metadata features.
"""

import pandas as pd
import numpy as np
import pickle
import os
from typing import Tuple, Dict, Any, Optional
from datetime import datetime

# Sklearn imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, f1_score, precision_recall_curve
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    print("Warning: XGBoost not available. Install with: pip install xgboost")
    XGBOOST_AVAILABLE = False

# Optional: LightGBM as alternative
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class BlockAnomalyDetector:
    """
    Main class for HDFS block anomaly detection.
    
    Combines TF-IDF features from template sequences with metadata features
    to train supervised classifiers for anomaly detection.
    """
    
    def __init__(self, 
                 tfidf_max_features: int = 1000,
                 tfidf_ngram_range: Tuple[int, int] = (1, 1),
                 random_state: int = 42):
        """
        Initialize the anomaly detector.
        
        Args:
            tfidf_max_features: Maximum number of TF-IDF features
            tfidf_ngram_range: N-gram range for TF-IDF (default unigrams)
            random_state: Random seed for reproducibility
        """
        self.tfidf_max_features = tfidf_max_features
        self.tfidf_ngram_range = tfidf_ngram_range
        self.random_state = random_state
        
        # Components that will be fitted
        self.tfidf_vectorizer = None
        self.metadata_scaler = None
        self.models = {}
        self.feature_names = []
        
        # Training data info
        self.training_stats = {}
        
    def preprocess_features(self, df: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """
        Preprocess the input dataframe to create feature matrix.
        
        Args:
            df: Input dataframe with required columns
            fit: Whether to fit the preprocessors (True for training, False for inference)
            
        Returns:
            Combined feature matrix (TF-IDF + metadata)
        """
        required_cols = ['template_sequence_str', 'sequence_length', 'time_span']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # 1. TF-IDF on template sequences
        if fit or self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=self.tfidf_max_features,
                ngram_range=self.tfidf_ngram_range,
                token_pattern=r'\d+',  # Match template IDs (numbers)
                lowercase=False,
                stop_words=None
            )
            tfidf_features = self.tfidf_vectorizer.fit_transform(df['template_sequence_str'])
        else:
            tfidf_features = self.tfidf_vectorizer.transform(df['template_sequence_str'])
        
        # 2. Metadata features
        metadata_features = df[['sequence_length', 'time_span']].values
        
        if fit or self.metadata_scaler is None:
            self.metadata_scaler = StandardScaler()
            metadata_scaled = self.metadata_scaler.fit_transform(metadata_features)
        else:
            metadata_scaled = self.metadata_scaler.transform(metadata_features)
        
        # 3. Combine features
        # Convert TF-IDF sparse matrix to dense for combination
        tfidf_dense = tfidf_features.toarray()
        combined_features = np.hstack([tfidf_dense, metadata_scaled])
        
        # Store feature names for interpretability
        if fit:
            tfidf_names = [f"tfidf_{name}" for name in self.tfidf_vectorizer.get_feature_names_out()]
            metadata_names = ['sequence_length_scaled', 'time_span_scaled']
            self.feature_names = tfidf_names + metadata_names
        
        return combined_features
    
    def train_models(self, 
                     df: pd.DataFrame,
                     test_size: float = 0.2,
                     validation_size: float = 0.1) -> Dict[str, Any]:
        """
        Train multiple models on the dataset.
        
        Args:
            df: Training dataframe
            test_size: Proportion of data for testing
            validation_size: Proportion of training data for validation
            
        Returns:
            Dictionary containing training results and metrics
        """
        # Validate input
        if 'anomaly_binary_val' not in df.columns:
            raise ValueError("Target column 'anomaly_binary_val' not found")
        
        print(f"Training on {len(df)} samples...")
        print(f"Anomaly distribution: {df['anomaly_binary_val'].value_counts().to_dict()}")
        
        # Preprocess features
        X = self.preprocess_features(df, fit=True)
        y = df['anomaly_binary_val'].values
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Further split training for validation
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train, y_train, test_size=validation_size, 
            random_state=self.random_state, stratify=y_train
        )
        
        print(f"Training set: {len(X_train_final)} samples")
        print(f"Validation set: {len(X_val)} samples") 
        print(f"Test set: {len(X_test)} samples")
        
        results = {}
        
        # 1. Logistic Regression (baseline)
        print("\n=== Training Logistic Regression ===")
        lr_model = LogisticRegression(
            random_state=self.random_state,
            class_weight='balanced',  # Handle class imbalance
            max_iter=1000
        )
        lr_model.fit(X_train_final, y_train_final)
        self.models['logistic_regression'] = lr_model
        
        lr_results = self._evaluate_model(lr_model, X_val, y_val, X_test, y_test, "Logistic Regression")
        results['logistic_regression'] = lr_results
        
        # 2. XGBoost (if available)
        if XGBOOST_AVAILABLE:
            print("\n=== Training XGBoost ===")
            # Calculate scale_pos_weight for class imbalance
            pos_weight = (y_train_final == 0).sum() / (y_train_final == 1).sum()
            
            xgb_model = xgb.XGBClassifier(
                random_state=self.random_state,
                scale_pos_weight=pos_weight,
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                eval_metric='logloss'
            )
            xgb_model.fit(X_train_final, y_train_final)
            self.models['xgboost'] = xgb_model
            
            xgb_results = self._evaluate_model(xgb_model, X_val, y_val, X_test, y_test, "XGBoost")
            results['xgboost'] = xgb_results
        
        # 3. LightGBM (alternative if XGBoost not available)
        elif LIGHTGBM_AVAILABLE:
            print("\n=== Training LightGBM ===")
            pos_weight = (y_train_final == 0).sum() / (y_train_final == 1).sum()
            
            lgb_model = lgb.LGBMClassifier(
                random_state=self.random_state,
                class_weight='balanced',
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                verbose=-1
            )
            lgb_model.fit(X_train_final, y_train_final)
            self.models['lightgbm'] = lgb_model
            
            lgb_results = self._evaluate_model(lgb_model, X_val, y_val, X_test, y_test, "LightGBM")
            results['lightgbm'] = lgb_results
        
        # Store training statistics
        self.training_stats = {
            'total_samples': len(df),
            'train_samples': len(X_train_final),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'anomaly_rate': y.mean(),
            'feature_count': X.shape[1],
            'tfidf_features': len(self.tfidf_vectorizer.get_feature_names_out()),
            'training_time': datetime.now().isoformat()
        }
        
        return results
    
    def _evaluate_model(self, model, X_val, y_val, X_test, y_test, model_name: str) -> Dict[str, Any]:
        """Evaluate a single model on validation and test sets."""
        results = {'model_name': model_name}
        
        # Validation predictions
        y_val_pred = model.predict(X_val)
        y_val_proba = model.predict_proba(X_val)[:, 1]
        
        # Test predictions  
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)[:, 1]
        
        # Validation metrics
        results['val_f1'] = f1_score(y_val, y_val_pred)
        results['val_roc_auc'] = roc_auc_score(y_val, y_val_proba)
        results['val_classification_report'] = classification_report(y_val, y_val_pred)
        
        # Test metrics
        results['test_f1'] = f1_score(y_test, y_test_pred)
        results['test_roc_auc'] = roc_auc_score(y_test, y_test_proba)
        results['test_classification_report'] = classification_report(y_test, y_test_pred)
        results['test_confusion_matrix'] = confusion_matrix(y_test, y_test_pred)
        
        # Store predictions for plotting
        results['test_y_true'] = y_test
        results['test_y_pred'] = y_test_pred  
        results['test_y_proba'] = y_test_proba
        
        print(f"\n{model_name} Results:")
        print(f"Validation F1: {results['val_f1']:.4f}, ROC-AUC: {results['val_roc_auc']:.4f}")
        print(f"Test F1: {results['test_f1']:.4f}, ROC-AUC: {results['test_roc_auc']:.4f}")
        print(f"\nTest Set Classification Report:\n{results['test_classification_report']}")
        
        return results
    
    def predict(self, df: pd.DataFrame, model_name: str = 'logistic_regression') -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new data.
        
        Args:
            df: Input dataframe
            model_name: Name of model to use for predictions
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not trained. Available models: {list(self.models.keys())}")
        
        X = self.preprocess_features(df, fit=False)
        model = self.models[model_name]
        
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]
        
        return predictions, probabilities
    
    def save_models(self, save_dir: str = 'saved_models'):
        """Save trained models and preprocessors."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save preprocessors
        with open(os.path.join(save_dir, 'tfidf_vectorizer.pkl'), 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)
        
        with open(os.path.join(save_dir, 'metadata_scaler.pkl'), 'wb') as f:
            pickle.dump(self.metadata_scaler, f)
        
        # Save models
        for name, model in self.models.items():
            with open(os.path.join(save_dir, f'{name}_model.pkl'), 'wb') as f:
                pickle.dump(model, f)
        
        # Save feature names and stats
        with open(os.path.join(save_dir, 'feature_names.pkl'), 'wb') as f:
            pickle.dump(self.feature_names, f)
            
        with open(os.path.join(save_dir, 'training_stats.pkl'), 'wb') as f:
            pickle.dump(self.training_stats, f)
        
        print(f"Models and preprocessors saved to {save_dir}/")
    
    def load_models(self, save_dir: str = 'saved_models'):
        """Load trained models and preprocessors."""
        # Load preprocessors
        with open(os.path.join(save_dir, 'tfidf_vectorizer.pkl'), 'rb') as f:
            self.tfidf_vectorizer = pickle.load(f)
        
        with open(os.path.join(save_dir, 'metadata_scaler.pkl'), 'rb') as f:
            self.metadata_scaler = pickle.load(f)
        
        # Load models
        self.models = {}
        model_files = [f for f in os.listdir(save_dir) if f.endswith('_model.pkl')]
        for model_file in model_files:
            model_name = model_file.replace('_model.pkl', '')
            with open(os.path.join(save_dir, model_file), 'rb') as f:
                self.models[model_name] = pickle.load(f)
        
        # Load feature names and stats
        try:
            with open(os.path.join(save_dir, 'feature_names.pkl'), 'rb') as f:
                self.feature_names = pickle.load(f)
                
            with open(os.path.join(save_dir, 'training_stats.pkl'), 'rb') as f:
                self.training_stats = pickle.load(f)
        except FileNotFoundError:
            print("Warning: Feature names or training stats not found")
        
        print(f"Loaded models: {list(self.models.keys())}")


def load_block_data(file_path: str) -> pd.DataFrame:
    """
    Load and validate HDFS block data.
    
    Args:
        file_path: Path to the data file (CSV, pickle, etc.)
        
    Returns:
        Validated dataframe
    """
    # Determine file type and load accordingly
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.pkl') or file_path.endswith('.pickle'):
        df = pd.read_pickle(file_path)
    elif file_path.endswith('.parquet'):
        df = pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    # Validate required columns
    required_columns = ['block_id', 'sequence_length', 'template_sequence_str', 'time_span', 'anomaly_binary_val']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Basic data validation
    print(f"Loaded {len(df)} block sequences")
    print(f"Data shape: {df.shape}")
    print(f"Anomaly distribution:")
    print(df['anomaly_binary_val'].value_counts())
    print(f"\nSequence length stats:")
    print(df['sequence_length'].describe())
    print(f"\nTime span stats:")
    print(df['time_span'].describe())
    
    return df




if __name__ == "__main__":
    # Demo usage
    print("=== HDFS Block Anomaly Detection Demo ===")
    
  
    # Initialize detector
    print("\n2. Initializing detector...")
    detector = BlockAnomalyDetector(
        tfidf_max_features=500,
        tfidf_ngram_range=(1, 2),  # Use bigrams too
        random_state=42
    )
    
    # Train models
    print("\n3. Training models...")
    results = detector.train_models(df, test_size=0.2)
    
    # Save models
    print("\n4. Saving models...")
    detector.save_models('demo_models')
    
    print("\n=== Demo completed ===")