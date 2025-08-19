#!/usr/bin/env python3
"""
HDFS Window Anomaly Detection Pipeline

This module provides functionality for detecting anomalous HDFS log windows
using unsupervised learning methods (Isolation Forest, One-Class SVM) on
TF-IDF vectorized template sequences.
"""

import pandas as pd
import numpy as np
import pickle
import os
from typing import Tuple, Dict, Any, Optional, List
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Sklearn imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Optional: Additional unsupervised models
try:
    from sklearn.neighbors import LocalOutlierFactor
    LOF_AVAILABLE = True
except ImportError:
    LOF_AVAILABLE = False

try:
    from sklearn.covariance import EllipticEnvelope
    ELLIPTIC_AVAILABLE = True
except ImportError:
    ELLIPTIC_AVAILABLE = False


class WindowAnomalyDetector:
    """
    Main class for HDFS window anomaly detection using unsupervised learning.
    
    Converts template sequences to TF-IDF features and trains unsupervised
    anomaly detection models to identify anomalous log windows.
    """
    
    def __init__(self, 
                 tfidf_max_features: int = 1000,
                 tfidf_ngram_range: Tuple[int, int] = (1, 2),
                 random_state: int = 42):
        """
        Initialize the window anomaly detector.
        
        Args:
            tfidf_max_features: Maximum number of TF-IDF features
            tfidf_ngram_range: N-gram range for TF-IDF (default bigrams)
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
        required_cols = ['template_sequence_str', 'log_count', 'time_span']
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
        metadata_features = df[['log_count', 'time_span', 'unique_templates']].values
        
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
            metadata_names = ['log_count_scaled', 'time_span_scaled', 'unique_templates_scaled']
            self.feature_names = tfidf_names + metadata_names
        
        return combined_features
    
    def train_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train multiple unsupervised anomaly detection models.
        
        Args:
            df: Training dataframe with window data
            
        Returns:
            Dictionary containing training results and model info
        """
        print(f"Training unsupervised models on {len(df)} windows...")
        
        # Preprocess features
        X = self.preprocess_features(df, fit=True)
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"TF-IDF features: {len(self.tfidf_vectorizer.get_feature_names_out())}")
        
        results = {}
        
        # 1. Isolation Forest
        print("\n=== Training Isolation Forest ===")
        iso_forest = IsolationForest(
            contamination=0.1,  # Expected proportion of anomalies
            random_state=self.random_state,
            n_estimators=100,
            max_samples='auto'
        )
        iso_forest.fit(X)
        self.models['isolation_forest'] = iso_forest
        
        # Get anomaly scores (lower = more anomalous)
        iso_scores = iso_forest.score_samples(X)
        iso_predictions = iso_forest.predict(X)
        
        # Convert to binary (1 = normal, -1 = anomaly)
        iso_binary = (iso_predictions == 1).astype(int)
        
        results['isolation_forest'] = {
            'model_name': 'Isolation Forest',
            'anomaly_scores': iso_scores,
            'predictions': iso_predictions,
            'binary_predictions': iso_binary,
            'contamination': iso_forest.contamination,
            'n_estimators': iso_forest.n_estimators
        }
        
        print(f"Isolation Forest trained with contamination={iso_forest.contamination}")
        print(f"Anomaly score range: {iso_scores.min():.4f} to {iso_scores.max():.4f}")
        
        # 2. Local Outlier Factor (if available)
        if LOF_AVAILABLE:
            print("\n=== Training Local Outlier Factor ===")
            lof = LocalOutlierFactor(
                contamination=0.1,
                n_neighbors=20,
                metric='euclidean'
            )
            
            # LOF doesn't have a fit method, we get scores directly
            lof_scores = lof.fit_predict(X)
            lof_predictions = lof.fit_predict(X)
            
            # Convert to binary (1 = normal, -1 = anomaly)
            lof_binary = (lof_predictions == 1).astype(int)
            
            results['local_outlier_factor'] = {
                'model_name': 'Local Outlier Factor',
                'anomaly_scores': lof_scores,
                'predictions': lof_predictions,
                'binary_predictions': lof_binary,
                'contamination': lof.contamination,
                'n_neighbors': lof.n_neighbors
            }
            
            print(f"Local Outlier Factor trained with contamination={lof.contamination}")
        
        # 3. Elliptic Envelope (if available)
        if ELLIPTIC_AVAILABLE:
            print("\n=== Training Elliptic Envelope ===")
            elliptic = EllipticEnvelope(
                contamination=0.1,
                random_state=self.random_state
            )
            elliptic.fit(X)
            self.models['elliptic_envelope'] = elliptic
            
            # Get anomaly scores
            elliptic_scores = elliptic.score_samples(X)
            elliptic_predictions = elliptic.predict(X)
            
            # Convert to binary (1 = normal, -1 = anomaly)
            elliptic_binary = (elliptic_predictions == 1).astype(int)
            
            results['elliptic_envelope'] = {
                'model_name': 'Elliptic Envelope',
                'anomaly_scores': elliptic_scores,
                'predictions': elliptic_predictions,
                'binary_predictions': elliptic_binary,
                'contamination': elliptic.contamination
            }
            
            print(f"Elliptic Envelope trained with contamination={elliptic.contamination}")
        
        # Store training statistics
        self.training_stats = {
            'total_windows': len(df),
            'feature_count': X.shape[1],
            'tfidf_features': len(self.tfidf_vectorizer.get_feature_names_out()),
            'training_time': datetime.now().isoformat(),
            'models_trained': list(self.models.keys())
        }
        
        print(f"\nTraining complete! Trained {len(self.models)} models")
        return results
    
    def compute_anomaly_scores(self, df: pd.DataFrame, model_name: str = 'isolation_forest') -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute anomaly scores for windows using a trained model.
        
        Args:
            df: Input dataframe with window data
            model_name: Name of model to use for scoring
            
        Returns:
            Tuple of (anomaly_scores, binary_predictions)
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not trained. Available models: {list(self.models.keys())}")
        
        X = self.preprocess_features(df, fit=False)
        model = self.models[model_name]
        
        # Get anomaly scores (lower = more anomalous for most models)
        if hasattr(model, 'score_samples'):
            anomaly_scores = model.score_samples(X)
        else:
            # For models without score_samples, use decision_function
            anomaly_scores = model.decision_function(X)
        
        # Get predictions
        predictions = model.predict(X)
        
        # Convert to binary (1 = normal, -1 = anomaly)
        binary_predictions = (predictions == 1).astype(int)
        
        return anomaly_scores, binary_predictions
    
    def get_top_anomalous_windows(self, df: pd.DataFrame, model_name: str = 'isolation_forest', 
                                 top_n: int = 10) -> pd.DataFrame:
        """
        Get the top N most anomalous windows.
        
        Args:
            df: Input dataframe with window data
            model_name: Name of model to use
            top_n: Number of top anomalous windows to return
            
        Returns:
            DataFrame with top anomalous windows and their scores
        """
        anomaly_scores, _ = self.compute_anomaly_scores(df, model_name)
        
        # Create a copy of the dataframe with anomaly scores
        result_df = df.copy()
        result_df['anomaly_score'] = anomaly_scores
        
        # Sort by anomaly score (ascending for most models since lower = more anomalous)
        result_df = result_df.sort_values('anomaly_score')
        
        # Return top N most anomalous
        return result_df.head(top_n)
    
    def evaluate_with_labels(self, df: pd.DataFrame, anomaly_labels: Dict[str, str], 
                           model_name: str = 'isolation_forest') -> Dict[str, Any]:
        """
        Evaluate the model using anomaly labels (for v1 logs).
        
        Args:
            df: Input dataframe with window data
            anomaly_labels: Dictionary mapping block IDs to anomaly labels
            model_name: Name of model to use for evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\nEvaluating {model_name} with anomaly labels...")
        
        # Get predictions
        anomaly_scores, binary_predictions = self.compute_anomaly_scores(df, model_name)
        
        # Create ground truth labels
        y_true = []
        for _, row in df.iterrows():
            # Check if any logs in this window belong to anomalous blocks
            window_anomalous = False
            for log in row.get('logs', []):
                if log.get('block_id') in anomaly_labels:
                    if anomaly_labels[log['block_id']] == 'Anomaly':
                        window_anomalous = True
                        break
            
            y_true.append(1 if window_anomalous else 0)
        
        y_true = np.array(y_true)
        y_pred = binary_predictions
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        results = {
            'model_name': model_name,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn,
            'total_windows': len(df),
            'anomalous_windows': y_true.sum(),
            'detected_anomalies': y_pred.sum()
        }
        
        print(f"Evaluation Results for {model_name}:")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Anomalous windows: {y_true.sum()}/{len(df)}")
        print(f"Detected anomalies: {y_pred.sum()}")
        
        return results
    
    def save_models(self, save_dir: str = 'saved_window_models'):
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
        
        print(f"Window models and preprocessors saved to {save_dir}/")
    
    def load_models(self, save_dir: str = 'saved_window_models'):
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
        
        print(f"Loaded window models: {list(self.models.keys())}")


def load_window_data(file_path: str) -> pd.DataFrame:
    """
    Load and validate HDFS window data.
    
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
    required_columns = ['window_id', 'template_sequence_str', 'log_count', 'time_span']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Basic data validation
    print(f"Loaded {len(df)} window sequences")
    print(f"Data shape: {df.shape}")
    print(f"\nWindow size stats:")
    print(df['log_count'].describe())
    print(f"\nTime span stats:")
    print(df['time_span'].describe())
    
    return df


if __name__ == "__main__":
    # Demo usage
    print("=== HDFS Window Anomaly Detection Demo ===")
    
    # Load data (you would need to create this first)
    # df = load_window_data('window_sequences.csv')
    
    # Initialize detector
    print("\n1. Initializing window detector...")
    detector = WindowAnomalyDetector(
        tfidf_max_features=500,
        tfidf_ngram_range=(1, 2),  # Use bigrams too
        random_state=42
    )
    
    print("\n2. Training models...")
    # results = detector.train_models(df)
    
    # Save models
    print("\n3. Saving models...")
    # detector.save_models('demo_window_models')
    
    print("\n=== Demo completed ===")
