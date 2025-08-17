#!/usr/bin/env python3
"""
Utility Functions Module
Contains helper functions for the HDFS anomaly detection pipeline.
"""

import os
import pandas as pd
from typing import Dict, Any


def load_anomaly_labels(labels_file: str) -> Dict[str, int]:
    """
    Load anomaly labels from CSV file.
    
    Args:
        labels_file: Path to CSV file with anomaly labels
        
    Returns:
        Dictionary mapping sequence IDs to anomaly labels
    """
    if not os.path.exists(labels_file):
        print(f"Warning: Anomaly labels file not found: {labels_file}")
        return {}
    
    try:
        df = pd.read_csv(labels_file)
        # Assuming CSV has columns: sequence_id, anomaly_label
        labels = dict(zip(df['sequence_id'], df['anomaly_label']))
        print(f"Loaded {len(labels)} anomaly labels from {labels_file}")
        return labels
    except Exception as e:
        print(f"Error loading anomaly labels: {e}")
        return {}


def ensure_output_directory(output_dir: str) -> str:
    """
    Ensure the output directory exists, create if it doesn't.
    
    Args:
        output_dir: Path to output directory
        
    Returns:
        Absolute path to the output directory
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    return os.path.abspath(output_dir)


def save_dataframe_to_file(df: pd.DataFrame, output_path: str, format: str = 'csv') -> None:
    """
    Save a DataFrame to file in the specified format.
    
    Args:
        df: DataFrame to save
        output_path: Path to save the file
        format: Output format ('csv', 'pickle', or 'json')
    """
    print(f"Saving DataFrame to: {output_path}")
    
    if format.lower() == 'csv':
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} rows to CSV")
        
    elif format.lower() == 'pickle':
        df.to_pickle(output_path)
        print(f"Saved {len(df)} rows to pickle")
        
    elif format.lower() == 'json':
        df.to_json(output_path, orient='records', indent=2)
        print(f"Saved {len(df)} rows to JSON")
        
    else:
        raise ValueError(f"Unsupported format: {format}")
