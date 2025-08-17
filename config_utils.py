#!/usr/bin/env python3
"""
Configuration Utilities Module
Handles loading and managing configuration from YAML files.
"""

import os
import yaml
from typing import Dict, Any


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    if not os.path.exists(config_path):
        print(f"Warning: Configuration file not found: {config_path}")
        print("Using default configuration...")
        return get_default_config()
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Configuration loaded from: {config_path}")
        return config
    except Exception as e:
        print(f"Error loading configuration: {e}")
        print("Using default configuration...")
        return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """Get default configuration if YAML file is not available."""
    return {
        'input': {
            'log_file': 'Data/hdfs_v1/HDFS.log',
            'labels_file': 'Data/hdfs_v1/anomaly_label.csv',
            'max_lines': None
        },
        'output': {
            'format': 'csv',
            'prefix': 'hdfs_sequences',
            'output_dir': 'output'  # New output directory setting
        },
        'drain3': {
            'persistence_file': 'drain3_state.json',
            'similarity_threshold': 0.4,
            'tree_depth': 4,
            'max_children': 100,
            'max_clusters': 1000
        },
        'sequences': {
            'time_window': {
                'window_size_seconds': 60,
                'overlap_seconds': 0
            }
        }
    }
