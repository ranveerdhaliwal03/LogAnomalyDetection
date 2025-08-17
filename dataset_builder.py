#!/usr/bin/env python3
"""
Dataset Builder Module
Handles building training datasets from sequences for anomaly detection models.
"""

import pandas as pd
import json
import pickle
from typing import Dict, Any, Optional


class DatasetBuilder:
    """Build training datasets from sequences for anomaly detection models."""
    
    def __init__(self, sequences: Dict[str, Dict[str, Any]], 
                 anomaly_labels: Optional[Dict[str, int]] = None,
                 config: Dict[str, Any] = None):
        """
        Initialize dataset builder.
        
        Args:
            sequences: Dictionary of sequences from SequenceCreator
            anomaly_labels: Optional dictionary mapping sequence IDs to anomaly labels (1=anomaly, 0=normal)
            config: Configuration dictionary
        """
        self.sequences = sequences
        self.anomaly_labels = anomaly_labels or {}
        self.config = config or {}
        
    def build_training_dataset(self) -> pd.DataFrame:
        """
        Build a training dataset from sequences.
        
        Returns:
            DataFrame with sequence features for training
        """
        print("\nBuilding training dataset from sequences...")
        
        dataset_rows = []
        
        for seq_id, seq_data in self.sequences.items():
            # Basic sequence features
            features = {
                'sequence_id': seq_id,
                'type': seq_data['type'],
                'length': seq_data['length'],
                'template_sequence': seq_data['template_sequence'],
                'template_sequence_str': ','.join(map(str, seq_data['template_sequence']))
            }
            
            # Add time-based features if available
            if 'start_time' in seq_data and 'end_time' in seq_data:
                features['start_time'] = seq_data['start_time']
                features['end_time'] = seq_data['end_time']
                
            if 'time_span' in seq_data:
                features['time_span'] = seq_data['time_span']
            elif 'start_time' in seq_data and 'end_time' in seq_data:
                features['time_span'] = (seq_data['end_time'] - seq_data['start_time']).total_seconds()
            
            # Add anomaly label if available
            if seq_id in self.anomaly_labels:
                features['anomaly_label'] = self.anomaly_labels[seq_id]
            else:
                features['anomaly_label'] = -1  # Unknown label
            
            dataset_rows.append(features)
        
        dataset = pd.DataFrame(dataset_rows)
        print(f"Built dataset with {len(dataset)} sequences and {len(dataset.columns)} features")
        
        return dataset
    
    def save_sequences(self, output_path: str, format: str = 'csv') -> None:
        """
        Save sequences to file.
        
        Args:
            output_path: Path to save the sequences
            format: Output format ('csv', 'pickle', or 'json')
        """
        print(f"\nSaving sequences to: {output_path}")
        
        if format.lower() == 'csv':
            dataset = self.build_training_dataset()
            dataset.to_csv(output_path, index=False)
            print(f"Saved {len(dataset)} sequences to CSV")
            
        elif format.lower() == 'pickle':
            with open(output_path, 'wb') as f:
                pickle.dump(self.sequences, f)
            print(f"Saved {len(self.sequences)} sequences to pickle")
            
        elif format.lower() == 'json':
            # Convert datetime objects to strings for JSON serialization
            json_sequences = {}
            for seq_id, seq_data in self.sequences.items():
                json_seq = seq_data.copy()
                if 'start_time' in json_seq:
                    json_seq['start_time'] = json_seq['start_time'].isoformat()
                if 'end_time' in json_seq:
                    json_seq['end_time'] = json_seq['end_time'].isoformat()
                json_sequences[seq_id] = json_seq
            
            with open(output_path, 'w') as f:
                json.dump(json_sequences, f, indent=2, default=str)
            print(f"Saved {len(self.sequences)} sequences to JSON")
            
        else:
            raise ValueError(f"Unsupported format: {format}")
