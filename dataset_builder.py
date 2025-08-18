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
    def __init__(self, 
                 block_sequences: Optional[Dict[str, Dict[str, Any]]] = None,
                 window_sequences: Optional[Dict[str, Dict[str, Any]]] = None,
                 anomaly_labels: Optional[Dict[str, int]] = None,
                 config: Dict[str, Any] = None):
        
        if block_sequences is None and window_sequences is None:
            raise AssertionError("Need to have at least one Sequence")
        if block_sequences is not None:
            self.block_sequences = block_sequences
        if window_sequences is not None:
            self.window_sequences = window_sequences

        self.anomaly_labels = anomaly_labels or {}
        self.config = config or {}
        
    def build_block_training_dataset(self) -> pd.DataFrame: 
        if self.block_sequences is None:
            raise AssertionError("Block sequences not initialized")
         
        print("\nBuilding Block training dataset from sequences...")

        dataset_rows = []
        for seq_id, seq_data in self.block_sequences.items():

            # Basic sequence features
            features = {
                'block_id': seq_id,
                'sequence_length': seq_data['length'],
                'template_sequence': seq_data['template_sequence'],
                'template_sequence_str': ','.join(map(str, seq_data['template_sequence']))
            }

            if 'start_time' in seq_data and 'end_time' in seq_data:
                features['start_time'] = seq_data['start_time']
                features['end_time'] = seq_data['end_time']
                
            if 'time_span' in seq_data:
                features['time_span'] = seq_data['time_span']
            elif 'start_time' in seq_data and 'end_time' in seq_data:
                features['time_span'] = (seq_data['end_time'] - seq_data['start_time']).total_seconds()

             # Add anomaly label if available
 
            if seq_id in self.anomaly_labels:
                anomaly_binary_value = -1 
                if self.anomaly_labels[seq_id] == "Normal":
                    anomaly_binary_value = 0
                elif self.anomaly_labels[seq_id] == "Anomaly":
                     anomaly_binary_value = 1
                features['anomaly_binary_val'] = anomaly_binary_value
                features['anomaly_labels'] = self.anomaly_labels
            else:
                print("NOT FOUND")
                features['anomaly_label'] = -1  # Unknown label
            
            dataset_rows.append(features)

        dataset = pd.DataFrame(dataset_rows)
        print(f"Built dataset with {len(dataset)} sequences and {len(dataset.columns)} features")
        return dataset

    def save_sequences(self, dataset, output_path: str, format: str = 'csv') -> None:
        """
        Save sequences to file.
        
        Args:
            output_path: Path to save the sequences
            format: Output format ('csv', 'pickle', or 'json')
        """
        print(f"\nSaving sequences to: {output_path}")
        
        if format.lower() == 'csv':
            dataset.to_csv(output_path, index=False)
            print(f"Saved {len(dataset)} sequences to CSV")
