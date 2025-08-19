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
        
        # Allow initialization without sequences for window detection
        # Sequences can be added later via build_window_dataset or build_block_training_dataset
        self.block_sequences = block_sequences
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
                features['anomaly_labels'] = self.anomaly_labels[seq_id]
            else:
                print("NOT FOUND")
                features['anomaly_label'] = -1  # Unknown label
            
            dataset_rows.append(features)

        dataset = pd.DataFrame(dataset_rows)
        print(f"Built dataset with {len(dataset)} sequences and {len(dataset.columns)} features")
        return dataset

    def build_window_dataset(self, logs: list, window_size: int = 100, overlap: int = 0) -> pd.DataFrame:
        """
        Build a window-level dataset for unsupervised anomaly detection.
        
        Args:
            logs: List of parsed log dictionaries
            window_size: Number of log entries per window
            overlap: Number of overlapping entries between consecutive windows
            
        Returns:
            DataFrame with window_id and templates_seq columns
        """
        print(f"\nBuilding window-level dataset (window_size={window_size}, overlap={overlap})...")
        
        if not logs:
            print("No logs provided for window dataset creation")
            return pd.DataFrame()
        
        # Filter logs with valid timestamps and sort by time
        valid_logs = [log for log in logs if log['timestamp']]
        valid_logs.sort(key=lambda x: x['timestamp'])
        
        if not valid_logs:
            print("No valid timestamps found for window dataset creation")
            return pd.DataFrame()
        
        print(f"Creating windows from {len(valid_logs)} valid log entries...")
        
        window_rows = []
        window_id = 0
        step_size = window_size - overlap
        
        for i in range(0, len(valid_logs), step_size):
            end_idx = min(i + window_size, len(valid_logs))
            window_logs = valid_logs[i:end_idx]
            
            if len(window_logs) >= window_size // 2:  # Only create windows with at least half the target size
                # Extract template IDs for this window
                template_sequence = [log['template_id'] for log in window_logs]
                template_sequence_str = ','.join(map(str, template_sequence))
                
                # Calculate window metadata
                start_time = window_logs[0]['timestamp']
                end_time = window_logs[-1]['timestamp']
                time_span = (end_time - start_time).total_seconds()
                
                window_row = {
                    'window_id': f"window_{window_id}",
                    'start_idx': i,
                    'end_idx': end_idx,
                    'start_time': start_time,
                    'end_time': end_time,
                    'time_span': time_span,
                    'log_count': len(window_logs),
                    'template_sequence': template_sequence,
                    'template_sequence_str': template_sequence_str,
                    'unique_templates': len(set(template_sequence))
                }
                
                window_rows.append(window_row)
                window_id += 1
        
        dataset = pd.DataFrame(window_rows)
        print(f"Built window dataset with {len(dataset)} windows")
        print(f"Average window size: {dataset['log_count'].mean():.1f} logs")
        print(f"Average time span: {dataset['time_span'].mean():.1f} seconds")
        
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
        elif format.lower() == 'pickle':
            dataset.to_pickle(output_path)
            print(f"Saved {len(dataset)} sequences to pickle")
        elif format.lower() == 'json':
            dataset.to_json(output_path, orient='records', date_format='iso')
            print(f"Saved {len(dataset)} sequences to JSON")
        else:
            raise ValueError(f"Unsupported format: {format}")
