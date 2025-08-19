#!/usr/bin/env python3
"""
HDFS Log Anomaly Detection Pipeline
A comprehensive tool to parse HDFS logs using Drain3, create sequences, and prepare data for anomaly detection.
Supports both block-level supervised and window-level unsupervised anomaly detection.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import re
from collections import defaultdict, Counter
import json
import pickle
from typing import List, Dict, Any, Tuple, Optional
import argparse

# Import our modular components
from log_parser import LogParser
from block_model import BlockAnomalyDetector
from window_model import WindowAnomalyDetector
from sequence_creator import SequenceCreator
from dataset_builder import DatasetBuilder
from config_utils import load_config
from utils import load_anomaly_labels, ensure_output_directory, save_dataframe_to_file


def run_block_supervised_detection(config: Dict[str, Any], output_dir: str, parsed_logs: List[Dict[str, Any]]):
    """
    Run block-level supervised anomaly detection.
    
    Args:
        config: Configuration dictionary
        output_dir: Output directory path
        parsed_logs: List of parsed log dictionaries
    """
    print("\n" + "="*60)
    print("BLOCK-LEVEL SUPERVISED ANOMALY DETECTION")
    print("="*60)
    
    # Step 1: Create block sequences
    print("\nStep 1: Creating block sequences...")
    sequence_creator = SequenceCreator(parsed_logs, config)
    block_sequences = sequence_creator.group_by_block()
    
    if not block_sequences:
        print("No block sequences created. Exiting block detection.")
        return
    
    # Step 2: Build block training dataset
    print("\nStep 2: Building block training dataset...")
    anomaly_labels = load_anomaly_labels(config['input']['labels_file'])
    dataset_builder = DatasetBuilder(block_sequences, anomaly_labels=anomaly_labels, config=config)
    block_dataset_df = dataset_builder.build_block_training_dataset()
    
    # Save block dataset
    block_output_file = os.path.join(output_dir, f"{config['output']['block_prefix']}.{config['output']['format']}")
    dataset_builder.save_sequences(block_dataset_df, block_output_file, format=config['output']['format'])
    
    # Step 3: Train block anomaly detection models
    print("\nStep 3: Training block anomaly detection models...")
    block_config = config.get('block_detection', {})
    detector = BlockAnomalyDetector(
        tfidf_max_features=block_config.get('tfidf_max_features', 1000),
        tfidf_ngram_range=tuple(block_config.get('tfidf_ngram_range', [1, 2])),
        random_state=42
    )
    
    results = detector.train_models(
        block_dataset_df, 
        test_size=block_config.get('test_size', 0.2),
        validation_size=block_config.get('validation_size', 0.1)
    )
    
    # Step 4: Save block models
    print("\nStep 4: Saving block models...")
    block_models_dir = os.path.join(output_dir, 'block_models')
    detector.save_models(block_models_dir)
    
    print("\nBlock-level supervised detection completed successfully!")
    return detector, results


def run_window_unsupervised_detection(config: Dict[str, Any], output_dir: str, parsed_logs: List[Dict[str, Any]]):
    """
    Run window-level unsupervised anomaly detection.
    
    Args:
        config: Configuration dictionary
        output_dir: Output directory path
        parsed_logs: List of parsed log dictionaries
    """
    print("\n" + "="*60)
    print("WINDOW-LEVEL UNSUPERVISED ANOMALY DETECTION")
    print("="*60)
    
    # Step 1: Create window dataset
    print("\nStep 1: Creating window dataset...")
    window_config = config.get('window_detection', {})
    
    # Initialize DatasetBuilder with None for sequences (window detection doesn't need pre-existing sequences)
    dataset_builder = DatasetBuilder(block_sequences=None, window_sequences=None)
    
    window_dataset_df = dataset_builder.build_window_dataset(
        logs=parsed_logs,
        window_size=window_config.get('window_size', 100),
        overlap=window_config.get('overlap', 0)
    )
    
    if window_dataset_df.empty:
        print("No window dataset created. Exiting window detection.")
        return None, None
    
    # Save window dataset
    window_output_file = os.path.join(output_dir, f"{config['output']['window_prefix']}.{config['output']['format']}")
    dataset_builder.save_sequences(window_dataset_df, window_output_file, format=config['output']['format'])
    
    # Step 2: Train window anomaly detection models
    print("\nStep 2: Training window anomaly detection models...")
    detector = WindowAnomalyDetector(
        tfidf_max_features=window_config.get('tfidf_max_features', 1000),
        tfidf_ngram_range=tuple(window_config.get('tfidf_ngram_range', [1, 2])),
        random_state=42
    )
    
    results = detector.train_models(window_dataset_df)
    
    # Step 3: Save window models
    print("\nStep 3: Saving window models...")
    window_models_dir = os.path.join(output_dir, 'window_models')
    detector.save_models(window_models_dir)
    
    # Step 4: Get top anomalous windows
    print("\nStep 4: Identifying top anomalous windows...")
    top_anomalous = detector.get_top_anomalous_windows(
        window_dataset_df, 
        model_name='isolation_forest', 
        top_n=10
    )
    
    print(f"\nTop 10 most anomalous windows:")
    for _, row in top_anomalous.iterrows():
        print(f"  {row['window_id']}: Score={row['anomaly_score']:.4f}, "
              f"Logs={row['log_count']}, Time={row['time_span']:.1f}s")
    
    # Step 5: Evaluate with labels if available (for v1 logs)
    if os.path.exists(config['input']['labels_file']):
        print("\nStep 5: Evaluating with anomaly labels...")
        anomaly_labels = load_anomaly_labels(config['input']['labels_file'])
        evaluation_results = detector.evaluate_with_labels(
            window_dataset_df, 
            anomaly_labels, 
            model_name='isolation_forest'
        )
    
    print("\nWindow-level unsupervised detection completed successfully!")
    return detector, results


def main():
    """Main pipeline execution with model selection."""
    parser = argparse.ArgumentParser(description='HDFS Log Anomaly Detection Pipeline')
    parser.add_argument('--model', choices=['block_supervised', 'window_unsupervised', 'both'], 
                       default='both', help='Which model(s) to run')
    parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    parser.add_argument('--max-lines', type=int, help='Maximum lines to parse (overrides config)')
    
    args = parser.parse_args()
    
    print("HDFS Log Anomaly Detection Pipeline")
    print("="*60)
    
    # Load configuration from YAML file
    print("Loading configuration from config.yaml...")
    config = load_config(args.config)
    
    # Override max_lines if provided as argument
    if args.max_lines:
        config['input']['max_lines'] = args.max_lines
    
    # Ensure output directory exists
    output_dir = config['output'].get('output_dir', 'output')
    output_dir = ensure_output_directory(output_dir)
    
    print(f"Configuration loaded successfully")
    print(f"Log file: {config['input']['log_file']}")
    print(f"Labels file: {config['input']['labels_file']}")
    print(f"Max lines: {config['input']['max_lines'] or 'All'}")
    print(f"Model(s) to run: {args.model}")
    print(f"Output directory: {output_dir}")
    print("="*60)
    
    # Check if log file exists
    if not os.path.exists(config['input']['log_file']):
        print(f"Error: Log file not found: {config['input']['log_file']}")
        sys.exit(1)
    
    try:
        # Step 1: Parse logs using Drain3 (common for both models)
        print("\nStep 1: Parsing logs with Drain3...")
        drain_parser = LogParser(config)
        parsed_logs = drain_parser.parse_logs(
            config['input']['log_file'], 
            max_lines=config['input']['max_lines']
        )
        if not parsed_logs:
            print("No logs were successfully parsed. Exiting.")
            sys.exit(1)
        
        print(f"Successfully parsed {len(parsed_logs)} log entries")
        
        # Step 2: Create sequences and save DataFrames (for analysis and debugging)
        print("\nStep 2: Creating sequences and saving DataFrames...")
        sequence_creator = SequenceCreator(parsed_logs, config)
        
        # Create sequences
        block_sequences = sequence_creator.group_by_block()
        time_sequences = sequence_creator.group_by_time_window()
        all_sequences = {**block_sequences, **time_sequences}
        
        # Print statistics
        sequence_creator.print_sequence_statistics()
        
        # Save logs DataFrame for analysis
        logs_df = sequence_creator.get_logs_dataframe()
        logs_output_file = os.path.join(output_dir, f"{config['output']['prefix']}_logs.{config['output']['format']}")
        save_dataframe_to_file(logs_df, logs_output_file, config['output']['format'])
        
        # Save individual sequence type DataFrames
        block_logs_df = sequence_creator.get_block_sequences_dataframe()
        time_logs_df = sequence_creator.get_time_sequences_dataframe()
        
        if not block_logs_df.empty:
            block_logs_file = os.path.join(output_dir, f"{config['output']['block_prefix']}_logs.{config['output']['format']}")
            save_dataframe_to_file(block_logs_df, block_logs_file, config['output']['format'])
        
        if not time_logs_df.empty:
            time_logs_file = os.path.join(output_dir, f"{config['output']['window_prefix']}_logs.{config['output']['format']}")
            save_dataframe_to_file(time_logs_df, time_logs_file, config['output']['format'])
        
        print("Sequence DataFrames saved for analysis and debugging")
        
        # Step 3: Run selected model(s) using the created sequences
        if args.model in ['block_supervised', 'both']:
            block_detector, block_results = run_block_supervised_detection(config, output_dir, parsed_logs)
        
        if args.model in ['window_unsupervised', 'both']:
            window_detector, window_results = run_window_unsupervised_detection(config, output_dir, parsed_logs)
            if window_detector is None:
                print("Warning: Window detection failed to create models")
        
        print(f"\nPipeline completed successfully!")
        print(f"All outputs saved to: {output_dir}")
        
        # Summary of what was created
        print(f"\n" + "="*60)
        print("PIPELINE SUMMARY")
        print("="*60)
        print(f"✓ Logs parsed: {len(parsed_logs):,}")
        print(f"✓ Block sequences: {len(block_sequences):,}")
        print(f"✓ Time-window sequences: {len(time_sequences):,}")
        print(f"✓ Sequence DataFrames saved for analysis")
        
        if args.model in ['block_supervised', 'both'] and 'block_detector' in locals() and block_detector:
            print(f"✓ Block models trained and saved")
        if args.model in ['window_unsupervised', 'both'] and 'window_detector' in locals() and window_detector:
            print(f"✓ Window models trained and saved")
        
        print("="*60)
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
