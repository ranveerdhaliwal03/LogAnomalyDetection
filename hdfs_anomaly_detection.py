#!/usr/bin/env python3
"""
HDFS Log Anomaly Detection Pipeline
A comprehensive tool to parse HDFS logs using Drain3, create sequences, and prepare data for anomaly detection.
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

# Import our modular components
from log_parser import LogParser
from sequence_creator import SequenceCreator
from dataset_builder import DatasetBuilder
from config_utils import load_config
from utils import load_anomaly_labels, ensure_output_directory, save_dataframe_to_file


def main():
    """Main pipeline execution."""
    print("HDFS Log Anomaly Detection Pipeline")
    print("="*60)
    
    # Load configuration from YAML file
    print("Loading configuration from config.yaml...")
    config = load_config("config.yaml")
    
    # Ensure output directory exists
    output_dir = config['output'].get('output_dir', 'output')
    output_dir = ensure_output_directory(output_dir)
    
    print(f"Configuration loaded successfully")
    print(f"Log file: {config['input']['log_file']}")
    print(f"Labels file: {config['input']['labels_file']}")
    print(f"Max lines: {config['input']['max_lines'] or 'All'}")
    print(f"Time window: {config['sequences']['time_window']['window_size_seconds']}s")
    print(f"Overlap: {config['sequences']['time_window']['overlap_seconds']}s")
    print(f"Output format: {config['output']['format']}")
    print(f"Output directory: {output_dir}")
    print("="*60)
    
    # Check if log file exists
    if not os.path.exists(config['input']['log_file']):
        print(f"Error: Log file not found: {config['input']['log_file']}")
        sys.exit(1)
    
    try:
        # Step 1: Parse logs using Drain3
        print("\nStep 1: Parsing logs with Drain3...")
        drain_parser = LogParser(config)
        parsed_logs = drain_parser.parse_logs(
            config['input']['log_file'], 
            max_lines=config['input']['max_lines']
        )
        if not parsed_logs:
            print("No logs were successfully parsed. Exiting.")
            sys.exit(1)
     
        # Step 2: Create sequences
        print("\nStep 2: Creating sequences...")
        sequence_creator = SequenceCreator(parsed_logs, config)
        
        # Create sequences
        block_sequences = sequence_creator.group_by_block()
        time_sequences = sequence_creator.group_by_time_window()
        all_sequences = {**block_sequences, **time_sequences}
        
        
        # Step 3: Convert sequences back to DataFrame format
        print("\nStep 3: Converting sequences to DataFrame format...")
        sequence_creator.print_sequence_statistics()
        logs_df = sequence_creator.get_logs_dataframe()
    
        logs_output_file = os.path.join(output_dir, f"{config['output']['prefix']}_logs.{config['output']['format']}")
        save_dataframe_to_file(logs_df, logs_output_file, config['output']['format'])
        
        # Also save individual sequence type DataFrames to output directory
        block_logs_df = sequence_creator.get_block_sequences_dataframe()
        time_logs_df = sequence_creator.get_time_sequences_dataframe()


        # Step 4: Load anomaly labels
        print("\nStep 4: Loading anomaly labels...")
        anomaly_labels = load_anomaly_labels(config['input']['labels_file'])
      
        # Cal some metrics to save m
        dataset_builder = DatasetBuilder(block_sequences, anomaly_labels=anomaly_labels, config=config)

        #block sequence 
        block_output_file = os.path.join(output_dir, f"{config['output']['block_prefix']}.{config['output']['format']}")
        block_dataset = dataset_builder.build_block_training_dataset()
       
 
        dataset_builder.save_sequences(block_dataset, block_output_file, format=config['output']['format'])
       
        block_logs_df.to_csv('test.csv', index=False) 

        
        
        # if not block_logs_df.empty:
        #     block_output_file = os.path.join(output_dir, f"{config['output']['prefix']}_block_logs.{config['output']['format']}")
        #     save_dataframe_to_file(block_logs_df, block_output_file, config['output']['format'])
        
        # if not time_logs_df.empty:
        #     time_output_file = os.path.join(output_dir, f"{config['output']['prefix']}_time_logs.{config['output']['format']}")
        #     save_dataframe_to_file(time_logs_df, time_output_file, config['output']['format'])
        
        
        
        # Step 5: Build and save dataset to output directory
        # print("\nStep 5: Building and saving dataset...")
        # dataset_builder = DatasetBuilder(all_sequences, anomaly_labels_df, config)
        
        # # Save sequences to output directory
        # output_file = os.path.join(output_dir, f"{config['output']['prefix']}.{config['output']['format']}")
        # dataset_builder.save_sequences(output_file, format=config['output']['format'])
        
        # # Show summary
        # print("\n" + "="*60)
        # print("PIPELINE SUMMARY")
        # print("="*60)
        # print(f"Total logs parsed: {len(parsed_logs):,}")
        # print(f"Unique templates found: {drain_parser.template_count}")
        # print(f"Block sequences: {len(block_sequences):,}")
        # print(f"Time-window sequences: {len(time_sequences):,}")
        # print(f"Total sequences: {len(all_sequences):,}")
        # print(f"Logs DataFrame created: {len(logs_df):,} entries")
        # print(f"Logs DataFrame columns: {list(logs_df.columns)}")
        # if not block_logs_df.empty:
        #     print(f"Block logs DataFrame: {len(block_logs_df):,} entries")
        # if not time_logs_df.empty:
        #     print(f"Time logs DataFrame: {len(time_logs_df):,} entries")
        # print(f"Anomaly labels loaded: {len(anomaly_labels_df):,}")
        # print(f"Sequences output saved to: {output_file}")
        # print(f"Logs DataFrame saved to: {logs_output_file}")
        # if not block_logs_df.empty:
        #     print(f"Block logs DataFrame saved to: {block_output_file}")
        # if not time_logs_df.empty:
        #     print(f"Time logs DataFrame saved to: {time_output_file}")
        # print(f"All outputs saved to directory: {output_dir}")
        # print("\nPipeline complete! Ready for anomaly detection model training.")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
