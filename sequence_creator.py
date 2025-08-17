#!/usr/bin/env python3
"""
Sequence Creator Module
Handles creation of sequences from parsed logs by grouping them into different types.
"""

import pandas as pd
from collections import defaultdict
from datetime import timedelta
from typing import List, Dict, Any


class SequenceCreator:
    """Create sequences from parsed logs by grouping them into different types."""
    
    def __init__(self, parsed_logs: List[Dict[str, Any]], config: Dict[str, Any]):
        """
        Initialize sequence creator with parsed logs and configuration.
        
        Args:
            parsed_logs: List of parsed log dictionaries from Drain3LogParser
            config: Configuration dictionary
        """
        self.parsed_logs = parsed_logs
        self.config = config
        self.sequences = {}
        
    def group_by_block(self) -> Dict[str, Dict[str, Any]]:
        """
        Group logs by block ID to create block-level sequences.
        
        Returns:
            Dictionary of block sequences with template ID sequences
        """
        print("\nCreating block-level sequences...")
        
        block_groups = defaultdict(list)
        
        # Group logs by block ID
        for log in self.parsed_logs:
            if log['block_id']:
                block_groups[log['block_id']].append(log)
        
        # Create sequences for each block
        for block_id, logs in block_groups.items():
            # Filter out logs without valid timestamps and sort by time
            valid_logs = [log for log in logs if log['timestamp']]
            if valid_logs:
                valid_logs.sort(key=lambda x: x['timestamp'])
                
                # Extract template ID sequence
                template_sequence = [log['template_id'] for log in valid_logs]
                
                self.sequences[f"block_{block_id}"] = {
                    'type': 'block',
                    'id': block_id,
                    'logs': valid_logs,
                    'template_sequence': template_sequence,
                    'length': len(valid_logs),
                    'start_time': valid_logs[0]['timestamp'],
                    'end_time': valid_logs[-1]['timestamp'],
                    'time_span': (valid_logs[-1]['timestamp'] - valid_logs[0]['timestamp']).total_seconds() if len(valid_logs) > 1 else 0
                }
        
        print(f"Created {len(self.sequences)} block sequences")
        return self.sequences
    
    def group_by_time_window(self) -> Dict[str, Dict[str, Any]]:
        """
        Group parsed logs into sliding time windows to create time-based sequences.
        
        Returns:
            Dictionary of time-window sequences with template ID sequences
        """
        time_config = self.config['sequences']['time_window']
        window_size_seconds = time_config['window_size_seconds']
        overlap_seconds = time_config['overlap_seconds']
        
        print(f"\nCreating time-window sequences (window={window_size_seconds}s, overlap={overlap_seconds}s)...")
        
        # Sort all logs by timestamp
        valid_logs = [log for log in self.parsed_logs if log['timestamp']]
        valid_logs.sort(key=lambda x: x['timestamp'])
        
        if not valid_logs:
            print("No valid timestamps found for time-window sequences")
            return {}
        
        sequences = {}
        sequence_id = 0
        
        # Calculate step size for sliding windows
        step_size = window_size_seconds - overlap_seconds
        
        # Create sliding windows
        start_time = valid_logs[0]['timestamp']
        end_time = valid_logs[-1]['timestamp']
        
        current_start = start_time
        while current_start < end_time:
            current_end = current_start + timedelta(seconds=window_size_seconds)
            
            # Find logs within current window
            window_logs = [
                log for log in valid_logs 
                if current_start <= log['timestamp'] < current_end
            ]
            
            if window_logs:
                # Extract template ID sequence
                template_sequence = [log['template_id'] for log in window_logs]
                
                sequences[f"time_{sequence_id}"] = {
                    'type': 'time_window',
                    'id': sequence_id,
                    'logs': window_logs,
                    'template_sequence': template_sequence,
                    'length': len(window_logs),
                    'start_time': current_start,
                    'end_time': current_end,
                    'window_size': window_size_seconds
                }
                sequence_id += 1
            
            current_start += timedelta(seconds=step_size)
        
        print(f"Created {len(sequences)} time-window sequences")
        return sequences
    
    def get_all_sequences(self) -> Dict[str, Dict[str, Any]]:
        """Get all created sequences."""
        return self.sequences
    
    def get_logs_dataframe(self) -> pd.DataFrame:
        """
        Convert all sequences back to a DataFrame format with original log structure.
        
        Returns:
            DataFrame with columns: timestamp, level, component, message, block_id, sequence_id, raw_line, etc.
        """
        all_logs = []
        
        for seq_id, seq_data in self.sequences.items():
            for log in seq_data['logs']:
                # Add sequence information to each log
                log_copy = log.copy()
                log_copy['sequence_id'] = seq_id
                log_copy['sequence_type'] = seq_data['type']
                all_logs.append(log_copy)
        
        if not all_logs:
            return pd.DataFrame()
        
        # Convert to DataFrame
        logs_df = pd.DataFrame(all_logs)
        
        # Reorder columns to prioritize important ones
        column_order = ['timestamp', 'level', 'component', 'message', 'block_id', 'sequence_id', 'raw_line']
        existing_columns = [col for col in column_order if col in logs_df.columns]
        remaining_columns = [col for col in logs_df.columns if col not in existing_columns]
        final_columns = existing_columns + remaining_columns
        
        return logs_df[final_columns]
    
    def get_block_sequences_dataframe(self) -> pd.DataFrame:
        """
        Get block sequences as a DataFrame.
        
        Returns:
            DataFrame with block sequence information
        """
        block_logs = []
        
        for seq_id, seq_data in self.sequences.items():
            if seq_data['type'] == 'block':
                for log in seq_data['logs']:
                    log_copy = log.copy()
                    log_copy['sequence_id'] = seq_id
                    log_copy['sequence_type'] = 'block'
                    block_logs.append(log_copy)
        
        if not block_logs:
            return pd.DataFrame()
        
        logs_df = pd.DataFrame(block_logs)
        column_order = ['timestamp', 'level', 'component', 'message', 'block_id', 'sequence_id', 'raw_line']
        existing_columns = [col for col in column_order if col in logs_df.columns]
        remaining_columns = [col for col in logs_df.columns if col not in existing_columns]
        final_columns = existing_columns + remaining_columns
        
        return logs_df[final_columns]
    
    def get_time_sequences_dataframe(self) -> pd.DataFrame:
        """
        Get time-window sequences as a DataFrame.
        
        Returns:
            DataFrame with time-window sequence information
        """
        time_logs = []
        
        for seq_id, seq_data in self.sequences.items():
            if seq_data['type'] == 'time_window':
                for log in seq_data['logs']:
                    log_copy = log.copy()
                    log_copy['sequence_id'] = seq_id
                    log_copy['sequence_type'] = 'time_window'
                    time_logs.append(log_copy)
        
        if not time_logs:
            return pd.DataFrame()
        
        logs_df = pd.DataFrame(time_logs)
        column_order = ['timestamp', 'level', 'component', 'message', 'block_id', 'sequence_id', 'raw_line']
        existing_columns = [col for col in column_order if col in logs_df.columns]
        remaining_columns = [col for col in logs_df.columns if col not in existing_columns]
        final_columns = existing_columns + remaining_columns
        
        return logs_df[final_columns]
    
    def print_sequence_statistics(self) -> None:
        """
        Print statistics about the created sequences for debugging and analysis.
        """
        if not self.sequences:
            print("No sequences created yet.")
            return
        
        print("\n" + "="*50)
        print("SEQUENCE STATISTICS")
        print("="*50)
        
        # Count by type
        type_counts = {}
        total_logs = 0
        
        for seq_id, seq_data in self.sequences.items():
            seq_type = seq_data['type']
            type_counts[seq_type] = type_counts.get(seq_type, 0) + 1
            total_logs += len(seq_data['logs'])
        
        print(f"Total sequences: {len(self.sequences):,}")
        print(f"Total log entries: {total_logs:,}")
        
        for seq_type, count in type_counts.items():
            print(f"{seq_type.title()} sequences: {count:,}")
        
        # Show some examples
        print(f"\nExample sequence IDs:")
        for i, seq_id in enumerate(list(self.sequences.keys())[:5]):
            seq_data = self.sequences[seq_id]
            print(f"  {seq_id}: {seq_data['type']} - {len(seq_data['logs'])} logs")
        
        if len(self.sequences) > 5:
            print(f"  ... and {len(self.sequences) - 5} more sequences")
        
        print("="*50)
