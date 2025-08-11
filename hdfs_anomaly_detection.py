#!/usr/bin/env python3
"""
HDFS Log Anomaly Detection MVP
A simple CLI tool to parse HDFS logs, create sequences, and detect anomalies.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
import re
from collections import defaultdict, Counter
import json

class HDFSLogParser:
    """Parse HDFS logs and extract structured information."""
    
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        self.logs = []
        self.parsed_logs = []
        
    def parse_log_line(self, line):
        """Parse a single log line and extract key information."""
        # HDFS log format: MMDDYY HHMMSS milliseconds INFO component: message
        # Example: 081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687919862906 src: /10.250.19.102:54106 dest: /10.250.19.102:50010
        
        # Basic regex pattern for actual HDFS logs
        pattern = r'(\d{6}) (\d{6}) (\d+) (\w+) ([^:]+): (.+)'
        match = re.match(pattern, line.strip())
        
        if match:
            date_str, time_str, milliseconds, level, component, message = match.groups()
            
            try:
                # Parse date: MMDDYY format
                month = int(date_str[:2])
                day = int(date_str[2:4])
                year = 2000 + int(date_str[4:6])  # Assume 20xx
                
                # Parse time: HHMMSS format
                hour = int(time_str[:2])
                minute = int(time_str[2:4])
                second = int(time_str[4:6])
                
                # Create timestamp (note: using 2008 as base year since logs are from 2008)
                timestamp = datetime(2008, month, day, hour, minute, second, int(milliseconds) * 1000)
            except:
                timestamp = None
                
            # Extract block ID if present
            block_id = None
            block_match = re.search(r'blk_(-?\d+)', message)
            if block_match:
                block_id = block_match.group(1)
                
            # Extract session/request ID if present (using block ID as session for now)
            session_id = block_id
                
            return {
                'timestamp': timestamp,
                'level': level,
                'component': component,
                'message': message,
                'block_id': block_id,
                'session_id': session_id,
                'raw_line': line.strip()
            }
        return None
    
    def parse_logs(self, max_lines=None):
        """Parse the log file and extract structured information."""
        print(f"Parsing logs from: {self.log_file_path}")
        
        parsed_count = 0
        total_lines = 0
        
        with open(self.log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                if max_lines and i >= max_lines:
                    break
                    
                total_lines += 1
                parsed_log = self.parse_log_line(line)
                if parsed_log:
                    self.parsed_logs.append(parsed_log)
                    parsed_count += 1
                    
                if total_lines % 10000 == 0:
                    print(f"Processed {total_lines:,} lines, parsed {parsed_count:,} successfully")
        
        print(f"\nParsing complete!")
        print(f"Total lines processed: {total_lines:,}")
        print(f"Successfully parsed: {parsed_count:,}")
     
        
        return self.parsed_logs

class SequenceCreator:
    """Create sequences from parsed logs by grouping them."""
    
    def __init__(self, parsed_logs):
        self.parsed_logs = parsed_logs
        self.sequences = {}
        
    def create_block_sequences(self):
        """Group logs by block ID to create sequences."""
        print("\nCreating block sequences")
        
        block_groups = defaultdict(list)
        
        for log in self.parsed_logs:
            if log['block_id']:
                block_groups[log['block_id']].append(log)
        
        # Sort each sequence by timestamp
        for block_id, logs in block_groups.items():
            # Filter out logs without valid timestamps
            valid_logs = [log for log in logs if log['timestamp']]
            if valid_logs:
                valid_logs.sort(key=lambda x: x['timestamp'])
                self.sequences[f"block_{block_id}"] = {
                    'type': 'block',
                    'id': block_id,
                    'logs': valid_logs,
                    'length': len(valid_logs),
                    'time_span': (valid_logs[-1]['timestamp'] - valid_logs[0]['timestamp']).total_seconds() if len(valid_logs) > 1 else 0
                }
        
        print(f"Created {len(self.sequences)} block sequences")
        return self.sequences
    
    def create_time_based_sequences(self, time_window_seconds=300):
        """Create sequences based on time windows."""
        print(f"\nCreating time-based sequences (length={time_window_seconds} seconds)")
        
        # Sort all logs by timestamp
        valid_logs = [log for log in self.parsed_logs if log['timestamp']]
        valid_logs.sort(key=lambda x: x['timestamp'])
        
        if not valid_logs:
            print("No valid timestamps found for time-based sequences")
            return {}
        
        time_sequences = {}
        current_sequence = []
        sequence_start = valid_logs[0]['timestamp']
        sequence_id = 0
        
        for log in valid_logs:
            time_diff = (log['timestamp'] - sequence_start).total_seconds()
            
            if time_diff > time_window_seconds and current_sequence:
                # Save current sequence
                time_sequences[f"time_{sequence_id}"] = {
                    'type': 'time',
                    'id': sequence_id,
                    'logs': current_sequence,
                    'length': len(current_sequence),
                    'start_time': sequence_start,
                    'end_time': current_sequence[-1]['timestamp'],
                    'time_span': time_diff
                }
                
                # Start new sequence
                current_sequence = [log]
                sequence_start = log['timestamp']
                sequence_id += 1
            else:
                current_sequence.append(log)
        
        # Don't forget the last sequence
        if current_sequence:
            time_sequences[f"time_{sequence_id}"] = {
                'type': 'time',
                'id': sequence_id,
                'logs': current_sequence,
                'length': len(current_sequence),
                'start_time': sequence_start,
                'end_time': current_sequence[-1]['timestamp'],
                'time_span': (current_sequence[-1]['timestamp'] - sequence_start).total_seconds()
            }
        
        print(f"Created {len(time_sequences)} time-based sequences")
        return time_sequences

class LogAnalyzer:
    """Analyze logs and sequences for insights."""
    
    def __init__(self, parsed_logs, sequences):
        self.parsed_logs = parsed_logs
        self.sequences = sequences
        
    def basic_statistics(self):
        """Generate basic statistics about the logs."""
        print("\n" + "="*50)
        print("BASIC LOG STATISTICS")
        print("="*50)
        
        # Log levels
        levels = Counter(log['level'] for log in self.parsed_logs if log['level'])
        print(f"\nLog Levels [INFO/ERROR]:")
        for level, count in levels.most_common():
            print(f"  {level}: {count:,}")
        
        # Components
        # components = Counter(log['component'] for log in self.parsed_logs if log['component'])
        # print(f"\nTop Components:")
        # for component, count in components.most_common(10):
        #     print(f"  {component}: {count:,}")
        
        # Block IDs
        block_ids = [log['block_id'] for log in self.parsed_logs if log['block_id']]
        unique_blocks = len(set(block_ids))
        print(f"\nBlock Information:")
        print(f"  Total block references: {len(block_ids):,}")
        print(f"  Unique blocks: {unique_blocks:,}")
        
        # # Time range
        # timestamps = [log['timestamp'] for log in self.parsed_logs if log['timestamp']]
        # if timestamps:
        #     print(f"\nTime Range:")
        #     print(f"  Start: {min(timestamps)}")
        #     print(f"  End: {max(timestamps)}")
        #     print(f"  Duration: {max(timestamps) - min(timestamps)}")
    
    def sequence_statistics(self):
        """Generate statistics about the sequences."""
        print("\n" + "="*50)
        print("SEQUENCE STATISTICS")
        print("="*50)
        
        if not self.sequences:
            print("No sequences available for analysis")
            return
        
        # Sequence lengths
        lengths = [seq['length'] for seq in self.sequences.values()]
        print(f"\nSequence Lengths:")
        print(f"  Total sequences: {len(self.sequences):,}")
        print(f"  Average length: {np.mean(lengths):.2f}")
        print(f"  Median length: {np.median(lengths):.2f}")
        print(f"  Min length: {min(lengths)}")
        print(f"  Max length: {max(lengths)}")
        
        # Sequence types
        types = Counter(seq['type'] for seq in self.sequences.values())
        print(f"\nSequence Types:")
        for seq_type, count in types.items():
            print(f"  {seq_type}: {count:,}")
        
        # Time spans for block sequences
        block_sequences = [seq for seq in self.sequences.values() if seq['type'] == 'block']
        if block_sequences:
            time_spans = [seq['time_span'] for seq in block_sequences if seq['time_span'] > 0]
            if time_spans:
                print(f"\nBlock Sequence Time Spans:")
                print(f"  Average: {np.mean(time_spans):.2f}s")
                print(f"  Median: {np.median(time_spans):.2f}s")
                print(f"  Min: {min(time_spans):.2f}s")
                print(f"  Max: {max(time_spans):.2f}s")

def main():
    parser = argparse.ArgumentParser(description='HDFS Log Anomaly Detection MVP')
    parser.add_argument('--log-file', default='Data/hdfs_v1/HDFS.log', 
                       help='Path to HDFS log file')
    parser.add_argument('--max-lines', type=int, default=None,
                       help='Maximum number of lines to parse (for testing)')
    parser.add_argument('--time-window', type=int, default=300,
                       help='Time window in seconds for time-based sequences')
    
    args = parser.parse_args()
    
    print("HDFS Log Anomaly Detection MVP")
    print("="*50)
    print(f"Log file: {args.log_file}")
    print(f"Max lines: {args.max_lines or 'All'}")
    print(f"Time window: {args.time_window}s")
    print("="*50)
    
    # Check if log file exists
    if not os.path.exists(args.log_file):
        print(f"Error: Log file not found: {args.log_file}")
        sys.exit(1)
    
    try:
        # Step 1: Parse logs
        print("\nStep 1: Parsing logs...")
        parser = HDFSLogParser(args.log_file)
        parsed_logs = parser.parse_logs(max_lines=args.max_lines)
        
        if not parsed_logs:
            print("No logs were successfully parsed. Exiting.")
            sys.exit(1)
        
        # Step 2: Create sequences
        print("\nStep 2: Creating sequences...")
        sequence_creator = SequenceCreator(parsed_logs)
        
        # Create block-based sequences
        block_sequences = sequence_creator.create_block_sequences()
        
        # Create time-based sequences
        time_sequences = sequence_creator.create_time_based_sequences(args.time_window)
        
        # Combine all sequences
        all_sequences = {**block_sequences, **time_sequences}
        
        # Step 3: Analyze logs and sequences
        print("\nStep 3: Analyzing logs and sequences...")
        analyzer = LogAnalyzer(parsed_logs, all_sequences)
        analyzer.basic_statistics()
        analyzer.sequence_statistics()
        
        # Step 4: Save results
        print("\nStep 4: Saving results...")
        
        # Save parsed logs summary
        summary = {
            'total_logs_parsed': len(parsed_logs),
            'total_sequences': len(all_sequences),
            'block_sequences': len(block_sequences),
            'time_sequences': len(time_sequences),
            'parse_timestamp': datetime.now().isoformat()
        }
        
        with open('parsing_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"Summary saved to: parsing_summary.json")
        
        # Show sample sequences
        print("\n" + "="*50)
        print("SAMPLE SEQUENCES")
        print("="*50)
        
        # Show a few block sequences
        block_samples = list(block_sequences.items())[:3]
        for seq_id, seq_data in block_samples:
            print(f"\n{seq_id}:")
            print(f"  Length: {seq_data['length']} logs")
            print(f"  Time span: {seq_data['time_span']:.2f}s")
            print(f"  First log: {seq_data['logs'][0]['message'][:80]}...")
            print(f"  Last log: {seq_data['logs'][-1]['message'][:80]}...")
        
        print(f"\nMVP Pipeline Complete!")
        print(f"Ready for next steps: embedding and anomaly detection")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
