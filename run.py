#!/usr/bin/env python3
"""
Demo script for HDFS Log Anomaly Detection Pipeline
Perfect for demonstrating to the Ford team!
"""

import time
import json
from datetime import datetime
from hdfs_anomaly_detection import HDFSLogParser, SequenceCreator, LogAnalyzer

def pipeline():
    print("HDFS Log Anomaly Detection Pipeline")
    print("="*50)
    
    # Configuration
    log_file = 'Data/hdfs_v1/HDFS.log'
    sample_size = 1000000 
    
    print(f"\nDataset: {log_file}")
    print(f"Sample Size: {sample_size:,} log lines")

    
    # Step 1: Log Parsing
    print(f"\n{'='*20} STEP 1: LOG PARSING {'='*20}")
    print("Extracting information from raw log files")
    print("Grabbing the Following:")
    print("   - Timestamps, log levels, components")
    print("   - Block IDs, session information")
    print("   - Message content and metadata")
    
    start_time = time.time()
    parser = HDFSLogParser(log_file)
    parsed_logs = parser.parse_logs(max_lines=sample_size)
    parse_time = time.time() - start_time
    
    if not parsed_logs:
        print("Parsing failed. Exiting demo.")
        return
    
    print(f"\n{'#'*10} Parsing Complete {'#'*10}")
    print(f"Successfully parsed: {len(parsed_logs):,} logs")
    
    # Step 2: Sequence Creation
    print(f"\n{'='*20} STEP 2: SEQUENCE CREATION {'='*20}")
    print("Grouping logs into sequences...")
    print("   - sequences of blocks -- HDFS block ID)")
    print("   - sequences of Time frmaes  (by time windows)")
    print("   - Maintaining chronological order")
    
    start_time = time.time()
    sequence_creator = SequenceCreator(parsed_logs)
    
    # Create block sequences
    block_sequences = sequence_creator.create_block_sequences()
    
    time_sequences = sequence_creator.create_time_based_sequences(60)
    
    all_sequences = {**block_sequences, **time_sequences}
    sequence_time = time.time() - start_time
    
    print(f"\n{'#'*10} Sequence Creation Complete {'#'*10}")
    print(f"Block sequences: {len(block_sequences):,}")
    print(f"Time sequences: {len(time_sequences):,}")
    print(f"Total sequences: {len(all_sequences):,}")
    
    # Step 3: Analysis and Insights
    print(f"\n{'='*20} STEP 3: ANALYSIS {'='*20}")
    print("Analyzing logs and sequences for patterns...")
    
    start_time = time.time()
    analyzer = LogAnalyzer(parsed_logs, all_sequences)
    analyzer.basic_statistics()
    analyzer.sequence_statistics()
    analysis_time = time.time() - start_time
    
    print(f"Analysis Complete!")
 
    
    # Calculate some ML-relevant metrics
    total_logs = len(parsed_logs)
    total_sequences = len(all_sequences)
    avg_sequence_length = sum(seq['length'] for seq in all_sequences.values()) / total_sequences
    
    print(f"Data Statistics for ML:")
    print(f"Total training samples: {total_sequences:,}")
    print(f"Average sequence length: {avg_sequence_length:.1f} logs")
    print(f"{total_logs} individual log events")
    
   
    # Save demo results
    demo_results = {
        'demo_timestamp': datetime.now().isoformat(),
        'sample_size': sample_size,
        'total_logs_parsed': total_logs,
        'total_sequences': total_sequences,
        'block_sequences': len(block_sequences),
        'time_sequences': len(time_sequences),
        'avg_sequence_length': avg_sequence_length,
        'parse_time': parse_time,
        'sequence_time': sequence_time,
        'analysis_time': analysis_time,
    }
    
    with open('demo_results.json', 'w') as f:
        json.dump(demo_results, f, indent=2, default=str)
    
    print(f"Results saved to: demo_results.json")
    
def main():
    """Main demo function."""
    try:
        pipeline()
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
