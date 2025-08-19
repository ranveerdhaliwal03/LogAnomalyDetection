#!/usr/bin/env python3
"""
Simple CLI script for running HDFS anomaly detection models.
This script provides an easy way to run different models with clear options.
"""

import sys
import os
import argparse
from pathlib import Path

def main():
    """Simple CLI interface for anomaly detection."""
    parser = argparse.ArgumentParser(
        description='HDFS Log Anomaly Detection - Simple CLI Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run both models (default)
  python run_anomaly_detection.py
  
  # Run only block-level supervised detection
  python run_anomaly_detection.py --block-only
  
  # Run only window-level unsupervised detection
  python run_anomaly_detection.py --window-only
  
  # Run with custom configuration
  python run_anomaly_detection.py --config my_config.yaml
  
  # Limit parsing to first 500k lines
  python run_anomaly_detection.py --max-lines 500000
        """
    )
    
    parser.add_argument('--block-only', action='store_true', 
                       help='Run only block-level supervised anomaly detection')
    parser.add_argument('--window-only', action='store_true', 
                       help='Run only window-level unsupervised anomaly detection')
    parser.add_argument('--config', default='config.yaml', 
                       help='Configuration file path (default: config.yaml)')
    parser.add_argument('--max-lines', type=int, 
                       help='Maximum lines to parse (overrides config)')
    parser.add_argument('--output-dir', 
                       help='Output directory (overrides config)')
    
    args = parser.parse_args()
    
    # Determine which model to run
    if args.block_only and args.window_only:
        print("Error: Cannot specify both --block-only and --window-only")
        sys.exit(1)
    
    if args.block_only:
        model = 'block_supervised'
    elif args.window_only:
        model = 'window_unsupervised'
    else:
        model = 'both'
    
    # Build command for the main pipeline
    cmd = ['python', 'hdfs_anomaly_detection.py', '--model', model]
    
    if args.config != 'config.yaml':
        cmd.extend(['--config', args.config])
    
    if args.max_lines:
        cmd.extend(['--max-lines', str(args.max_lines)])
    
    # Print what we're about to run
    print("="*60)
    print("HDFS ANOMALY DETECTION CLI")
    print("="*60)
    print(f"Model(s): {model}")
    print(f"Config: {args.config}")
    if args.max_lines:
        print(f"Max lines: {args.max_lines:,}")
    print(f"Command: {' '.join(cmd)}")
    print("="*60)
    
    # Ask for confirmation
    response = input("\nProceed with execution? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("Execution cancelled.")
        sys.exit(0)
    
    # Execute the main pipeline
    print("\nExecuting main pipeline...")
    os.system(' '.join(cmd))
    
    print("\nCLI execution completed!")


if __name__ == "__main__":
    main()
