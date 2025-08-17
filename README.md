# HDFS Log Anomaly Detection Pipeline

A comprehensive pipeline for detecting anomalies in HDFS logs using Drain3 log template mining and sequence analysis.

## Overview

This pipeline processes HDFS logs through several stages:
1. **Log Parsing**: Uses Drain3 to extract log templates and assign template IDs
2. **Sequence Creation**: Groups logs into block-level and time-window sequences
3. **DataFrame Conversion**: Converts sequences back to DataFrame format with your requested columns
4. **Dataset Building**: Creates training datasets with anomaly labels for ML models
5. **Output Generation**: Saves all outputs to the `/output` directory

## Features

- **YAML Configuration**: All parameters configurable via `config.yaml` file
- **No Command Line Arguments**: Simple execution with just `python hdfs_anomaly_detection.py`
- **Drain3 Integration**: Advanced log template mining with configurable similarity thresholds
- **Dual Sequence Types**: 
  - Block-level sequences (logs related to same HDFS block)
  - Time-window sequences (sliding time windows of configurable size)
- **Flexible Output**: Multiple output formats for different use cases
- **Anomaly Label Support**: Integrates with labeled datasets for supervised learning
- **Modular Design**: Clean, reusable classes in separate files
- **Clean Output Structure**: All generated files saved to `/output` directory

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd LogAnomalyDetection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
LogAnomalyDetection/
├── hdfs_anomaly_detection.py    # Main pipeline script
├── sequence_creator.py           # SequenceCreator class
├── dataset_builder.py            # DatasetBuilder class
├── config_utils.py               # Configuration utilities
├── utils.py                      # General utility functions
├── log_parser.py                 # Log parsing functionality
├── config.yaml                   # Configuration file
├── requirements.txt              # Python dependencies
├── Data/                         # Input data directory
│   └── hdfs_v1/
│       └── HDFS.log             # HDFS log file
├── output/                       # Output directory (auto-created)
└── README.md                     # This file
```

## Configuration

The pipeline uses a YAML configuration file (`config.yaml`) for all parameters:

```yaml
# Input/Output Configuration
input:
  log_file: "Data/hdfs_v1/HDFS.log"
  labels_file: "Data/hdfs_v1/anomaly_label.csv"
  max_lines: 1000000  # Set to null for all lines, or specify a number for testing

output:
  format: "csv"        # Options: csv, pickle, json
  prefix: "hdfs_sequences"
  output_dir: "output" # Output directory for all generated files

# Drain3 Configuration
drain3:
  persistence_file: "drain3_state.json"
  similarity_threshold: 0.4      # Similarity threshold for template matching
  tree_depth: 4                  # Tree depth for template mining
  max_children: 100              # Max children per node
  max_clusters: 1000             # Max number of clusters

# Sequence Creation Configuration
sequences:
  time_window:
    window_size_seconds: 60      # Size of each time window
    overlap_seconds: 0           # Overlap between consecutive windows
```

## Usage

### Simple Execution

Run the complete pipeline with default configuration:
```bash
python hdfs_anomaly_detection.py
```

That's it! No command line arguments needed. All settings are controlled via `config.yaml`.

### Custom Configuration

1. **Edit the YAML file** (`config.yaml`) to change parameters
2. **Run the pipeline**:
```bash
python hdfs_anomaly_detection.py
```

### Using Individual Modules

For programmatic use, import the modular classes:

```python
from sequence_creator import SequenceCreator
from dataset_builder import DatasetBuilder
from config_utils import load_config
from utils import load_anomaly_labels, ensure_output_directory

# Load configuration
config = load_config("config.yaml")

# Create sequences
sequence_creator = SequenceCreator(parsed_logs, config)
block_sequences = sequence_creator.group_by_block()
time_sequences = sequence_creator.group_by_time_window()

# Get DataFrames
logs_df = sequence_creator.get_logs_dataframe()
block_logs_df = sequence_creator.get_block_sequences_dataframe()
time_logs_df = sequence_creator.get_time_sequences_dataframe()

# Build dataset
dataset_builder = DatasetBuilder(all_sequences, anomaly_labels, config)
training_dataset = dataset_builder.build_training_dataset()
```

## Pipeline Stages

### 1. Log Parsing with Drain3

The `LogParser` class:
- Parses HDFS log format (MMDDYY HHMMSS milliseconds INFO component: message)
- Extracts timestamps, log levels, components, and block IDs
- Uses Drain3 to generate template IDs and template strings
- Handles incremental processing with state persistence
- **Configurable via YAML**: similarity threshold, tree depth, max clusters

### 2. Sequence Creation

The `SequenceCreator` class creates two types of sequences:

**Block Sequences:**
- Groups logs by HDFS block ID
- Maintains temporal order within each block
- Extracts template ID sequences for ML training
- Automatically sorts logs by timestamp

**Time-Window Sequences:**
- Creates sliding time windows (configurable size and overlap)
- Groups logs within each time window
- Useful for detecting temporal patterns and anomalies
- Configurable via YAML: window size and overlap

### 3. DataFrame Conversion

**NEW FEATURE**: The pipeline now converts sequences back to DataFrame format with exactly the columns you requested:

- `timestamp` - Log timestamp
- `level` - Log level (INFO, WARN, ERROR, etc.)
- `component` - HDFS component name
- `message` - Log message content
- `block_id` - HDFS block identifier
- `sequence_id` - Sequence identifier (replaces sessionid)
- `raw_line` - Original raw log line

Plus additional useful columns:
- `template_id` - Drain3 template ID
- `template_str` - Drain3 template string
- `sequence_type` - Type of sequence (block or time_window)

### 4. Dataset Building

The `DatasetBuilder` class:
- Combines sequences with anomaly labels
- Creates feature-rich datasets for ML models
- Supports multiple output formats (CSV, Pickle, JSON)
- Handles datetime serialization for JSON output

## Generated Output Files

The pipeline generates these files in the `/output/` directory:

1. **`hdfs_sequences.csv`** - Main sequences dataset for model training
2. **`hdfs_sequences_logs.csv`** - All logs converted back to DataFrame format
3. **`hdfs_sequences_block_logs.csv`** - Block sequence logs only

## Output Format Examples

### Sequences Dataset (for ML training)
```csv
sequence_id,type,length,template_sequence,template_sequence_str,start_time,end_time,time_span,anomaly_label
block_123,block,5,"[1,2,3,4,5]","1,2,3,4,5",2008-08-11 20:35:18,2008-08-11 20:35:25,7.0,0
time_0,time_window,12,"[1,2,1,3,4,2,1,5,6,7,8,9]","1,2,1,3,4,2,1,5,6,7,8,9",2008-08-11 20:35:00,2008-08-11 20:36:00,60.0,1
```

### Logs DataFrame (for analysis)
```csv
timestamp,level,component,message,block_id,sequence_id,raw_line,template_id,template_str,sequence_type
2008-08-11 20:35:18,INFO,dfs.DataNode$DataXceiver,Receiving block blk_123...,123,block_123,081109 203518...,1,Receiving block blk_<*>...,block
```

## Key Benefits of New Structure

1. **No Command Line Arguments**: Simple execution with just `python hdfs_anomaly_detection.py`
2. **Modular Design**: Each class in its own file for better organization
3. **Clean Outputs**: All files organized in `/output` directory
4. **Maintained Functionality**: All sequence-to-DataFrame conversion features preserved
5. **Easy Customization**: Modify individual modules without affecting others
6. **Better Maintainability**: Cleaner code structure and separation of concerns
7. **Incremental Learning**: Drain3 state persistence for consistent template IDs across runs

## Configuration Options

### Drain3 Settings

```yaml
drain3:
  similarity_threshold: 0.4      # Lower = more strict template matching
  tree_depth: 4                  # Deeper = more detailed templates
  max_children: 100              # Max children per node
  max_clusters: 1000             # Max number of clusters
```

### Sequence Settings

```yaml
sequences:
  time_window:
    window_size_seconds: 60      # Time window size in seconds
    overlap_seconds: 0           # Overlap between windows in seconds
```

### Output Settings

```yaml
output:
  format: "csv"                  # Output format: csv, pickle, or json
  prefix: "hdfs_sequences"       # Prefix for output filenames
  output_dir: "output"           # Directory for all output files
```

## Use Cases

### Training Phase (HDFS v1)
1. Configure parameters in `config.yaml`
2. Run `python hdfs_anomaly_detection.py`
3. Get labeled dataset for ML model training
4. Use logs DataFrame for analysis and visualization
5. **Drain3 state is automatically saved** to `drain3_state.json`

### Inference Phase (HDFS v2/v3)
1. Use same configuration and Drain3 state
2. Parse new logs with same parameters
3. Create sequences using same settings
4. Apply trained models for anomaly detection
5. **Template IDs remain consistent** across runs due to state persistence

## Drain3 State Persistence

### Why It's Important
- **Template Consistency**: Same log pattern always gets same template ID
- **Incremental Learning**: Each run builds upon previous knowledge
- **Production Ready**: Can process new logs while maintaining learned patterns

### How It Works
1. **Automatic Saving**: Drain3 saves state after processing each log line
2. **State File**: `drain3_state.json` (configured in `config.yaml`)
3. **Persistent Learning**: Templates learned in first run are available in subsequent runs
4. **Template ID Stability**: Critical for sequence analysis and anomaly detection

### Example Workflow
```
Run 1 (HDFS v1): Learn templates → Save state to drain3_state.json
Run 2 (HDFS v2): Load state → Add new templates → Update state
Run 3 (HDFS v3): Load state → Add new templates → Update state
```

**Result**: Template IDs remain consistent across all runs!

## Performance Considerations

- **Memory Usage**: Large log files may require processing in batches
- **Processing Speed**: Drain3 is optimized for speed but consider `max_lines` for testing
- **Storage**: Template sequences can be large; consider compression for long sequences
- **Incremental Processing**: Drain3 state persistence allows resuming interrupted processing

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce `max_lines` in config.yaml
2. **Template Quality**: Adjust Drain3 similarity threshold in config
3. **Sequence Length**: Very long sequences may need windowing or truncation
4. **Label Mismatch**: Ensure sequence IDs in labels file match generated sequence IDs
5. **Configuration Errors**: Check YAML syntax and parameter names

### Configuration Validation

The pipeline will:
- Load YAML configuration automatically
- Validate parameter ranges
- Use defaults for missing parameters
- Show loaded configuration at startup
- Create output directory if it doesn't exist

## Example Run Output

```
HDFS Log Anomaly Detection Pipeline
============================================================
Loading configuration from config.yaml...
Configuration loaded successfully
Log file: Data/hdfs_v1/HDFS.log
Output directory: output
============================================================

Step 1: Parsing logs with Drain3...
Parsing complete!
Total lines processed: 1,000,000
Successfully parsed: 1,000,000
Unique templates found: 35

Step 2: Creating sequences...
Created 10,735 block sequences
Created 17 time-window sequences

Step 3: Converting sequences to DataFrame format...
Created DataFrame with 38,036 log entries and 10 columns
Columns: ['timestamp', 'level', 'component', 'message', 'block_id', 'sequence_id', 'raw_line', 'template_id', 'template_str', 'sequence_type']

Step 4: Loading anomaly labels...
Step 5: Building and saving dataset...

PIPELINE SUMMARY
============================================================
Total logs parsed: 1,000,000
Total sequences: 10,752
Logs DataFrame created: 38,036 entries
All outputs saved to directory: output
Pipeline complete! Ready for anomaly detection model training.
```

## Next Steps

After running the pipeline:
1. **Feature Engineering**: Create additional features from template sequences
2. **Model Training**: Use scikit-learn or other ML frameworks
3. **Evaluation**: Assess model performance on test data
4. **Deployment**: Apply models to new, unlabeled logs
5. **Analysis**: Use the logs DataFrame for log analysis and visualization

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the pipeline.

## License

[Add your license information here]
