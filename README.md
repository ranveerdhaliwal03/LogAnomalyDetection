# HDFS Log Anomaly Detection Pipeline

A comprehensive tool for detecting anomalies in HDFS logs using both supervised and unsupervised machine learning approaches.

## Features

### 🔧 **Fixed Issues**
- **Timestamp Parsing**: Fixed microsecond overflow errors when parsing logs with milliseconds > 999
- **Large Log Handling**: Improved performance for logs over 1 million lines

### 🚀 **New Features**
- **Dual Model Support**: 
  - **Block-level Supervised**: TF-IDF + Logistic Regression/XGBoost with anomaly labels
  - **Window-level Unsupervised**: Isolation Forest, One-Class SVM on sliding windows
- **Flexible Execution**: Run models separately or together
- **Enhanced CLI**: Easy model selection and configuration
- **Comprehensive Evaluation**: Precision/Recall/F1 metrics for labeled data

## Architecture

```
HDFS Logs → Drain3 Parsing → Sequence Creation → Feature Extraction → Model Training
                                    ↓
                            ┌─────────────────┬─────────────────┐
                            │ Block Sequences │ Window Sequences │
                            │ (Supervised)    │ (Unsupervised)  │
                            └─────────────────┴─────────────────┘
                                    ↓
                            ┌─────────────────┬─────────────────┐
                            │ TF-IDF +       │ TF-IDF +        │
                            │ Metadata       │ Metadata        │
                            └─────────────────┴─────────────────┘
                                    ↓
                            ┌─────────────────┬─────────────────┐
                            │ Logistic       │ Isolation       │
                            │ Regression     │ Forest          │
                            │ XGBoost        │ One-Class SVM   │
                            └─────────────────┴─────────────────┘
```

## Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd LogAnomalyDetection
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Verify installation**:
```bash
python -c "import drain3, sklearn, pandas, numpy; print('All dependencies installed!')"
```

## Quick Start

### 🎯 **Run Both Models (Default)**
```bash
python run_anomaly_detection.py
```

### 🔍 **Block-Level Supervised Only**
```bash
python run_anomaly_detection.py --block-only
```

### 📊 **Window-Level Unsupervised Only**
```bash
python run_anomaly_detection.py --window-only
```

### ⚙️ **Custom Configuration**
```bash
python run_anomaly_detection.py --config my_config.yaml --max-lines 500000
```

## Configuration

Edit `config.yaml` to customize:

```yaml
# Window-level detection settings
window_detection:
  window_size: 100               # Log entries per window
  overlap: 0                     # Overlap between windows
  tfidf_max_features: 1000      # TF-IDF feature limit
  contamination: 0.1             # Expected anomaly rate

# Block-level detection settings  
block_detection:
  tfidf_max_features: 1000      # TF-IDF feature limit
  test_size: 0.2                # Test set proportion
  validation_size: 0.1          # Validation set proportion
```

## Model Details

### 🔗 **Block-Level Supervised Model**
- **Purpose**: Detect anomalies in specific HDFS block operations
- **Features**: TF-IDF of template sequences + metadata (length, time span)
- **Models**: Logistic Regression, XGBoost, LightGBM
- **Requires**: Anomaly labels (`anomaly_label.csv`)
- **Output**: Binary classification (Normal/Anomaly)

### 🪟 **Window-Level Unsupervised Model**
- **Purpose**: Detect anomalous time windows in log streams
- **Features**: TF-IDF of template sequences + metadata (log count, time span, unique templates)
- **Models**: Isolation Forest, One-Class SVM, Local Outlier Factor
- **Requires**: No labels (fully unsupervised)
- **Output**: Anomaly scores and rankings

## Usage Examples

### 📁 **Basic Usage with HDFS v1 Dataset**
```bash
# Run both models on full dataset
python run_anomaly_detection.py

# Run only unsupervised model (no labels needed)
python run_anomaly_detection.py --window-only

# Limit to first 500k lines for testing
python run_anomaly_detection.py --max-lines 500000
```

### 🔧 **Advanced Usage**
```bash
# Direct pipeline execution
python hdfs_anomaly_detection.py --model window_unsupervised

# Custom configuration
python hdfs_anomaly_detection.py --config production_config.yaml

# Specific model with custom parameters
python hdfs_anomaly_detection.py --model block_supervised --max-lines 1000000
```

## Output Files

The pipeline generates several output files:

```
Output/
├── block_sequences.csv          # Block-level training dataset
├── window_sequences.csv         # Window-level dataset
├── hdfs_sequences_logs.csv     # All logs with sequence info
├── block_sequences_logs.csv    # Block sequence logs
├── window_sequences_logs.csv   # Window sequence logs
├── block_models/               # Trained block models
└── window_models/              # Trained window models
```

## Model Evaluation

### 📊 **Supervised Models (Block-level)**
- **Metrics**: Precision, Recall, F1-Score, ROC-AUC
- **Cross-validation**: Stratified k-fold
- **Class balancing**: Automatic handling of imbalanced data

### 📈 **Unsupervised Models (Window-level)**
- **Metrics**: Anomaly scores, top-N rankings
- **Evaluation**: Precision/Recall/F1 (if labels available)
- **Interpretability**: Feature importance and anomaly explanations

## Performance Tips

### 🚀 **For Large Logs (>1M lines)**
1. **Use `--max-lines`** to limit parsing for testing
2. **Adjust window size** based on your anomaly patterns
3. **Monitor memory usage** during TF-IDF vectorization

### ⚡ **Optimization Settings**
```yaml
# Reduce memory usage
tfidf_max_features: 500        # Instead of 1000+
window_size: 50                # Smaller windows for faster processing

# Improve accuracy
contamination: 0.05            # Lower for rare anomalies
tfidf_ngram_range: [1, 3]     # Include trigrams
```

## Troubleshooting

### ❌ **Common Issues**

1. **Memory Errors**: Reduce `tfidf_max_features` or `window_size`
2. **Slow Processing**: Use `--max-lines` for testing, adjust overlap
3. **Import Errors**: Ensure all dependencies are installed
4. **Timestamp Warnings**: Fixed in latest version, safe to ignore

### 🔍 **Debug Mode**
```bash
# Verbose logging
python -u hdfs_anomaly_detection.py --model window_unsupervised 2>&1 | tee debug.log
```

## Contributing

1. **Fork the repository**
2. **Create a feature branch**
3. **Add tests for new functionality**
4. **Submit a pull request**

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{hdfs_anomaly_detection,
  title={HDFS Log Anomaly Detection Pipeline},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/LogAnomalyDetection}
}
```

## Support

- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Email**: your.email@example.com

---

**Happy anomaly hunting! 🕵️‍♂️**
