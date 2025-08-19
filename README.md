# HDFS Log Anomaly Detection -- Deep Dive on Block‑Level and Window‑Level Models



A comprehensive tool for detecting anomalies in HDFS logs using both supervised and unsupervised machine learning approaches.
This project detects anomalies in Hadoop HDFS logs on two levels that complement each other:
  - Block‑level (micro view): “Is this block’s lifecycle normal?”
  - Window‑level (macro view): “Is this time slice of the system behaving normally?”

Both levels share the same parsing/encoding (Drain3 → log templates) so the models agree on what an “event” is. 
The block model is supervised on HDFS_v1 labels; the window model is unsupervised, so it generalizes to new datasets 
(e.g., HDFS_v2/v3) and surfaces system‑wide issues without ground truth.

#Why two levels?

**Block‑level Anomaly Detection (Supervised)**
**Problem definition:** Predict whether a block sequence (all events for a single block_id) is normal or anomalous.
**What:** All log lines associated to the same block_id (HDFS unit of work).
**Why:** Anomalies in HDFS benchmarks are labeled at the block level (via anomaly_label.csv). This enables supervised learning and objective metrics (Precision/Recall/F1/ROC‑AUC).
**Value:** Catches localized faults (e.g., a replication failure for a specific block) and provides clear, actionable context to engineers (the exact block lifecycle that went wrong).

At the block level, the system treats a sliding window of log messages as a single observation. Each raw log line is first parsed into a template ID (using **Drain3 parsing**) so that variable values like IP addresses or request IDs are abstracted away, leaving behind only the underlying event structure. These template IDs within each block are then vectorized using **TF-IDF**, which transforms the block into a sparse numerical feature vector representing the relative frequency and importance of each template. This approach was chosen because TF-IDF is lightweight, interpretable, and effective for capturing frequency-based signals while ignoring irrelevant noise.

For model training, I used a **Logistic Regression classifier**. The choice of Logistic Regression was deliberate: it is fast to train and provides a clear probability score for each block, which can be use to tune decision thresholds depending on whether we want higher recall (catching more anomalies) or higher precision (avoiding false alarms). Training is done by pairing each block with its anomaly label (from the benchmark dataset), and the model learns to differentiate between "normal" and "anomalous" blocks based on the distribution of templates.

This design decision gives us a strong baseline: it quickly highlights when a block diverges from typical system behavior without requiring sequence-aware models. However, because TF-IDF and Logistic Regression ignore order and temporal structure, anomalies that depend on subtle transitions between log events may go undetected. This limitation is precisely why we later extended the system to window-level sequence modeling — but block-level detection remains a simple, reliable foundation that works well for frequency-driven anomalies.


**Window‑level (macro)**
**Problem definition:** Score each time (or count) window for how “unusual” it is relative to normal behavior—without labels. Then highlight which lines/templates make it suspicious.
**What:** Sliding time or count windows (e.g., 60s or every 100 lines) across the entire log stream.
**Why:** Real outages can be system‑wide and may not map neatly to a single block_id. Also, new datasets (v2/v3) might lack labels, so we need unsupervised detection.
**Value:** Catches bursts, drifts, and rare traffic patterns across components, even when no labels are available, making it deployable on new domains.

While block-level detection provided a strong frequency-based baseline, it had a critical limitation: it completely ignored the temporal order of events. In real-world systems, many anomalies arise not from the presence of an unusual template itself, but from the sequence in which otherwise normal templates occur. For example, an authentication success immediately followed by an error may indicate suspicious behavior even if both events are individually common. To capture these patterns, we extended our approach to window-level sequence modeling.

In this setup, I treat each log block as an ordered sequence of events. Each block is represented by the series of template IDs in the order they appear, preserving temporal structure. To model this, we experimented with n-gram–based TF-IDF features, where bigrams and trigrams capture local transitions between templates. This simple extension allows the model to recognize unusual short-term sequences that would be invisible in unigram-only TF-IDF.

For the implementation, we used the Isolation Forest algorithm, which is well-suited for unsupervised anomaly detection in high-dimensional data. Each windowed sequence of logs was first transformed into a numerical feature vector using n-gram TF-IDF representations, capturing both the frequency and local ordering of templates. These vectors were then passed into Isolation Forest, which works by randomly partitioning feature space and isolating points that require fewer partitions. In practice, this means windows with unusual event patterns are scored as anomalies. This approach provides a computationally efficient, unsupervised way to identify rare or abnormal execution flows without requiring labeled data — a key requirement when moving from v1 logs (with anomaly labels) to v2/v3 logs (without labels).

While Isolation Forest provided a solid balance between scalability and interpretability, we also considered more sequence-aware models to enhance accuracy:

Markov Chains: Model the transition probabilities between log templates, flagging anomalies when unlikely or unseen transitions occur.

LSTM-based Predictors (DeepLog-style): Train a neural sequence model to predict the next event in a sequence; if the observed event is outside the predicted set, it is marked anomalous.

We chose Isolation Forest as the initial implementation because it is lightweight, interpretable, and robust across datasets, making it a practical starting point. However, the Markov and LSTM approaches represent natural next steps for improving the system’s ability to capture complex temporal dependencies.

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


## Usage Examples
# Full pipeline (parse → block + window datasets → train → score)
python run_anomaly_detection.py

# Block‑only supervised training & eval
python run_anomaly_detection.py --block-only


# Window‑only unsupervised training & scoring
python run_anomaly_detection.py --window-only


# Useful options (see config.yaml)
# --window-size, --window-overlap, --contamination, --topk, --save-path, etc.


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





## Citation
Jingwen Zhou, Zhenbang Chen, Ji Wang, Zibin Zheng, and Michael R. Lyu. TraceBench: An Open Data Set for Trace-oriented Monitoring, in Proceedings of the 6th IEEE International Conference on Cloud Computing Technology and Science (CloudCom), 2014.
Jieming Zhu, Shilin He, Pinjia He, Jinyang Liu, Michael R. Lyu. Loghub: A Large Collection of System Log Datasets for AI-driven Log Analytics. IEEE International Symposium on Software Reliability Engineering (ISSRE), 2023.
