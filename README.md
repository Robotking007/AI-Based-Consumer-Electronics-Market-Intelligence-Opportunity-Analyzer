# AI-Based Consumer Electronics Market Intelligence & Opportunity Analyzer

> **Production MLOps & Orchestration Pipeline** · Focus: True Wireless Stereo (TWS) Earbuds  
> Group No. 10 · Department of Artificial Intelligence and Data Science

---

## 1. Overview

An enterprise decision-support and opportunity analyzer tailored specifically for the **True Wireless Stereo (TWS) Earbuds** consumer electronics category. The platform ingests omni-channel market data (pricing, specifications, customer reviews, ratings), extracts 27 high-signal hardware, sentiment, and elasticity features, predicts **30-day demand growth**, and determines actionable market entry sweet spots.

The platform includes two industry-standard MLOps orchestration workflows:
1. **Apache Airflow Pipeline (`dags/consumer_electronics_airflow_dag.py`)**: End-to-end DAG orchestrating data availability, 80:20 feature engineering, model training, accuracy evaluation, quality gate branching ($\ge 92\%$), automated S3 storage, and model registration.
2. **AWS SageMaker & S3 MLOps (`sagemaker_pipeline.py` & `train_and_upload_s3.py`)**: S3 partitioned ingestion, SageMaker containerized feature extraction, XGBoost training, and artifact publishing.
3. **Kaggle Training Package (`kaggle_train_model.ipynb` & `kaggle_train.py`)**: Ready-to-run Kaggle notebook and standalone script for training on cloud accelerators and syncing directly to AWS S3.

---

## 2. Pipeline Architecture

```
                                  [ 50k Pure TWS Master Dataset ]
                                                 │
                                                 ▼
             ┌───────────────────────────────────────────────────────────────────────┐
             │            Apache Airflow DAG / AWS SageMaker Pipeline               │
             └───────────────────────────────────────────────────────────────────────┘
                                                 │
                     ┌───────────────────────────┴───────────────────────────┐
                     ▼                                                       ▼
          [ check_data_availability ]                             [ S3 Date Ingestion ]
                     │                                                       │
                     ▼                                                       ▼
       [ feature_engineering_80_20 ]                             [ SageMaker Processing ]
          (27 Specs, Sentiment, Price)                               (Train: 40k, Test: 10k)
                     │                                                       │
                     ▼                                                       ▼
         [ train_xgboost_model ]                                  [ SageMaker Training ]
          (Tuned XGBoost Regressor)                                  (Built-in Container)
                     │                                                       │
                     ▼                                                       ▼
        [ evaluate_model_accuracy ]                               [ Model Evaluation ]
          (R²: 94.00%, RMSE: 1.1922)                                 (evaluation.json)
                     │                                                       │
                     ▼                                                       ▼
        [ quality_gate_decision ] ──── Accuracy < 92% ────────► [ send_failure_alert ]
                     │
              Accuracy >= 92%
                     │
                     ▼
            [ store_model_in_s3 ]  ──►  s3://consumer-electronics-mlops-533267153786/models/latest/
                     │
                     ▼
         [ register_approved_model ]
```

---

## 3. 50,000 Pure TWS Master Dataset

* **Dataset File:** [`consumer_electronics_market.csv`](consumer_electronics_market.csv) (89.55 MB)
* **Dataset Generator:** [`build_50k_tws_dataset.py`](build_50k_tws_dataset.py)
* **Category Focus:** **100% True Wireless Stereo (TWS) Earbuds** (`Electronics|Headphones,Earbuds&Accessories|Headphones|In-Ear`). Exclusively TWS; zero cables, chargers, smartphones, or unrelated appliances.
* **Sorting Hierarchy:** Dataset is **cleanly sorted** by:
  $$\text{Category} \longrightarrow \text{Brand Name} \longrightarrow \text{Discounted Price}$$
* **Data Quality:** Zero missing/null values across all 17 schema columns, 50,000 unique product names, and sequential IDs (`TWS_000001` to `TWS_050000`).
* **Multi-Channel Market Distribution:**
  - **Amazon India (Online Marketplace):** 17,049 products (34.1%)
  - **Croma (Offline Retail Network - Tata Infiniti):** 13,599 products (27.2%)
  - **Offline Electronics Dealer MOP Network:** 10,958 products (21.9%)
  - **Flipkart (Online Marketplace):** 8,394 products (16.8%)

### 27 Extracted Feature Groups (80:20 Split)

* **TWS Acoustic & Hardware Specs (Regex Extracted):** Active Noise Cancellation (`has_anc`), Environmental Noise Cancellation (`has_enc`), Mic system (`has_mic`), Fast Charging (`fast_charge`), Low Latency Mode (`low_latency`), Voice Assistant (`voice_assistant`), Bluetooth Version (`bt_version` 5.0–5.4), IPX Waterproof Rating (`ipx_rating` 0–8), Audio Driver Diameter (`driver_mm`), Total Battery Playtime (`playtime_hrs`).
* **Price & Elasticity Signals:** `discounted_price`, `actual_price`, `discount_pct`, `price_gap`, `price_vs_median`, and categorical price tiers (`Budget`, `Mid-Tier`, `Flagship`).
* **Customer Response & NLP Sentiment:** Star rating, rating volume, log-scaled rating count, NLTK VADER compound sentiment score, positive review sentiment ratio, and negative sentiment ratio.
* **Target Variable:** 30-day market demand growth percentage (`demand_growth_pct`).

---

## 4. Model Performance & Mathematical Proof of Accuracy

The model was trained strictly on an **80:20 split** ($40,000$ training samples and $10,000$ unseen holdout test samples) using a tuned **XGBoost Regressor** (`n_estimators=180`, `max_depth=6`, `learning_rate=0.06`).

### Model Evaluation Scorecard

| Metric | Model Performance | Benchmark Requirement | Quality Gate Status |
| :--- | :---: | :---: | :---: |
| **Accuracy ($R^2 \times 100\%$)** | **94.00%** | $\ge \mathbf{92.00\%}$ | **PASSED** |
| **Coefficient of Determination ($R^2$)** | **0.9400** | $\ge 0.9200$ | **PASSED** |
| **Root Mean Squared Error (RMSE)** | **1.1922** | $\le 12.00$ | **PASSED** |
| **Mean Absolute Error (MAE)** | **0.9491** | $\le 10.00$ | **PASSED** |
| **Relative Error** | **2.46%** | $\le 15.0\%$ | **PASSED** |
| **Relative Closeness Accuracy** | **97.54%** | $\ge 85.0\%$ | **PASSED** |

### Mathematical Proof of Accuracy Using Formulas

Evaluated on the $N = 10,000$ test set from [`processed_data/test.csv`](processed_data/test.csv) with mean target $\bar{y} = 38.5450$:

1. **Total Sum of Squares ($SS_{\text{tot}}$)**:
   $$SS_{\text{tot}} = \sum_{i=1}^{N} (y_i - \bar{y})^2 = \mathbf{236,975.9204}$$

2. **Residual Sum of Squares ($SS_{\text{res}}$)**:
   $$SS_{\text{res}} = \sum_{i=1}^{N} (y_i - \hat{y}_i)^2 = \mathbf{14,213.3099}$$

3. **Coefficient of Determination ($R^2$ / Accuracy)**:
   $$R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}} = 1 - \frac{14,213.3099}{236,975.9204} = 1 - 0.059978 = \mathbf{0.940022}$$
   $$\mathbf{\text{Accuracy Percentage} = R^2 \times 100\% = 94.00\%}$$

4. **Root Mean Squared Error (RMSE)**:
   $$\text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N} (y_i - \hat{y}_i)^2} = \sqrt{\frac{14,213.3099}{10,000}} = \mathbf{1.1922}$$

5. **Mean Absolute Error (MAE)**:
   $$\text{MAE} = \frac{1}{N}\sum_{i=1}^{N} |y_i - \hat{y}_i| = \mathbf{0.9491}$$

6. **Relative Closeness**:
   $$\text{Relative Error} = \left(\frac{0.9491}{38.5450}\right) \times 100\% = 2.46\% \implies \mathbf{\text{Closeness} = 97.54\%}$$

All certified metrics are saved in [`evaluation_output/evaluation.json`](evaluation_output/evaluation.json).

---

## 5. AWS S3 Storage & Artifacts

All approved models and artifacts are automatically synced to the production AWS S3 bucket:
* **S3 URI:** `s3://consumer-electronics-mlops-533267153786/models/latest/`
* **AWS Region:** `ap-south-1` (Mumbai)
* **Available Artifacts:**
  - `model.joblib`: Production scikit-learn/joblib serialized model
  - `model.pkl`: Standard pickle serialized model
  - `xgboost-model.json`: Native XGBoost booster format
  - `model.tar.gz`: SageMaker-compatible container deployment bundle
  - `evaluation.json`: Certified quality gate regression metrics
  - `feature_metadata.json`: Feature baselines and distribution statistics

---

## 6. Repository Structure

```
├── consumer_electronics_market.csv     # 50,000 pure TWS master dataset (cleanly sorted)
├── build_50k_tws_dataset.py            # Master dataset generator & TWS synthesizer
├── preprocessing.py                    # 80:20 train/test split & 27 feature extractor
├── train_xgboost.py                    # XGBoost training script with S3 upload flag
├── evaluate.py                         # Model evaluation script producing evaluation.json
├── train_and_upload_s3.py              # Automated 80:20 training & direct S3 publisher
├── kaggle_train.py                     # Standalone Kaggle script (cloud training & S3 export)
├── kaggle_train_model.ipynb            # Kaggle step-by-step Jupyter Notebook
├── dags/
│   └── consumer_electronics_airflow_dag.py # Apache Airflow DAG (8 tasks, quality gate branching)
├── test_airflow_pipeline.py            # Apache Airflow integration test suite
├── sagemaker_pipeline.py               # AWS SageMaker Pipeline DAG JSON definition
├── mock_sagemaker_runner.py            # Local SageMaker simulator & DAG runner
├── test_mock_pipeline.py               # Automated pytest suite for SageMaker pipeline
├── app.py                              # FastAPI backend & Decision Intelligence Dashboard
├── requirements.txt                    # Project dependencies
└── README.md                           # Documentation
```

---

## 7. Quickstart & Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Apache Airflow Verification Suite
Validates the Airflow DAG structure, topological task ordering, end-to-end task callables, and quality gate threshold enforcement ($\ge 92\%$):
```bash
python test_airflow_pipeline.py
```

### 3. Run Automated Pytest Suite
```bash
pytest test_mock_pipeline.py -v
```

### 4. Train Model with 80:20 Split & Upload to S3
```bash
# Trains model on 80:20 split, achieves 94% accuracy, and uploads to AWS S3
python train_and_upload_s3.py
```

### 5. Run on Kaggle
1. Upload [`consumer_electronics_market.csv`](consumer_electronics_market.csv) to Kaggle as a dataset.
2. Open [`kaggle_train_model.ipynb`](kaggle_train_model.ipynb) in Kaggle Notebooks.
3. Add your AWS credentials in Kaggle Secrets (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`).
4. Execute all cells to train with an 80:20 split and export models to AWS S3.

### 6. Launch the Interactive Dashboard
```bash
python app.py
```
Open **[http://127.0.0.1:8080](http://127.0.0.1:8080)** to access the real-time opportunity analyzer, what-if demand simulator, acoustic radar charts, and live MLOps pipeline telemetry.

---

## 8. License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.
