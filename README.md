# AI-Based Consumer Electronics Market Intelligence & Opportunity Analyzer

> **Production MLOps Pipeline on AWS** · Focus: Wireless Audio & Consumer Electronics  
> Group No. 10 · Department of Artificial Intelligence and Data Science

---

## 1. Overview

A decision-support system that tells a company whether launching a new consumer electronics product (e.g., Wireless TWS Earbuds) represents a high-potential market opportunity. The platform collects and unifies omni-channel market data (prices, customer reviews, rating velocity, hardware specifications), predicts measurable signals like **30-day demand growth**, and combines them into an explainable **High / Medium / Low** opportunity assessment with competitive pricing bands and sentiment analysis.

The second half of the project is automated **MLOps on AWS**: data collection, S3 partitioned ingestion, SageMaker feature engineering, model training with XGBoost, fixed test-set evaluation, champion-challenger quality gates, and model registry packaging.

---

## 2. Architecture

```
DATA     EventBridge → Lambda → S3 (raw/date=YYYY-MM-DD/) → SageMaker Processing (Feature Engineering)
                                                                    │
TRAIN    Model Registry ← Quality Gate (RMSE Gate) ← Evaluation ← Training Job (XGBoost)
              │
SERVE    SageMaker Endpoint → API Gateway (+Lambda) → Dashboard → CloudWatch (Drift Monitor)
```

### Pipeline Workflow
1. **S3 Ingestion (`s3_ingest.py`)**: Partitions raw market snapshots by date (`s3://<bucket>/raw/date=YYYY-MM-DD/`) with schema validation and metadata tracking. Compatible with AWS Lambda and local simulation.
2. **Feature Engineering (`preprocessing.py`)**: Cleans currency/ratings, extracts 27 numerical/spec/sentiment features, simulates 30-day demand growth proxy, and prepares SageMaker-ready train/validation/test splits.
3. **Model Training (`train_xgboost.py`)**: Trains an XGBoost Regressor on SageMaker container infrastructure.
4. **Evaluation (`evaluate.py`)**: Evaluates the challenger model against the test set, computing RMSE, MAE, and R², outputting `evaluation.json`.
5. **Quality Gate (`sagemaker_pipeline.py`)**: ConditionStep verifies that challenger RMSE meets required threshold (e.g., RMSE ≤ 12.0) before approving for SageMaker Model Registry.
6. **Decision Engine & Demo (`run_pipeline.py`)**: Combines predicted growth, competitor pricing bands, and customer sentiment into business recommendations.

---

## 3. Dataset Overview

**Unified Master Dataset:** [`consumer_electronics_market_20k.csv`](consumer_electronics_market_20k.csv)  
- **Total Records:** 20,237 products across 17 unified schema attributes
- **Omni-Channel Coverage:**
  - **Amazon India (Online Marketplace):** 9,000 products
  - **Croma (Offline Retail Chain - Tata Infiniti Retail):** 8,500 products
  - **Flipkart (Online Marketplace):** 1,861 products
  - **Offline Electronics Market (Dealer MOP):** 876 products
- **Audio / Earbuds / Headphones Focus Subset:** 9,409 products

### Extracted Feature Groups
- **Price Features:** `discounted_price`, `actual_price`, `discount_pct`, `price_gap`, `price_vs_median`, `price_tier`
- **Customer Response:** `rating`, `rating_count`, `log_rating_count`, `review_rating_score`, `rating_count_share`
- **Hardware Specs (Regex Parsed):** `is_wireless`, `has_anc`, `has_enc`, `has_mic`, `fast_charge`, `low_latency`, `voice_assistant`, `bt_version`, `ipx_rating`, `driver_mm`, `playtime_hrs`
- **Sentiment Signals:** NLTK VADER compound sentiment score & positive review ratio
- **Target Variable:** 30-day review count growth (% demand proxy)

---

## 4. Repository Structure

```
├── consumer_electronics_market_20k.csv  # Unified 20k multi-channel dataset
├── s3_ingest.py                         # S3 ingestion with Lambda handler & local fallback
├── preprocessing.py                     # SageMaker Processing feature engineering step
├── train_xgboost.py                     # SageMaker XGBoost training step
├── evaluate.py                          # Evaluation step producing evaluation.json
├── sagemaker_pipeline.py                # AWS SageMaker Pipeline DAG JSON definition
├── run_pipeline.py                      # Master MLOps pipeline orchestrator
├── verify_dataset.py                    # Dataset validation and integrity check
├── generate_dataset.py                  # Dataset generation and consolidation script
├── PROJECT.md                           # Detailed project design document
├── requirements.txt                     # Python dependencies
└── README.md                            # Documentation
```

---

## 5. Quickstart & Execution

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run the Complete MLOps Pipeline Locally
Executes all 5 stages end-to-end (Ingestion -> Feature Engineering -> Training -> Evaluation -> Quality Gate -> Demo):
```bash
python run_pipeline.py --local
```

### Deploy to AWS S3 & SageMaker
```bash
python run_pipeline.py --bucket <your-s3-bucket> --role-arn <your-sagemaker-role-arn>
```

### Validate Dataset Integrity
```bash
python verify_dataset.py
```

---

## 6. Model Evaluation Results

| Metric | Model Performance | Benchmark Target | Status |
|---|---|---|---|
| **RMSE** | **2.55** | < 12.00 | **Passed Quality Gate** |
| **MAE** | **2.04** | < 10.00 | **Low Average Error** |
| **R² Score** | **0.902** | > 0.80 | **High Variance Explained** |
| **Relative Error** | **9.1%** | < 15.0% | **Production-Ready** |
