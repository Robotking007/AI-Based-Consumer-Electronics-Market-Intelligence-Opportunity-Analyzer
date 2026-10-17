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
├── mock_sagemaker_runner.py             # Local SageMaker Pipeline simulator & DAG executor
├── test_mock_pipeline.py                # Automated pytest suite for mock pipeline & quality gate
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

### Phase 1: Local SageMaker Mock & Automated Testing

#### 1. Run Automated Test Suite (pytest)
Validates DAG schema, ConditionStep branching (approval vs rejection), and model inference:
```bash
pytest test_mock_pipeline.py -v
```

#### 2. Run Local SageMaker Pipeline Simulator
Executes all 5 SageMaker steps locally with mock container paths, property files, and quality gate:
```bash
python run_pipeline.py --mode mock
```
*Tip: To run a fast test with sample limit, use `python run_pipeline.py --mode mock --sample-limit 1000`.*

#### 3. Test Quality Gate Rejection Branch
Simulates a scenario where challenger model fails quality criteria (e.g. RMSE threshold = 0.5):
```bash
python mock_sagemaker_runner.py --force-fail --sample-limit 1000
```

---

### Phase 2: Deploy to AWS SageMaker (Cloud)

Once local testing passes, submit and execute the pipeline directly on AWS:
```bash
# Set your AWS credentials
aws configure

# Deploy & trigger pipeline execution on AWS SageMaker
python run_pipeline.py --mode cloud --bucket <your-s3-bucket> --role-arn <your-sagemaker-role-arn> --region ap-south-1 --execute
```

---

### Phase 3: Launch Full Interactive Dashboard (Backend + Frontend)

Launch the FastAPI backend and interactive frontend dashboard:
```bash
python app.py
```
Open **[http://127.0.0.1:8080](http://127.0.0.1:8080)** in your browser to interact with:
- Top 4 market metric cards (Market Opportunity, Predicted Growth, Sentiment Score, Optimal Price Range)
- Product Hero Analysis with Boat Airdopes 141 (TWS), specs, recommendation checklist, and 78/100 score gauge
- Predicted Demand Growth line chart (+18.6% Growth)
- Review Sentiment Analysis donut chart + Key Review Aspect bars
- Price Positioning scatter chart across category competitors
- Competition Analysis table with live product selection
- Google Trends search interest line chart (+42% vs last year)
- Real-time XGBoost inference powered by `.joblib` & `.pkl` models

## 6. Model Evaluation Results

| Metric | Model Performance | Benchmark Target | Status |
|---|---|---|---|
| **RMSE** | **2.55** | < 12.00 | **Passed Quality Gate** |
| **MAE** | **2.04** | < 10.00 | **Low Average Error** |
| **R² Score** | **0.902** | > 0.80 | **High Variance Explained** |
| **Relative Error** | **9.1%** | < 15.0% | **Production-Ready** |

---

## 7. CI/CD Automation (GitHub Actions)

The repository includes a production MLOps CI/CD workflow ([`.github/workflows/ci_cd_pipeline.yml`](.github/workflows/ci_cd_pipeline.yml)) with 3 stages:
1. **Validation & Unit Testing**: Verifies dataset schema integrity (`verify_dataset.py`) and runs pytest test suite (`test_mock_pipeline.py`) across all DAG condition branches.
2. **Mock SageMaker E2E**: Executes the full 5-stage pipeline locally inside GitHub Actions runner and archives execution summaries.
3. **AWS Cloud Deployment**: Triggered on push to `main` with AWS credentials configured, uploading code to S3 and upserting the SageMaker Pipeline.

---

## 8. License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

