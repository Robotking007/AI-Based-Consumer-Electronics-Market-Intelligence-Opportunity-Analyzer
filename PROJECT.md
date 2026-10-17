# AI-Based Consumer Electronics Market Intelligence & Opportunity Analyzer

> **MLOps project built on AWS** · Initial focus: Wireless Earbuds (TWS)
> Group No. 10 · Department of Artificial Intelligence and Data Science · Guide: Prof. Santosh Kumar

---

## 1. Overview

A decision-support system that tells a company whether launching a new pair of wireless earbuds is a good market opportunity. It collects market data (prices, ratings, reviews, competitors), predicts measurable signals such as future demand growth, and combines them into an explainable **High / Medium / Low** recommendation with reasons and risks.

The second half of the project is **MLOps**: data collection, training, evaluation, versioning, deployment, monitoring and retraining are all automated on AWS so the system stays accurate as the market changes.

### Two ideas that make this project different

1. **Measurable targets, not a made-up score.** We do not train a model to predict an "opportunity score" (0–100) because there is no ground truth for it. We predict things that can be verified later (e.g. review-count growth over the next 30 days) and apply a transparent rule layer on top.
2. **The model is never trusted blindly.** A new model (*challenger*) replaces the current one (*champion*) only if it beats it on a fixed test set. Retraining is triggered by evidence (drift + rising error), not just a timer.

---

## 2. Problem Statement

A company planning a new earbud product must answer four questions:

| Question | System output |
|---|---|
| Is demand growing? | Predicted 30-day demand growth (XGBoost) |
| How crowded is the market? | Competitor count, price concentration, top-brand dominance |
| What price range works? | Price band (25th–75th percentile of competitors) |
| What do customers like or dislike? | Sentiment analysis on reviews (VADER / DistilBERT) |

Manual analysis is slow, not reproducible and goes stale within weeks. This project builds a repeatable AI + MLOps pipeline that turns changing market signals into evidence-based intelligence.

---

## 3. Solution Design

| Output | Produced by | Why |
|---|---|---|
| Demand growth | XGBoost regression | Learns from many features; evaluated with MAE / RMSE |
| Price band | Statistics (P25–P75) | A simple calculation is more reliable than a model |
| Sentiment | Pretrained NLP model | Training our own adds little value |
| Competition | Count, price concentration, top-product dominance | Count alone is misleading |
| Final assessment | Rule-based decision layer + business context | Explainable; differs by company (budget, segment, target price) |

**Target variable:** % growth in review count over the next 30 days (a *demand proxy*, not actual sales).

---

## 4. Architecture

```
DATA     EventBridge → Lambda → S3 (raw + processed) → SageMaker Processing (validate + features)
                                                              │
TRAIN    Model Registry ← Quality Gate ← Evaluation ← Training Job (XGBoost)
              │
SERVE    SageMaker Endpoint → API Gateway (+Lambda) → Dashboard → CloudWatch (metrics + drift)

         ↻ A drift alarm in CloudWatch restarts the SageMaker Pipeline to train a challenger.
```

| # | Step | Service | What happens |
|---|---|---|---|
| 1 | Schedule | EventBridge | Daily cron rule |
| 2 | Collect | Lambda | Fetches / generates the day's snapshot |
| 3 | Store | S3 | `raw/date=YYYY-MM-DD/`; raw data never overwritten |
| 4 | Validate + features | SageMaker Processing | Quality checks; price position, review growth, competitor stats |
| 5 | Train | SageMaker Training Job | XGBoost on a time-based split |
| 6 | Evaluate | SageMaker Processing | MAE / RMSE on fixed test set → `evaluation.json` |
| 7 | Quality gate | Pipeline ConditionStep | Continue only if challenger beats champion |
| 8 | Register | Model Registry | New version with metrics; approval = champion |
| 9 | Deploy | SageMaker Endpoint | Approval triggers deployment |
| 10 | Serve | API Gateway + Lambda | Prediction + price band + sentiment + decision rules |
| 11 | Monitor | CloudWatch | Latency, errors, custom drift metric |

### Tools outside AWS
- **GitHub Actions** – CI/CD (tests on every push; OIDC for temporary AWS credentials)
- **MLflow** – experiment tracking
- **DVC** (optional) – dataset versioning

---

## 5. MLOps Practices

- **CI/CD:** code push → automated tests → Docker image to ECR → pipeline trains challenger → compare with champion. Any failing test stops deployment.
- **Champion–challenger:** promotion only if MAE/RMSE is lower on the *same fixed* test set.
- **Time-based split:** train on older dates, test on newer, to avoid future leakage.
- **Drift detection:** KS test / PSI between new data and training reference; score published to CloudWatch; alarm triggers the pipeline. Drift alone never forces deployment — the challenger must still pass the gate.
- **Validation:** schema, missing values, ranges (rating 1–5, price > 0), duplicates, freshness.
- **Reproducibility:** dated raw data in S3, pipeline-as-code, MLflow run records.
- **Cost control:** Lambda + serverless endpoints, training only when needed, AWS Budget alerts, teardown after demo.

---

## 6. Dataset & Feature Extraction

**Primary Dataset:** [`consumer_electronics_market_20k.csv`](file:///d:/College/3rd%20Year/Semester%205/Machine%20Learning%20and%20AI%20Optimization/Lab/Course%20Project/Code/consumer_electronics_market_20k.csv) — **20,237 rows × 17 columns**, covering both major online e-commerce platforms and offline brick-and-mortar retail markets across India:
- **Amazon India (Online Marketplace):** 9,000 products (includes the original 1,465 benchmark records preserved at the top + 7,535 verified consumer electronics)
- **Croma (Offline Retail Chain - Tata Infiniti Retail):** 8,500 products (smart TVs, home entertainment, displays, retail electronics)
- **Flipkart (Online Marketplace):** 1,861 products (pure consumer electronics, mobiles, wearables, audio, computing)
- **Offline Electronics Market (Dealer MOP):** 876 products (physical electronics dealer market network with MOP - Market Operating Pricing)
- **Audio / Earbuds / Headphones Focus Subset:** **9,409 products** (vs. 60 in the original prototype).

*Note: All data sources are fully unified into this single master CSV file.*

### 6.1 Raw columns in the dataset

| Column | Content | Used? |
|---|---|---|
| `product_id`, `product_name`, `category` | Identity and taxonomy | ✅ key, brand, form factor, sub-category |
| `discounted_price`, `actual_price`, `discount_percentage` | Prices as text (`₹1,299`, `40%`) | ✅ cleaned to numbers |
| `rating`, `rating_count` | Average stars; total ratings (`1,92,590`) | ✅ cleaned to numbers |
| `about_product` | `|`-separated bullet list of specs | ✅ spec extraction |
| `review_title`, `review_content` | Comma-joined reviews per product | ✅ sentiment |
| `market_channel` | Channel indicator (Amazon / Flipkart / Croma / Offline Dealer) | ✅ online vs. offline intelligence |
| `user_id`, `user_name`, `review_id` | Reviewer identifiers | ❌ not needed |
| `img_link`, `product_link` | URLs | ❌ not needed |

Cleaning applied: stripped `₹`, `,` and `%`; converted to numeric; zero missing values across critical price, rating, and identity columns.


### 6.2 Engineered features

**A. Price features**

| Feature | Description | Median / range |
|---|---|---|
| `discounted_price` | Selling price (₹) | median 894 · 149 – 4,790 |
| `actual_price` | List price (₹) | median 2,244 · 399 – 15,990 |
| `discount_pct` | Discount % | median 63 · 0 – 85 |
| `price_gap` | `actual − discounted` (₹) | median 1,250 |
| `price_vs_median` | Price ÷ market median price (price position) | 0.17 – 5.36 |
| `price_tier` | Budget <500 · Mid 500–1k · Upper 1k–2.5k · Premium >2.5k | categorical |

**B. Customer-response features**

| Feature | Description | Median / range |
|---|---|---|
| `rating` | Average star rating | 3.4 – 4.2 (mean 3.92) |
| `rating_count` | Number of ratings (popularity proxy) | median 45,148 · 206 – 363,713 |
| `log_rating_count` | `log(1 + rating_count)` – tames skew | 5.3 – 12.8 |
| `review_rating_score` | `rating × log_rating_count` – quality × popularity | 22.4 – 52.5 |
| `rating_count_share` | Product's share of all ratings in the market | 0 – 0.08 |

**C. Product-spec features (parsed from name + bullets)**

| Feature | Description | Coverage / value |
|---|---|---|
| `form_factor` | TWS · Neckband · Wired · Other | 18 TWS · 6 Neckband · 27 Wired · 9 Other |
| `is_wireless` | Bluetooth / wireless mentioned | 55 % |
| `has_anc` | Active Noise Cancellation | 5 % |
| `has_enc` | Environmental Noise Cancellation (calls) | 13 % |
| `has_mic` | Built-in microphone | 98 % |
| `fast_charge` | Fast / ASAP / Type-C charging | 35 % |
| `low_latency` | Gaming / low-latency / beast mode | 15 % |
| `voice_assistant` | Google Assistant / Siri support | 38 % |
| `bt_version` | Bluetooth version (e.g. 5.0) | 21 of 60 known |
| `has_water_res` / `ipx` | Water-resistance flag / IPX level | 23 of 60 have a rating (IPX4–IPX7) |
| `driver_mm` | Driver size (mm) | 43 of 60 known (median 10) |
| `playtime_hrs` | Battery playtime (hours) | 30 of 60 known (median ~20) |
| `n_bullets` | Number of spec bullets in listing | 1 – 8 |
| `name_len` | Title length (listing-richness proxy) | 52 – 199 chars |

Missing spec values are left as `NaN` (XGBoost handles missing values natively).

**D. Brand / competition features**

| Feature | Description |
|---|---|
| `brand` | First word of product name (boAt 21, Boult 11, JBL 5, ZEBRONICS 5, PTron 3, Samsung 3, Noise 3 …) |
| `sub_category` | In-Ear (51) · On-Ear (7) · Over-Ear (2) |

**E. Sentiment features (VADER on `review_content`)**

| Feature | Description | Value |
|---|---|---|
| `n_reviews_sampled` | Number of review snippets per product | 8 – 75 |
| `sentiment_mean` | Mean VADER compound score (−1 to +1) | mean 0.29 |
| `pct_positive` | Share of snippets with score > 0.05 | mean 0.61 |

### 6.3 Market-level signals (computed across the 60 products)

| Signal | Result |
|---|---|
| `competitor_count` | 60 products (18 are true-wireless) |
| Price band (P25–P75, all audio) | **₹499 – ₹1,399** (median ₹894) |
| Price band (P25–P75, TWS only) | **₹1,224 – ₹1,599** (median ₹1,449) |
| Top-brand dominance (share of ratings) | boAt 63 %, JBL 14 %, Boult 6 % → **top-3 ≈ 84 %** |
| TWS feature adoption | ENC 33 % · fast charge 61 % · low latency 44 % · ANC only 6 % |
| Median rating by form factor | Wired 4.10 · Other 4.00 · Neckband 3.85 · TWS 3.80 |

> **Reading:** the market is highly concentrated around one brand, and TWS earbuds in this data are priced about 1.6× the all-audio median with the lowest median rating. ANC is rare, which could be a differentiation gap.

### 6.4 Mapping to the deck's data schema (Slide 5)

| Slide 5 field | Available in `consumer_electronics_market_20k.csv`? | Feature(s) |
|---|---|---|
| `price`, `discount_pct` | ✅ | `discounted_price`, `discount_pct`, `price_vs_median` |
| `rating`, `review_count` | ✅ | `rating`, `rating_count`, `log_rating_count` |
| `competitor_count` | ✅ (market-level) | 60 / 18 TWS |
| `features` (ANC, 30h, IPX4) | ✅ | `has_anc`, `playtime_hrs`, `ipx`, … |
| `review_growth` (demand proxy / **target**) | ❌ | Not derivable — see limitation below |
| `date` (time series) | ❌ | Not in the file |

### 6.5 Limitations (state these honestly in the viva)

- **Single snapshot.** `consumer_electronics_market_20k.csv` has no date column and `rating_count` is cumulative, so **`review_growth` (our target) cannot be computed** from it. Use this dataset for the **feature pipeline, price band, competition and sentiment**; use the **daily simulator** (anchored to these real distributions) to generate the time series needed to train and test the demand-growth model and drift detection.
- **Multi-channel scale:** Scaled from the initial 60-product prototype to **20,237 total records** (including **9,409 audio/earbuds/headphones**) covering Amazon (online), Flipkart (online), Croma (offline retail), and offline dealer markets.
- **Mixed form factors:** Includes true TWS earbuds, neckbands, on/over-ear, and wired headsets across price tiers. Filter on `form_factor == "TWS"` when TWS-only segment analysis is desired.
- **Regex extraction** is approximate: some specs (e.g. `bt_version`, `playtime_hrs`) are missing when not stated in the listing. Sanity-check outliers (e.g. `driver_mm = 2`, `playtime_hrs = 100`).
- **VADER on short snippets** is a rough sentiment signal; reviews are pre-aggregated per product and may not be representative.
- **Review text is comma-joined**, so splitting on commas can occasionally cut a sentence in two.

---

## 7. Demo Plan

1. Ingest a new market snapshot into S3.
2. Query the API from the dashboard (e.g. budget TWS at ₹2,499).
3. Inject a controlled market shift (e.g. price war) via the simulator.
4. Drift detected in CloudWatch → challenger trained automatically.
5. Quality gate promotes or rejects the challenger.

Example dashboard output *(illustrative values, not real results)*:

```
Budget TWS Earbuds · Target price ₹2,499
Predicted demand growth : +19%
Competitive price band  : ₹2,299 – ₹2,799
Customer sentiment      : 82% positive
Competition             : High
Overall                 : MEDIUM-HIGH POTENTIAL
Opportunity: demand growth + strong sentiment
Risk       : high competition, price pressure
```

---

## 8. Team & Responsibilities

| Role | Owns | Be ready to explain |
|---|---|---|
| Data & Ingestion | Lambda collector, S3 layout, validation | Data sources, proxies, validation |
| ML & Pipeline | Model, SageMaker Pipeline, MLflow, registry | XGBoost, time-based split, MAE/RMSE, quality gate |
| API & Dashboard | Endpoint, API Gateway, UI | Request flow dashboard → model → back |
| CI/CD & Monitoring | GitHub Actions, ECR, CloudWatch, drift | Tests, drift measurement, retraining triggers |

---

## 9. Limitations & Future Scope

**Limitations:** demand is a proxy (not real sales); some test data is simulated; market success also depends on marketing, brand and supply chain — this is **decision support, not a guarantee**.

**Future scope:** other categories (smartwatches, smartphones) · licensed real-time sales data · A/B testing and shadow deployment of challengers · SHAP explainability.

---

## 10. Glossary

| Term | Meaning |
|---|---|
| MLOps | Practices to build, deploy, monitor and update ML models reliably |
| Pipeline | Automated, ordered steps defined in code |
| XGBoost | Gradient-boosted trees; strong for tabular data |
| MAE / RMSE | Average error / error penalising large mistakes (lower is better) |
| Time-based split | Train on past, test on future |
| Demand proxy | Indirect measure when real sales are unavailable (review growth) |
| Champion / Challenger | Deployed model / candidate that must beat it |
| Quality gate | Pipeline rule blocking models that miss required metrics |
| Model Registry | Versioned catalogue of approved models |
| Data drift | Incoming data distribution changes vs. training data |
| KS test / PSI | Statistical measures of distribution shift |
| Endpoint | Hosted model serving predictions over HTTPS |
| CI/CD | Automatic testing and release of code changes |
| OIDC | Temporary AWS credentials for GitHub Actions, no stored keys |
