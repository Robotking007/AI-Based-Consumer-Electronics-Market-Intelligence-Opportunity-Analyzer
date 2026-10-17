"""
preprocessing.py - Data Validation & Feature Engineering for SageMaker Processing Job
Implements Section 6.2 of PROJECT.md:
- Data cleaning (stripping currency, %, commas)
- Price features (price_gap, price_vs_median, price_tier)
- Customer response features (log_rating_count, review_rating_score)
- Product spec extraction via regex (ANC, ENC, Mic, Battery, IPX, Driver size, BT version)
- Sentiment analysis (NLTK VADER compound score)
- Demand growth proxy target simulation for 30-day window
- Train/Validation/Test split formatted for SageMaker XGBoost
"""

import os
import re
import json
import argparse
import numpy as np
import pandas as pd

try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon', quiet=True)
    VADER_AVAILABLE = True
except Exception:
    VADER_AVAILABLE = False


def clean_currency(val):
    if pd.isna(val) or val is None:
        return 0.0
    s = re.sub(r'[^\d.]', '', str(val)).strip()
    try:
        return float(s)
    except:
        return 0.0


def extract_specs(row):
    """Extract hardware and feature specs from product name and about_product bullets."""
    text = (str(row.get('product_name', '')) + " " + str(row.get('about_product', ''))).lower()

    is_wireless = 1 if any(k in text for k in ['wireless', 'bluetooth', 'tws', 'airpods', 'earbuds', 'neckband']) else 0
    has_anc = 1 if any(k in text for k in ['anc', 'active noise cancellation', 'active noise cancelling']) else 0
    has_enc = 1 if any(k in text for k in ['enc', 'environmental noise cancellation', 'ai enc', 'quad mic', 'quad-mic']) else 0
    has_mic = 1 if any(k in text for k in ['mic', 'microphone', 'calling', 'voice call']) else 0
    fast_charge = 1 if any(k in text for k in ['fast charge', 'asap charge', 'quick charge', 'type-c', 'warp charge']) else 0
    low_latency = 1 if any(k in text for k in ['low latency', 'beast mode', 'gaming mode', 'latency', 'game mode']) else 0
    voice_assistant = 1 if any(k in text for k in ['siri', 'google assistant', 'alexa', 'voice assistant']) else 0

    # BT version
    m_bt = re.search(r'bluetooth\s*(?:v|version)?\s*([45]\.\d)', text)
    bt_ver = float(m_bt.group(1)) if m_bt else 5.0

    # IPX rating
    m_ipx = re.search(r'ipx\s*([0-8])', text)
    ipx_rating = int(m_ipx.group(1)) if m_ipx else (4 if is_wireless else 0)

    # Driver size in mm
    m_driver = re.search(r'(\d{1,2}(?:\.\d)?)\s*mm\s*(?:driver|dynamic)', text)
    driver_mm = float(m_driver.group(1)) if m_driver else 10.0

    # Playtime hours
    m_play = re.search(r'(\d{1,3})\s*(?:hours|hrs|hr)\s*(?:play|battery|backup)', text)
    playtime_hrs = float(m_play.group(1)) if m_play else (24.0 if is_wireless else 0.0)

    # Bullet counts & title length
    about = str(row.get('about_product', ''))
    n_bullets = len([b for b in about.split('|') if len(b.strip()) > 3])
    name_len = len(str(row.get('product_name', '')))

    return pd.Series({
        'is_wireless': is_wireless,
        'has_anc': has_anc,
        'has_enc': has_enc,
        'has_mic': has_mic,
        'fast_charge': fast_charge,
        'low_latency': low_latency,
        'voice_assistant': voice_assistant,
        'bt_version': bt_ver,
        'ipx_rating': ipx_rating,
        'driver_mm': driver_mm,
        'playtime_hrs': playtime_hrs,
        'n_bullets': n_bullets,
        'name_len': name_len
    })


def compute_sentiment(text, analyzer):
    if not text or pd.isna(text):
        return 0.25, 0.60
    if not analyzer:
        # Fallback heuristic
        t = str(text).lower()
        pos = sum(t.count(w) for w in ['great', 'good', 'best', 'excellent', 'love', 'nice', 'clear', 'impressive'])
        neg = sum(t.count(w) for w in ['bad', 'poor', 'worst', 'issue', 'broken', 'latency', 'heating', 'fail'])
        total = pos + neg + 1
        compound = (pos - neg) / total
        return round(compound, 3), round(pos / total, 3)

    try:
        scores = analyzer.polarity_scores(str(text)[:500])
        compound = scores['compound']
        pct_pos = scores['pos'] / (scores['pos'] + scores['neg'] + 1e-5)
        return round(compound, 3), round(min(1.0, pct_pos), 3)
    except:
        return 0.25, 0.60


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run full feature engineering pipeline matching PROJECT.md."""
    print("Executing feature engineering pipeline...")
    data = df.copy()

    # 1. Clean price & numerical columns
    data['discounted_price_num'] = data['discounted_price'].apply(clean_currency)
    data['actual_price_num'] = data['actual_price'].apply(clean_currency)
    data['discount_pct_num'] = data['discount_percentage'].apply(clean_currency)
    data['rating_num'] = pd.to_numeric(data['rating'], errors='coerce').fillna(4.0)
    data['rating_count_num'] = data['rating_count'].astype(str).str.replace(',', '').str.replace(' ', '')
    data['rating_count_num'] = pd.to_numeric(data['rating_count_num'], errors='coerce').fillna(500.0)

    # 2. Engineered Price Features (Section 6.2 A)
    data['price_gap'] = data['actual_price_num'] - data['discounted_price_num']
    market_median_price = data['discounted_price_num'].median() or 1399.0
    data['price_vs_median'] = (data['discounted_price_num'] / market_median_price).round(3)

    # Price tiers: 0=Budget (<500), 1=Mid (500-1k), 2=Upper (1k-2.5k), 3=Premium (>2.5k)
    data['price_tier'] = pd.cut(
        data['discounted_price_num'],
        bins=[-np.inf, 500, 1000, 2500, np.inf],
        labels=[0, 1, 2, 3]
    ).astype(int)

    # 3. Customer Response Features (Section 6.2 B)
    data['log_rating_count'] = np.log1p(data['rating_count_num']).round(3)
    data['review_rating_score'] = (data['rating_num'] * data['log_rating_count']).round(3)
    total_ratings = data['rating_count_num'].sum()
    data['rating_count_share'] = (data['rating_count_num'] / total_ratings * 100.0).round(4)

    # 4. Product Spec Extraction (Section 6.2 C)
    print("Extracting hardware & connectivity specs via regex...")
    spec_df = data.apply(extract_specs, axis=1)
    data = pd.concat([data, spec_df], axis=1)

    # 5. Sentiment Analysis (Section 6.2 E)
    print("Calculating review sentiment signals...")
    analyzer = SentimentIntensityAnalyzer() if VADER_AVAILABLE else None
    sent_results = [compute_sentiment(rev, analyzer) for rev in data['review_content']]
    data['sentiment_compound'] = [s[0] for s in sent_results]
    data['pct_positive_reviews'] = [s[1] for s in sent_results]

    # 6. Market Channel Encoding
    channel_map = {
        'Amazon (Online)': 0,
        'Croma (Offline Retail Chain)': 1,
        'Flipkart (Online)': 2,
        'Offline Electronics Market (Dealer MOP)': 3
    }
    data['channel_code'] = data['market_channel'].map(channel_map).fillna(0).astype(int)

    # 7. Form Factor Encoding
    def get_form_factor(row):
        t = (str(row.get('product_name', '')) + " " + str(row.get('category', ''))).lower()
        if 'tws' in t or 'true wireless' in t or 'airpod' in t or 'earbuds' in t:
            return 0  # TWS
        elif 'neckband' in t:
            return 1  # Neckband
        elif 'wired' in t or '3.5mm' in t or 'aux' in t:
            return 2  # Wired
        else:
            return 3  # Other

    data['form_factor_code'] = data.apply(get_form_factor, axis=1)

    # 8. Demand Growth Proxy Target (Section 3 & 6.4 of PROJECT.md)
    # Realistic 30-day review growth (%) based on:
    # - Discount attractiveness (higher discount = higher demand growth)
    # - Price vs median (overpriced items grow slower)
    # - Sentiment momentum (positive sentiment = word-of-mouth growth)
    # - Feature richness (ANC, battery, fast charge)
    print("Simulating target variable: 30-day review count growth (% demand proxy)...")
    np.random.seed(42)
    base_growth = 12.0
    discount_boost = (data['discount_pct_num'] / 100.0) * 10.0
    price_penalty = np.clip((data['price_vs_median'] - 1.0) * 4.0, -5.0, 8.0)
    sentiment_boost = data['sentiment_compound'] * 8.0
    spec_boost = (data['has_anc'] * 3.5 + data['fast_charge'] * 2.0 + data['has_enc'] * 1.5)
    noise = np.random.normal(0, 2.5, size=len(data))

    target = base_growth + discount_boost - price_penalty + sentiment_boost + spec_boost + noise
    data['target_demand_growth_pct'] = np.clip(np.round(target, 2), 1.0, 65.0)

    return data


def run_preprocessing(input_csv: str, output_dir: str):
    """Full preprocessing pipeline entrypoint."""
    os.makedirs(output_dir, exist_ok=True)
    df_raw = pd.read_csv(input_csv)
    print(f"Loaded raw dataset with {len(df_raw):,} records.")

    featured_df = engineer_features(df_raw)

    # Model feature list
    feature_cols = [
        'discounted_price_num', 'actual_price_num', 'discount_pct_num',
        'price_gap', 'price_vs_median', 'price_tier',
        'rating_num', 'log_rating_count', 'review_rating_score', 'rating_count_share',
        'is_wireless', 'has_anc', 'has_enc', 'has_mic', 'fast_charge',
        'low_latency', 'voice_assistant', 'bt_version', 'ipx_rating',
        'driver_mm', 'playtime_hrs', 'n_bullets', 'name_len',
        'sentiment_compound', 'pct_positive_reviews',
        'channel_code', 'form_factor_code'
    ]
    target_col = 'target_demand_growth_pct'

    # Filter clean dataset
    clean_df = featured_df[[target_col] + feature_cols].dropna().reset_index(drop=True)

    # Split: 70% Train, 15% Validation, 15% Test
    n = len(clean_df)
    train_end = int(0.70 * n)
    val_end = int(0.85 * n)

    # Time-based order / random shuffle
    clean_df = clean_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    train_df = clean_df.iloc[:train_end]
    val_df = clean_df.iloc[train_end:val_end]
    test_df = clean_df.iloc[val_end:]

    # Save SageMaker XGBoost ready CSVs (target in first column, no header for SM built-in algorithm)
    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "validation.csv")
    test_path = os.path.join(output_dir, "test.csv")
    meta_path = os.path.join(output_dir, "feature_metadata.json")

    train_df.to_csv(train_path, index=False, header=False)
    val_df.to_csv(val_path, index=False, header=False)
    test_df.to_csv(test_path, index=False, header=True)  # Include header for evaluation readability

    # Baseline distribution statistics for drift monitoring (KS test / PSI)
    baseline_stats = {
        "record_counts": {
            "total": len(clean_df),
            "train": len(train_df),
            "validation": len(val_df),
            "test": len(test_df)
        },
        "target_stats": {
            "mean": float(clean_df[target_col].mean()),
            "std": float(clean_df[target_col].std()),
            "median": float(clean_df[target_col].median())
        },
        "feature_names": feature_cols,
        "feature_stats": {
            col: {
                "mean": float(clean_df[col].mean()),
                "std": float(clean_df[col].std()),
                "min": float(clean_df[col].min()),
                "max": float(clean_df[col].max())
            } for col in feature_cols
        }
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(baseline_stats, f, indent=2)

    print(f"\nPreprocessing completed successfully:")
    print(f" - Train set      : {len(train_df):,} rows -> {train_path}")
    print(f" - Validation set : {len(val_df):,} rows -> {val_path}")
    print(f" - Test set       : {len(test_df):,} rows -> {test_path}")
    print(f" - Feature metadata: {meta_path}")

    return train_path, val_path, test_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SageMaker Preprocessing & Feature Engineering Step")
    parser.add_argument("--input-data", default="consumer_electronics_market_20k.csv", help="Input CSV path")
    parser.add_argument("--output-dir", default="processed_data", help="Output directory")
    args = parser.parse_args()

    run_preprocessing(args.input_data, args.output_dir)
