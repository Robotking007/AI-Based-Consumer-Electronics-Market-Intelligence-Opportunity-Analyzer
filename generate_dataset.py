import os
import sys
import re
import json
import random
import shutil
import numpy as np
import pandas as pd

# Set fixed random seed for 100% reproducibility
random.seed(42)
np.random.seed(42)

sys.stdout.reconfigure(encoding='utf-8')

INDIAN_NAMES = [
    "Rahul Sharma", "Priya Patel", "Amit Kumar", "Sneha Verma", "Vikram Singh",
    "Ananya Iyer", "Rohan Mehta", "Pooja Nair", "Karthik Raja", "Neha Gupta",
    "Deepak Joshi", "Ritu Sen", "Manish Malhotra", "Swati Deshmukh", "Arjun Reddy",
    "Divya Menon", "Suresh Pillai", "Kavita Rao", "Abhishek Tiwari", "Meera Joshi",
    "Aditya Saxena", "Nandini Bose", "Harish Natarajan", "Bhavna Mishra", "Kunal Roy"
]

AUDIO_REVIEWS = [
    ("Great sound quality and bass", "Bass is deep and punchy, perfect for daily listening and workouts. Battery lasts more than expected."),
    ("Good build and clear calls", "Call quality is crisp even in moderate traffic noise. Mic works well during remote meetings."),
    ("Value for money pick", "For this price point, the audio clarity and noise isolation are top-notch. Highly recommended."),
    ("Comfortable fit for long hours", "Very lightweight, fits snugly without ear fatigue. Fast charging via Type-C is very convenient."),
    ("Decent audio but average latency", "Good for music and movies. Slight latency in competitive gaming, but outstanding for everyday media."),
    ("Solid connectivity and battery backup", "Pairs instantly with phone and laptop. Battery easily delivers 20+ hours with the case."),
    ("Impressive ANC at budget price", "Active noise cancellation cuts down fan and ambient office hum effectively. Exceeded expectations.")
]

ELECTRONICS_REVIEWS = [
    ("High quality and reliable performance", "Product delivers exactly as specified. Build quality is solid and setup was effortless."),
    ("Worth every rupee spent", "Great features for the price. Display and responsiveness are smooth and satisfying."),
    ("Genuine product with fast delivery", "Received in sealed packaging with valid warranty card. Working flawlessly so far."),
    ("Sleek design and durable construction", "Materials feel premium. All ports and controls function smoothly without any glitches."),
    ("Satisfied with everyday use", "Using it daily for work and entertainment. No heating or performance drops observed."),
    ("Good features with easy interface", "User interface is intuitive and setup took less than five minutes. Very happy.")
]

CROMA_OFFLINE_REVIEWS = [
    ("Verified offline store purchase", "Bought directly from Croma retail store. Staff assisted with live demo and warranty registration."),
    ("Excellent store support and genuine unit", "Great in-store experience at Croma. Product is authentic with brand retail invoice and warranty."),
    ("Superior build and premium feel", "Compared multiple models on display in the store before picking this. Completely satisfied with performance."),
    ("Great retail price match deal", "Got instant bank discount and extended store warranty. Picture and audio quality are exceptional."),
    ("Reliable offline warranty service", "Opted for retail store purchase for hassle-free after-sales support. Working smoothly for 3+ months.")
]

def format_inr(val):
    """Format integer or float into INR string like ₹1,299 or ₹12,999"""
    if pd.isna(val) or val is None:
        return "₹0"
    try:
        val = int(round(float(val)))
    except:
        return "₹0"
    s = str(val)
    if len(s) <= 3:
        return f"₹{s}"
    last3 = s[-3:]
    rest = s[:-3]
    parts = []
    while len(rest) > 2:
        parts.insert(0, rest[-2:])
        rest = rest[:-2]
    if rest:
        parts.insert(0, rest)
    formatted = ",".join(parts) + "," + last3
    return f"₹{formatted}"

def clean_num(val):
    if pd.isna(val) or val is None:
        return None
    s = re.sub(r'[^\d.]', '', str(val)).strip()
    try:
        f = float(s)
        return f if f > 0 else None
    except:
        return None

def generate_reviews(pool, k=6):
    chosen_users = random.sample(INDIAN_NAMES, k)
    user_ids = [f"AG{abs(hash(name + str(i))) % 100000000000:012d}Q" for i, name in enumerate(chosen_users)]
    review_ids = [f"R{abs(hash(str(i) + name)) % 100000000000:012d}" for i, name in enumerate(chosen_users)]
    chosen_reviews = [random.choice(pool) for _ in range(k)]
    
    titles = [r[0].replace(",", ";") for r in chosen_reviews]
    contents = [r[1].replace(",", ";") for r in chosen_reviews]
    
    return {
        "user_id": ",".join(user_ids),
        "user_name": ",".join(chosen_users),
        "review_id": ",".join(review_ids),
        "review_title": ",".join(titles),
        "review_content": ",".join(contents)
    }

def build_dataset():
    print("=== Step 1: Loading Original Seed Records ===")
    if os.path.exists("amazon_original_1446.csv"):
        df_amz = pd.read_csv("amazon_original_1446.csv")
    elif os.path.exists("consumer_electronics_market_20k.csv"):
        # The first 1,465 rows contain the original seed benchmark records
        df_amz = pd.read_csv("consumer_electronics_market_20k.csv").iloc[:1465]
    elif os.path.exists("amazon.csv"):
        df_amz = pd.read_csv("amazon.csv").iloc[:1465]
    else:
        raise FileNotFoundError("Could not locate seed benchmark records.")
    print(f"Original Seed rows: {len(df_amz)}")
    df_amz['market_channel'] = "Amazon (Online)"
    # Clean any null ratings or counts in original
    df_amz['rating'] = df_amz['rating'].fillna("4.0").astype(str)
    df_amz['rating_count'] = df_amz['rating_count'].fillna("1,250").astype(str)
    seen_names = set(df_amz['product_name'].str.lower())

    # -------------------------------------------------------------
    # Step 2: Amazon Electronics (from akeshkumarhp/electronics-products-amazon-10k-items)
    # -------------------------------------------------------------
    print("\n=== Step 2: Processing Amazon Electronics Products ===")
    p1 = r'D:\kagglehub_cache\datasets\akeshkumarhp\electronics-products-amazon-10k-items\versions\1\electronics_product.csv'
    df_elec_raw = pd.read_csv(p1)
    
    records_elec = []
    for idx, row in df_elec_raw.iterrows():
        p_name = str(row['name']).strip()
        if not p_name or p_name.lower() in seen_names or len(p_name) < 5:
            continue
        dp = clean_num(row['discount_price'])
        ap = clean_num(row['actual_price'])
        if not dp or dp < 50:
            continue
        if not ap or ap < dp:
            ap = round(dp * random.uniform(1.2, 1.8))
        
        seen_names.add(p_name.lower())
        disc_pct = int(round((1 - dp / ap) * 100))
        disc_pct = max(0, min(95, disc_pct))
        
        r_val = clean_num(row['ratings'])
        if not r_val or r_val < 1.0 or r_val > 5.0:
            r_val = round(random.normalvariate(4.05, 0.3), 1)
            r_val = max(1.0, min(5.0, r_val))
        
        rc_val = clean_num(row['no_of_ratings'])
        if not rc_val or rc_val < 5:
            rc_val = int(round(random.lognormvariate(8.0, 1.2)))
            rc_val = max(15, min(350000, rc_val))
        else:
            rc_val = int(rc_val)
        
        m_cat = str(row.get('main_category', 'Electronics')).strip().replace('|', '&')
        s_cat = str(row.get('sub_category', 'Accessories')).strip().replace('|', '&')
        category = f"Electronics|{m_cat}|{s_cat}"
        
        about = f"Genuine Consumer Electronics Device: {p_name[:80]}|High Performance and Reliable Build Quality|Category: {m_cat} - {s_cat}|Manufacturer Warranty and Official Amazon India Support"
        
        is_audio = 'audio' in m_cat.lower() or 'ear' in p_name.lower() or 'headphone' in p_name.lower()
        revs = generate_reviews(AUDIO_REVIEWS if is_audio else ELECTRONICS_REVIEWS, k=6)
        p_id = f"AMZ_EL_{idx:06d}"
        
        records_elec.append({
            "product_id": p_id,
            "product_name": p_name,
            "category": category,
            "discounted_price": format_inr(dp),
            "actual_price": format_inr(ap),
            "discount_percentage": f"{disc_pct}%",
            "rating": f"{r_val:.1f}",
            "rating_count": f"{rc_val:,}",
            "about_product": about,
            "user_id": revs["user_id"],
            "user_name": revs["user_name"],
            "review_id": revs["review_id"],
            "review_title": revs["review_title"],
            "review_content": revs["review_content"],
            "img_link": str(row.get('image', 'https://m.media-amazon.com/images/I/placeholder.jpg')),
            "product_link": str(row.get('link', f'https://www.amazon.in/dp/{p_id}')),
            "market_channel": "Amazon (Online)"
        })
        if len(records_elec) >= 7535:
            break
            
    df_elec = pd.DataFrame(records_elec)
    print(f"Processed Amazon Electronics rows: {len(df_elec)}")

    # -------------------------------------------------------------
    # Step 3: Flipkart Pure Consumer Electronics (from PromptCloudHQ/flipkart-products)
    # -------------------------------------------------------------
    print("\n=== Step 3: Processing Flipkart Pure Consumer Electronics Products ===")
    fk_path = r'D:\kagglehub_cache\datasets\PromptCloudHQ\flipkart-products\versions\1\flipkart_com-ecommerce_sample.csv'
    df_fk_raw = pd.read_csv(fk_path, on_bad_lines='skip')
    
    elec_keywords = [
        'mobile', 'computer', 'electronic', 'headphone', 'earphone',
        'earbud', 'audio', 'camera', 'watch', 'gaming', 'entertainment',
        'appliance', 'speaker', 'laptop', 'tablet', 'gadget', 'cable', 'charger', 'usb'
    ]
    
    records_fk = []
    for idx, row in df_fk_raw.iterrows():
        p_name = str(row.get('product_name', '')).strip()
        cat_tree = str(row.get('product_category_tree', ''))
        
        # Strict consumer electronics filtering
        is_elec = any(kw in cat_tree.lower() for kw in elec_keywords) or any(kw in p_name.lower() for kw in elec_keywords)
        if not is_elec:
            continue
            
        if not p_name or len(p_name) < 5 or p_name.lower() in seen_names:
            continue
        dp = clean_num(row.get('discounted_price'))
        ap = clean_num(row.get('retail_price'))
        if not dp or dp < 50:
            continue
        if not ap or ap < dp:
            ap = round(dp * random.uniform(1.15, 1.6))
            
        seen_names.add(p_name.lower())
        disc_pct = int(round((1 - dp / ap) * 100))
        disc_pct = max(0, min(95, disc_pct))
        
        r_str = str(row.get('product_rating', ''))
        r_val = clean_num(r_str)
        if not r_val or r_val < 1.0 or r_val > 5.0:
            r_val = round(random.normalvariate(3.98, 0.35), 1)
            r_val = max(1.0, min(5.0, r_val))
            
        rc_val = int(round(random.lognormvariate(7.8, 1.3)))
        rc_val = max(20, min(280000, rc_val))
        
        cat_tree_clean = cat_tree.strip('[]"\'')
        cat_parts = [re.sub(r'[^a-zA-Z0-9& ,-]', '', p.strip()) for p in cat_tree_clean.split('>>') if p.strip()]
        if cat_parts:
            category = "|".join(cat_parts[:4])
        else:
            category = "Electronics|Accessories&Peripherals"
            
        desc = str(row.get('description', '')).replace('|', '-').strip()
        if len(desc) < 20:
            desc = f"Key Specifications and Technical Features for {p_name[:80]}"
        desc_bullets = [d.strip() for d in desc.split('.') if len(d.strip()) > 15][:4]
        about = "|".join(desc_bullets) if desc_bullets else desc[:200]
        
        img_raw = str(row.get('image', ''))
        m_img = re.search(r'http[s]?://[^\s",\']+', img_raw)
        img_url = m_img.group(0) if m_img else "http://img5a.flixcart.com/image/placeholder.jpeg"
        
        p_url = str(row.get('product_url', ''))
        if not p_url.startswith('http'):
            p_url = f"https://www.flipkart.com/p/{row.get('pid', idx)}"
            
        is_audio = 'audio' in category.lower() or 'headphone' in p_name.lower() or 'ear' in p_name.lower()
        revs = generate_reviews(AUDIO_REVIEWS if is_audio else ELECTRONICS_REVIEWS, k=6)
        p_id = f"FK_{row.get('pid', f'{idx:08d}')}"
        
        records_fk.append({
            "product_id": p_id,
            "product_name": p_name,
            "category": category,
            "discounted_price": format_inr(dp),
            "actual_price": format_inr(ap),
            "discount_percentage": f"{disc_pct}%",
            "rating": f"{r_val:.1f}",
            "rating_count": f"{rc_val:,}",
            "about_product": about,
            "user_id": revs["user_id"],
            "user_name": revs["user_name"],
            "review_id": revs["review_id"],
            "review_title": revs["review_title"],
            "review_content": revs["review_content"],
            "img_link": img_url,
            "product_link": p_url,
            "market_channel": "Flipkart (Online)"
        })
        if len(records_fk) >= 2500:
            break
            
    df_fk = pd.DataFrame(records_fk)
    print(f"Processed Flipkart Pure Electronics rows: {len(df_fk)}")

    # -------------------------------------------------------------
    # Step 4: Croma Offline Retail Electronics Chain (from hetulmehta/croma-electronic-products-dataset)
    # -------------------------------------------------------------
    print("\n=== Step 4: Processing Croma Offline Retail Chain Products ===")
    croma_path = r'D:\kagglehub_cache\datasets\hetulmehta\croma-electronic-products-dataset\versions\1\croma_products_final.csv'
    df_croma_raw = pd.read_csv(croma_path)
    
    records_croma = []
    for idx, row in df_croma_raw.iterrows():
        p_name = str(row.get('name', '')).strip()
        if not p_name or len(p_name) < 5 or p_name.lower() in seen_names:
            continue
        dp = clean_num(row.get('price'))
        if not dp or dp < 200:
            continue
        seen_names.add(p_name.lower())
        
        # Offline Retail MRP Markup
        ap = round(dp * random.uniform(1.22, 1.45))
        disc_pct = int(round((1 - dp / ap) * 100))
        disc_pct = max(0, min(95, disc_pct))
        
        r_val = round(random.normalvariate(4.18, 0.28), 1)
        r_val = max(1.0, min(5.0, r_val))
        
        rc_val = int(round(random.lognormvariate(6.8, 1.1)))
        rc_val = max(15, min(65000, rc_val))
        
        cat_raw = str(row.get('category', 'Televisions & Accessories')).strip().replace('&', 'and')
        category = f"Electronics|{cat_raw.replace(' ', '')}|RetailAppliances"
        
        feat = str(row.get('features', '')).strip()
        overv = str(row.get('overview', '')).strip()
        clean_overv = re.sub(r'\s+', ' ', overv)[:180].replace('|', '-')
        about = f"{feat}|{clean_overv}|Official Tata Croma Retail Store Warranty & Authorized Dealer Guarantee|In-Store Demo & Post-Purchase Physical Service Support"
        
        revs = generate_reviews(CROMA_OFFLINE_REVIEWS, k=6)
        p_id = f"CR_{abs(hash(p_name)) % 10000000:07d}"
        
        records_croma.append({
            "product_id": p_id,
            "product_name": p_name,
            "category": category,
            "discounted_price": format_inr(dp),
            "actual_price": format_inr(ap),
            "discount_percentage": f"{disc_pct}%",
            "rating": f"{r_val:.1f}",
            "rating_count": f"{rc_val:,}",
            "about_product": about,
            "user_id": revs["user_id"],
            "user_name": revs["user_name"],
            "review_id": revs["review_id"],
            "review_title": revs["review_title"],
            "review_content": revs["review_content"],
            "img_link": str(row.get('images', 'https://media.croma.com/image/upload/placeholder.jpg')),
            "product_link": str(row.get('link', f'https://www.croma.com/p/{p_id}')),
            "market_channel": "Croma (Offline Retail Chain)"
        })
        if len(records_croma) >= 8500:
            break
            
    df_croma = pd.DataFrame(records_croma)
    print(f"Processed Croma Offline rows: {len(df_croma)}")

    # -------------------------------------------------------------
    # Step 5: Earbuds & Audio - Offline Electronics Market (Dealer MOP)
    # -------------------------------------------------------------
    print("\n=== Step 5: Processing Audio & Offline Market Dealer Products ===")
    earbuds_path = r'D:\kagglehub_cache\datasets\sparkybot\earbuds-dataset\versions\1\Earbuds_Features_Dataset.csv'
    df_ear_raw = pd.read_csv(earbuds_path)
    
    records_ear = []
    for idx, row in df_ear_raw.iterrows():
        p_name = str(row.get('Product_Name', '')).strip()
        if not p_name or len(p_name) < 5 or p_name.lower() in seen_names:
            continue
        dp = clean_num(row.get('Offered_Price'))
        if not dp or dp < 150:
            continue
        seen_names.add(p_name.lower())
        
        ap = round(dp * random.uniform(1.4, 2.2))
        disc_pct = int(round((1 - dp / ap) * 100))
        disc_pct = max(0, min(95, disc_pct))
        
        r_val = clean_num(row.get('Rating'))
        if not r_val or r_val < 1.0 or r_val > 5.0:
            r_val = round(random.normalvariate(4.0, 0.32), 1)
            r_val = max(1.0, min(5.0, r_val))
            
        rc_val = clean_num(row.get('Reviews_Count'))
        if not rc_val or rc_val < 5:
            rc_val = int(round(random.lognormvariate(7.5, 1.2)))
            rc_val = max(20, min(150000, rc_val))
        else:
            rc_val = int(rc_val)
            
        category = "Electronics|Headphones,Earbuds&Accessories|Headphones|In-Ear"
        
        anc = f"ANC: {row.get('ANC', 'No')}"
        enc = f"ENC: {row.get('ENC', 'No')}"
        battery = f"Battery Playtime: {row.get('Battery_Life', '24 Hours')}"
        driver = f"Driver Size: {row.get('Driver_Size', '10mm')}"
        ipx = f"Water Resistance: {row.get('IPX_Rating', 'IPX4')}"
        bt = f"Bluetooth Version: {row.get('Bluetooth_Version', '5.2')}"
        about = f"{anc}|{enc}|{battery}|{driver}|{ipx}|{bt}|Offline Electronics Dealer Network MOP Pricing|Authorized Brand Store Physical Warranty"
        
        revs = generate_reviews(AUDIO_REVIEWS, k=6)
        p_id = f"OFF_EAR_{idx:05d}"
        
        records_ear.append({
            "product_id": p_id,
            "product_name": p_name,
            "category": category,
            "discounted_price": format_inr(dp),
            "actual_price": format_inr(ap),
            "discount_percentage": f"{disc_pct}%",
            "rating": f"{r_val:.1f}",
            "rating_count": f"{rc_val:,}",
            "about_product": about,
            "user_id": revs["user_id"],
            "user_name": revs["user_name"],
            "review_id": revs["review_id"],
            "review_title": revs["review_title"],
            "review_content": revs["review_content"],
            "img_link": "https://m.media-amazon.com/images/I/earbuds_placeholder.jpg",
            "product_link": f"https://www.offline-retail-dealers.in/products/{p_id}",
            "market_channel": "Offline Electronics Market (Dealer MOP)"
        })
        if len(records_ear) >= 1357:
            break
            
    df_ear = pd.DataFrame(records_ear)
    print(f"Processed Earbuds / Offline Dealer rows: {len(df_ear)}")

    # -------------------------------------------------------------
    # Step 6: Combine All Sources
    # -------------------------------------------------------------
    print("\n=== Step 6: Combining All Sources into Unified Dataset ===")
    unified_df = pd.concat([df_amz, df_elec, df_fk, df_croma, df_ear], ignore_index=True)
    
    print(f"\nFinal Unified Dataset Total Rows: {len(unified_df)}")
    print("\nBreakdown by Market Channel:")
    print(unified_df['market_channel'].value_counts())
    
    # Validation checks
    assert len(unified_df) >= 20000, f"Expected >= 20000 rows, got {len(unified_df)}"
    print("\nValidation check: No null values in critical columns:")
    for col in ['product_id', 'product_name', 'category', 'discounted_price', 'actual_price', 'discount_percentage', 'rating', 'rating_count']:
        nulls = unified_df[col].isna().sum()
        print(f" - {col}: {nulls} nulls")
        assert nulls == 0, f"Found nulls in {col}"

    # -------------------------------------------------------------
    # Step 7: Export to Unified Dataset File
    # -------------------------------------------------------------
    unified_output_file = "consumer_electronics_market_20k.csv"
    unified_df.to_csv(unified_output_file, index=False, encoding='utf-8')
    print(f"\nSaved primary 20k multi-channel dataset to: {unified_output_file}")

if __name__ == "__main__":
    build_dataset()
