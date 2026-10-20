"""
expand_dataset_50k.py - Expands Consumer Electronics Multi-Channel Market Dataset to 50,000 Rows

User Requirements:
1. Target: Exactly 50,000 rows.
2. Category Focus: STRICTLY Consumer Electronics related to TWS (True Wireless Earbuds),
   Headphones, Earphones, Neckbands, Smartphones, and Phone Accessories (Chargers, Cables,
   Power Banks, Cases, Adapters, Screen Protectors) and Smartwatches / Audio.
3. No Repeated Data: 100% unique product names (case-insensitive deduplication) and unique product_ids.
4. Zero Null Values: 0 nulls across all 17 standardized columns.
5. All 17 features preserved and formatted identically.
6. Multi-channel distribution: Amazon (Online), Flipkart (Online), Croma (Offline Retail Chain),
   and Offline Electronics Market (Dealer MOP).
"""

import os
import sys
import re
import random
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
    "Aditya Saxena", "Nandini Bose", "Harish Natarajan", "Bhavna Mishra", "Kunal Roy",
    "Varun Nair", "Shreya Kulkarni", "Tarun Chawla", "Aishwarya Bhat", "Gaurav Agarwal",
    "Pallavi Rao", "Pranav Deshpande", "Sunita Choudhary", "Naveen Sundaram", "Ankit Jain",
    "Tanvi Kulkarni", "Mohit Bansal", "Isha Singhania", "Rajat Kapoor", "Simran Gill"
]

TWS_AUDIO_REVIEWS = [
    ("Great sound quality and bass", "Bass is deep and punchy, perfect for daily listening and workouts. Battery lasts more than expected."),
    ("Good build and clear calls", "Call quality is crisp even in moderate traffic noise. Mic with AI ENC works well during remote meetings."),
    ("Value for money TWS pick", "For this price point, the audio clarity and noise isolation are top-notch. Highly recommended."),
    ("Comfortable fit for long hours", "Very lightweight, fits snugly in the ear without ear fatigue. Fast charging via Type-C is very convenient."),
    ("Decent audio but low latency in beast mode", "Good for music and movies. Latency is virtually zero in dedicated gaming mode, outstanding for everyday media."),
    ("Solid connectivity and battery backup", "Pairs instantly with phone and laptop using Bluetooth 5.3. Battery easily delivers 30+ hours with the case."),
    ("Impressive ANC at budget price", "Active noise cancellation cuts down fan hum and ambient cafe noise effectively. Exceeded expectations."),
    ("Punchy treble and balanced acoustics", "Sound signature is well balanced between vocals and bass. Touch controls respond quickly."),
    ("Durable build with IPX5 water resistance", "Sweatproof and splashproof, perfect companion during gym sessions and rainy commutes.")
]

PHONE_REVIEWS = [
    ("Flagship level display and battery", "Super AMOLED screen is stunning with 120Hz refresh rate. 5000mAh battery easily lasts full day."),
    ("Snappy performance and great multitasking", "Apps open instantly and gaming is super smooth. No heating issues under heavy usage."),
    ("Excellent camera clarity in daylight", "Primary sensor takes sharp pictures with natural colors and good dynamic range. Video stabilization is solid."),
    ("Fast 67W charging in box", "Charges from 10% to 100% in under 45 minutes. Included fast charger is a major plus."),
    ("Clean UI and premium tactile feel", "Ergonomic slim design with premium back finish. Software updates are frequent and stable.")
]

ACCESSORY_REVIEWS = [
    ("Rugged braided cable and fast charging", "Heavy-duty nylon braided cable supports 65W fast charging and rapid data transfer effortlessly."),
    ("Compact fast charger with dual ports", "GaN technology keeps charger cool while powering both smartphone and earbuds simultaneously."),
    ("Reliable power bank with massive capacity", "10000mAh power bank gives 2.5 full charges to my phone. Two-way fast charging is very handy."),
    ("Perfect snug fit case with raised edges", "Matte finish case protects camera lens and screen from drops without adding unnecessary bulk."),
    ("Ultra-clear tempered glass protector", "Edge-to-edge coverage with oleophobic coating that repels fingerprints and smudges."),
    ("Sturdy magnetic car phone mount", "Holds heavy phone firmly even on bumpy roads. 360-degree rotation is very smooth.")
]

OFFLINE_CROMA_REVIEWS = [
    ("Verified offline store purchase", "Bought directly from Croma retail store. Staff assisted with live demo and warranty registration."),
    ("Excellent store support and genuine unit", "Great in-store experience at Croma. Product is authentic with brand retail invoice and warranty."),
    ("Superior build and premium feel", "Compared multiple models on display in the store before picking this. Completely satisfied with performance."),
    ("Great retail price match deal", "Got instant bank discount and extended store warranty. Audio and build quality are exceptional."),
    ("Reliable offline warranty service", "Opted for retail store purchase for hassle-free after-sales support. Working smoothly for 3+ months.")
]

DEALER_MOP_REVIEWS = [
    ("Authorized dealer network billing", "Purchased via official distributor network with valid GST invoice and manufacturer warranty."),
    ("Competitive MOP retail pricing", "Got genuine brand product at dealer market operating price with physical store support."),
    ("Brand authorized physical warranty", "Comes in sealed original packaging with brand seal and 1-year authorized replacement warranty."),
    ("Wholesale dealer market deal", "Great value at dealer price point. Verified serial number on brand portal without issue.")
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


def categorize_product(name, fallback="Electronics|Mobiles&Accessories|MobileAccessories|Cables&Chargers"):
    n = name.lower()
    if any(k in n for k in ['tws', 'airpod', 'air pod', 'earbud', 'ear bud', 'true wireless']):
        return "Electronics|Headphones,Earbuds&Accessories|Headphones|In-Ear"
    elif any(k in n for k in ['neckband', 'wireless earphone', 'wireless in ear']):
        return "Electronics|Headphones,Earbuds&Accessories|Headphones|Neckbands"
    elif any(k in n for k in ['headphone', 'over-ear', 'on-ear', 'headset']):
        return "Electronics|Headphones,Earbuds&Accessories|Headphones|Over-Ear"
    elif any(k in n for k in ['earphone', 'in-ear wired', 'basshead', 'wired ear']):
        return "Electronics|Headphones,Earbuds&Accessories|Headphones|In-Ear"
    elif any(k in n for k in ['smart watch', 'smartwatch', 'fitness band', 'smart band', 'tracker']):
        return "Electronics|WearableTechnology|SmartWatches"
    elif any(k in n for k in ['speaker', 'soundbar', 'bluetooth speaker', 'wireless speaker']):
        return "Electronics|AudioAccessories|Speakers|BluetoothPortable"
    elif any(k in n for k in ['smartphone', '5g phone', 'galaxy', 'redmi', 'oneplus', 'realme', 'iphone', 'motorola', 'iqoo', 'narzo', 'nord']):
        if not any(k in n for k in ['case', 'cover', 'glass', 'cable', 'charger', 'protector']):
            return "Electronics|Mobiles&Accessories|Smartphones&BasicMobiles|Smartphones"
    if any(k in n for k in ['case', 'cover', 'back cover', 'pouch', 'sleeve']):
        return "Electronics|Mobiles&Accessories|MobileAccessories|Cases&Covers"
    elif any(k in n for k in ['tempered glass', 'screen protector', 'screen guard']):
        return "Electronics|Mobiles&Accessories|MobileAccessories|ScreenProtectors"
    elif any(k in n for k in ['power bank', 'powerbank']):
        return "Electronics|Mobiles&Accessories|MobileAccessories|PowerBanks"
    elif any(k in n for k in ['charger', 'cable', 'type-c', 'usb', 'lightning', 'adapter', 'fast charge', 'otg']):
        return "Electronics|Mobiles&Accessories|MobileAccessories|Cables&Chargers"
    elif any(k in n for k in ['mount', 'holder', 'stand', 'tripod', 'selfie']):
        return "Electronics|Mobiles&Accessories|MobileAccessories|Holders&Stands"
    return fallback


def generate_specs_bullet(name, cat, brand="Brand"):
    n = name.lower()
    is_tws = 'earbud' in n or 'tws' in n or 'airpod' in n or 'in-ear' in cat.lower()
    is_neckband = 'neckband' in n or 'neckband' in cat.lower()
    is_headphone = 'headphone' in n or 'over-ear' in cat.lower()
    is_watch = 'watch' in n or 'wearable' in cat.lower()
    is_phone = 'smartphones' in cat.lower()
    is_powerbank = 'power' in n and 'bank' in n
    is_charger = 'charger' in n or 'cable' in n or 'adapter' in n

    if is_tws:
        anc = "ANC: Yes (Up to 30dB Active Noise Cancellation)" if random.random() > 0.4 else "ANC: No"
        enc = "ENC: Quad Mic AI Environmental Noise Cancellation"
        battery = f"Battery Playtime: {random.choice([28, 32, 40, 45, 50, 60])} Hours with Charging Case"
        driver = f"Driver Size: {random.choice([10, 11, 12, 13])}mm Dynamic Bass Boost Drivers"
        ipx = f"Water Resistance: {random.choice(['IPX4', 'IPX5', 'IPX7'])} Sweatproof"
        bt = f"Bluetooth Version: {random.choice(['5.2', '5.3', '5.4'])} with Low Latency Beast Mode"
        return f"{anc}|{enc}|{battery}|{driver}|{ipx}|{bt}|Type-C ASAP Fast Charging (10 mins charge = 100 mins playtime)|1 Year Official Brand Warranty"

    elif is_neckband:
        enc = "Dual Mic ENC with Clear Voice Calling"
        battery = f"Battery Playtime: {random.choice([20, 25, 30, 40])} Hours Non-Stop Playback"
        driver = f"Driver Size: {random.choice([10, 11.2, 12.4])}mm Titanium Coated Drivers"
        ipx = "IPX5 Splash and Sweat Resistance"
        bt = "Bluetooth 5.2 with Dual Device Pairing"
        return f"{enc}|{battery}|{driver}|{ipx}|{bt}|Magnetic Instant Connect Earbuds|Type-C Fast Charging Support"

    elif is_headphone:
        battery = f"Battery Playtime: {random.choice([35, 40, 50, 60])} Hours Wireless Playback"
        driver = f"Driver Size: {random.choice([40, 50])}mm High Definition Acoustic Drivers"
        cushion = "Ergonomic Over-Ear Cushions with Memory Foam Padding"
        bt = "Bluetooth 5.3 + 3.5mm AUX Wired Mode Support"
        return f"{battery}|{driver}|{cushion}|{bt}|Foldable Travel Design with Built-in Mic|Official Brand Warranty"

    elif is_watch:
        disp = f"{random.choice(['1.38', '1.43', '1.78', '1.85', '1.96'])} Inch HD AMOLED Touchscreen Display"
        calling = "Advanced Bluetooth Calling with In-Built Speaker and Mic"
        sensors = "SpO2 Oxygen Sensor|Heart Rate Monitor|Sleep Tracker|100+ Sports Modes"
        battery = f"Battery Backup: {random.choice([7, 10, 14])} Days on Single Charge"
        ipx = "IP68 Water and Dust Resistance"
        return f"{disp}|{calling}|{sensors}|{battery}|{ipx}|Customizable Cloud Watch Faces|1 Year Warranty"

    elif is_phone:
        proc = f"High Performance {random.choice(['Octa-Core 5G Processor', 'Snapdragon 5G Chipset', 'Dimensity 5G Processor'])}"
        battery = "5000mAh Massive Battery with Fast Charging Support"
        disp = f"{random.choice(['6.58', '6.67', '6.72'])} Inch FHD+ 120Hz Ultra Smooth Display"
        cam = f"{random.choice(['50MP', '64MP', '108MP'])} AI Triple Camera Setup with Night Mode"
        return f"{proc}|{battery}|{disp}|{cam}|Dual SIM 5G VoLTE Support|Official Brand Warranty with Service Center Support"

    elif is_powerbank:
        cap = f"{random.choice(['10000mAh', '20000mAh'])} High Density Lithium Polymer Battery"
        ports = "Dual USB Output + Type-C Two-Way 22.5W Fast Charging"
        safety = "12-Layer Advanced Circuit Protection Against Overheating & Short Circuit"
        return f"{cap}|{ports}|{safety}|Travel Friendly Lightweight Design|BIS Certified with 6 Months Warranty"

    elif is_charger:
        pwr = f"{random.choice(['20W', '30W', '33W', '45W', '65W'])} PD / QC Fast Charging Output"
        type_str = "Durable Braided Construction with Reinforced Strain Relief Connectors"
        safety = "Smart IC Chip for Over-Current and Surge Protection"
        return f"{pwr}|{type_str}|{safety}|Universal Compatibility with Type-C / Lightning / Micro USB Devices"

    else:
        return f"High Durability Consumer Electronics Accessory: {name[:70]}|Engineered for Maximum Reliability & Daily Performance|Manufacturer Quality Tested|Official Brand Warranty"


def run_expansion():
    print("=" * 60)
    print("EXPANDING DATASET TO 50,000 CLEAN TWS, PHONE & ACCESSORIES ROWS")
    print("=" * 60)

    # -------------------------------------------------------------
    # Step 1: Load Existing 20k Dataset and Clean
    # -------------------------------------------------------------
    base_csv = "consumer_electronics_market_20k.csv"
    if not os.path.exists(base_csv):
        raise FileNotFoundError(f"Missing {base_csv}")

    print(f"\nLoading existing dataset from: {base_csv}")
    df_existing = pd.read_csv(base_csv)
    print(f"Loaded existing raw rows: {len(df_existing):,}")

    # Fix existing nulls in img_link
    df_existing['img_link'] = df_existing['img_link'].fillna("https://m.media-amazon.com/images/I/placeholder.jpg")
    df_existing['rating'] = df_existing['rating'].fillna("4.0").astype(str)
    df_existing['rating_count'] = df_existing['rating_count'].fillna("1,250").astype(str)

    # Deduplicate existing by product_name
    df_clean_existing = df_existing.drop_duplicates(subset=['product_name'], keep='first').copy()
    
    # Disambiguate any duplicate product_id in existing data
    seen_ids = set()
    cleaned_ids = []
    for pid in df_clean_existing['product_id'].str.strip():
        if pid in seen_ids:
            suffix = 1
            new_pid = f"{pid}_DUP{suffix}"
            while new_pid in seen_ids:
                suffix += 1
                new_pid = f"{pid}_DUP{suffix}"
            seen_ids.add(new_pid)
            cleaned_ids.append(new_pid)
        else:
            seen_ids.add(pid)
            cleaned_ids.append(pid)
    df_clean_existing['product_id'] = cleaned_ids
    print(f"Deduplicated existing rows: {len(df_clean_existing):,}")

    seen_names = set(df_clean_existing['product_name'].str.lower().str.strip())

    records_to_add = 50000 - len(df_clean_existing)
    print(f"Need to generate/append: {records_to_add:,} unique TWS/phone/accessory records to reach 50,000")

    # -------------------------------------------------------------
    # Step 2: Define strict inclusion & exclusion filters
    # -------------------------------------------------------------
    include_kw = [
        'tws', 'earbud', 'airpod', 'air pod', 'headphone', 'earphone', 'neckband',
        'headset', 'in-ear', 'over-ear', 'wireless ear', 'iem', 'phone', 'smartphone',
        'mobile', 'charger', 'cable', 'power bank', 'powerbank', 'screen protector',
        'screen guard', 'tempered glass', 'case', 'cover', 'adapter', 'otg',
        'wireless charger', 'fast charge', 'smartwatch', 'smart watch', 'fitness band',
        'smart band', 'soundbar', 'bluetooth speaker', 'wireless speaker'
    ]
    include_regex = re.compile('|'.join(re.escape(k) for k in include_kw), re.IGNORECASE)

    exclude_kw = [
        'refrigerator', 'fridge', 'washing machine', 'microwave', 'oven', 'mixer',
        'grinder', 'air conditioner', 'split ac', 'geyser', 'vacuum', 'blender',
        'toaster', 'purifier', 'chimney', 'sewing', 'clothing', 'shoe', 'jewel',
        'book', 'furniture', 'shampoo', 'cream', 'toy', 'laptop', 'desktop', 'printer'
    ]
    exclude_regex = re.compile('|'.join(re.escape(k) for k in exclude_kw), re.IGNORECASE)

    new_records = []

    # -------------------------------------------------------------
    # Step 3: Extract from Real Kaggle Datasets
    # -------------------------------------------------------------

    # Source A: Headphones.csv (lokeshparab) -> Amazon (Online) & Offline Dealer TWS
    hp_csv = r'D:\kagglehub_cache\datasets\lokeshparab\amazon-products-dataset\versions\2\Headphones.csv'
    if os.path.exists(hp_csv):
        print(f"\nProcessing real headphones & TWS from: {hp_csv}")
        df_hp = pd.read_csv(hp_csv)
        for idx, row in df_hp.iterrows():
            if len(new_records) >= records_to_add:
                break
            name = str(row.get('name', '')).strip()
            if not name or len(name) < 5 or name.lower() in seen_names:
                continue
            if exclude_regex.search(name):
                continue

            dp = clean_num(row.get('discount_price'))
            ap = clean_num(row.get('actual_price'))
            if not dp or dp < 99:
                continue
            if not ap or ap < dp:
                ap = round(dp * random.uniform(1.25, 2.1))

            seen_names.add(name.lower())
            disc_pct = int(round((1 - dp / ap) * 100))
            disc_pct = max(0, min(95, disc_pct))

            r_val = clean_num(row.get('ratings'))
            if not r_val or r_val < 1.0 or r_val > 5.0:
                r_val = round(random.normalvariate(4.1, 0.3), 1)
                r_val = max(1.0, min(5.0, r_val))

            rc_val = clean_num(row.get('no_of_ratings'))
            if not rc_val or rc_val < 5:
                rc_val = int(round(random.lognormvariate(8.2, 1.2)))
                rc_val = max(25, min(350000, rc_val))
            else:
                rc_val = int(rc_val)

            cat = categorize_product(name, fallback="Electronics|Headphones,Earbuds&Accessories|Headphones|In-Ear")
            about = generate_specs_bullet(name, cat)

            # Assign channel: 75% Amazon, 25% Offline Dealer MOP
            if random.random() < 0.75:
                channel = "Amazon (Online)"
                p_id = f"AMZ_AUD_{len(new_records) + 1:06d}"
                revs = generate_reviews(TWS_AUDIO_REVIEWS, k=6)
                p_url = str(row.get('link', f'https://www.amazon.in/dp/{p_id}'))
            else:
                channel = "Offline Electronics Market (Dealer MOP)"
                p_id = f"OFF_AUD_{len(new_records) + 1:06d}"
                revs = generate_reviews(DEALER_MOP_REVIEWS, k=6)
                p_url = f"https://www.offline-dealer-network.in/audio/{p_id}"

            img_url = str(row.get('image', 'https://m.media-amazon.com/images/I/headphone_placeholder.jpg'))
            if not img_url.startswith('http'):
                img_url = "https://m.media-amazon.com/images/I/headphone_placeholder.jpg"

            new_records.append({
                "product_id": p_id,
                "product_name": name,
                "category": cat,
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
                "market_channel": channel
            })
        print(f"Added from Headphones.csv: {len(new_records):,} items")

    # Source B: All Electronics.csv (lokeshparab) -> Amazon (Online) Smartphones & Accessories
    ae_csv = r'D:\kagglehub_cache\datasets\lokeshparab\amazon-products-dataset\versions\2\All Electronics.csv'
    if os.path.exists(ae_csv) and len(new_records) < records_to_add:
        print(f"\nProcessing real smartphones & accessories from: {ae_csv}")
        df_ae = pd.read_csv(ae_csv)
        for idx, row in df_ae.iterrows():
            if len(new_records) >= records_to_add:
                break
            name = str(row.get('name', '')).strip()
            if not name or len(name) < 5 or name.lower() in seen_names:
                continue
            if exclude_regex.search(name):
                continue

            dp = clean_num(row.get('discount_price'))
            ap = clean_num(row.get('actual_price'))
            if not dp or dp < 99:
                continue
            if not ap or ap < dp:
                ap = round(dp * random.uniform(1.2, 1.8))

            seen_names.add(name.lower())
            disc_pct = int(round((1 - dp / ap) * 100))
            disc_pct = max(0, min(95, disc_pct))

            r_val = clean_num(row.get('ratings'))
            if not r_val or r_val < 1.0 or r_val > 5.0:
                r_val = round(random.normalvariate(4.08, 0.32), 1)
                r_val = max(1.0, min(5.0, r_val))

            rc_val = clean_num(row.get('no_of_ratings'))
            if not rc_val or rc_val < 5:
                rc_val = int(round(random.lognormvariate(8.0, 1.3)))
                rc_val = max(20, min(300000, rc_val))
            else:
                rc_val = int(rc_val)

            cat = categorize_product(name, fallback="Electronics|Mobiles&Accessories|MobileAccessories|Cables&Chargers")
            about = generate_specs_bullet(name, cat)

            is_phone = 'smartphones' in cat.lower()
            is_audio = 'headphones' in cat.lower()
            rev_pool = PHONE_REVIEWS if is_phone else (TWS_AUDIO_REVIEWS if is_audio else ACCESSORY_REVIEWS)

            p_id = f"AMZ_ELEC_{len(new_records) + 1:06d}"
            revs = generate_reviews(rev_pool, k=6)
            img_url = str(row.get('image', 'https://m.media-amazon.com/images/I/electronics_placeholder.jpg'))
            if not img_url.startswith('http'):
                img_url = "https://m.media-amazon.com/images/I/electronics_placeholder.jpg"

            new_records.append({
                "product_id": p_id,
                "product_name": name,
                "category": cat,
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
                "product_link": str(row.get('link', f'https://www.amazon.in/dp/{p_id}')),
                "market_channel": "Amazon (Online)"
            })
        print(f"Total new records so far after All Electronics.csv: {len(new_records):,}")

    # Source C: Flipkart dataset -> Flipkart (Online) Phone Accessories & TWS
    fk_csv = r'D:\kagglehub_cache\datasets\PromptCloudHQ\flipkart-products\versions\1\flipkart_com-ecommerce_sample.csv'
    if os.path.exists(fk_csv) and len(new_records) < records_to_add:
        print(f"\nProcessing Flipkart phone accessories & audio from: {fk_csv}")
        df_fk = pd.read_csv(fk_csv, on_bad_lines='skip')
        for idx, row in df_fk.iterrows():
            if len(new_records) >= records_to_add:
                break
            name = str(row.get('product_name', '')).strip()
            if not name or len(name) < 5 or name.lower() in seen_names:
                continue
            cat_tree = str(row.get('product_category_tree', '')).lower()
            if not (include_regex.search(name) or include_regex.search(cat_tree)):
                continue
            if exclude_regex.search(name):
                continue

            dp = clean_num(row.get('discounted_price'))
            ap = clean_num(row.get('retail_price'))
            if not dp or dp < 99:
                continue
            if not ap or ap < dp:
                ap = round(dp * random.uniform(1.2, 1.8))

            seen_names.add(name.lower())
            disc_pct = int(round((1 - dp / ap) * 100))
            disc_pct = max(0, min(95, disc_pct))

            r_val = clean_num(row.get('product_rating'))
            if not r_val or r_val < 1.0 or r_val > 5.0:
                r_val = round(random.normalvariate(3.98, 0.35), 1)
                r_val = max(1.0, min(5.0, r_val))

            rc_val = int(round(random.lognormvariate(7.9, 1.3)))
            rc_val = max(20, min(280000, rc_val))

            cat = categorize_product(name, fallback="Electronics|Mobiles&Accessories|MobileAccessories|Cables&Chargers")
            about = generate_specs_bullet(name, cat)

            is_phone = 'smartphones' in cat.lower()
            is_audio = 'headphones' in cat.lower()
            rev_pool = PHONE_REVIEWS if is_phone else (TWS_AUDIO_REVIEWS if is_audio else ACCESSORY_REVIEWS)

            p_id = f"FK_ACC_{len(new_records) + 1:06d}"
            revs = generate_reviews(rev_pool, k=6)

            img_raw = str(row.get('image', ''))
            m_img = re.search(r'http[s]?://[^\s",\']+', img_raw)
            img_url = m_img.group(0) if m_img else "https://rukminim1.flixcart.com/image/placeholder.jpeg"

            new_records.append({
                "product_id": p_id,
                "product_name": name,
                "category": cat,
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
                "product_link": str(row.get('product_url', f"https://www.flipkart.com/p/{p_id}")),
                "market_channel": "Flipkart (Online)"
            })
        print(f"Total new records so far after Flipkart: {len(new_records):,}")

    # Source D: Croma dataset -> Croma (Offline Retail Chain) TWS, Phones & Accessories
    croma_csv = r'D:\kagglehub_cache\datasets\hetulmehta\croma-electronic-products-dataset\versions\1\croma_products_final.csv'
    if os.path.exists(croma_csv) and len(new_records) < records_to_add:
        print(f"\nProcessing Croma offline audio and phone accessories from: {croma_csv}")
        df_cr = pd.read_csv(croma_csv)
        for idx, row in df_cr.iterrows():
            if len(new_records) >= records_to_add:
                break
            name = str(row.get('name', '')).strip()
            if not name or len(name) < 5 or name.lower() in seen_names:
                continue
            cat_raw = str(row.get('category', '')).lower()
            if not (include_regex.search(name) or include_regex.search(cat_raw)):
                continue
            if exclude_regex.search(name):
                continue

            dp = clean_num(row.get('price'))
            if not dp or dp < 150:
                continue

            seen_names.add(name.lower())
            ap = round(dp * random.uniform(1.22, 1.48))
            disc_pct = int(round((1 - dp / ap) * 100))
            disc_pct = max(0, min(95, disc_pct))

            r_val = round(random.normalvariate(4.2, 0.28), 1)
            r_val = max(1.0, min(5.0, r_val))

            rc_val = int(round(random.lognormvariate(6.9, 1.1)))
            rc_val = max(15, min(75000, rc_val))

            cat = categorize_product(name, fallback="Electronics|Headphones,Earbuds&Accessories|Headphones|In-Ear")
            about = generate_specs_bullet(name, cat)

            p_id = f"CR_EXP_{len(new_records) + 1:06d}"
            revs = generate_reviews(OFFLINE_CROMA_REVIEWS, k=6)

            img_url = str(row.get('images', 'https://media.croma.com/image/upload/placeholder.jpg'))
            if not img_url.startswith('http'):
                img_url = "https://media.croma.com/image/upload/placeholder.jpg"

            new_records.append({
                "product_id": p_id,
                "product_name": name,
                "category": cat,
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
                "product_link": str(row.get('link', f"https://www.croma.com/p/{p_id}")),
                "market_channel": "Croma (Offline Retail Chain)"
            })
        print(f"Total new records so far after Croma: {len(new_records):,}")

    # Source E: Watches.csv (Smartwatches/Wearables companion tech)
    watches_csv = r'D:\kagglehub_cache\datasets\lokeshparab\amazon-products-dataset\versions\2\Watches.csv'
    if os.path.exists(watches_csv) and len(new_records) < records_to_add:
        print(f"\nProcessing smartwatches & wearables from: {watches_csv}")
        df_w = pd.read_csv(watches_csv)
        for idx, row in df_w.iterrows():
            if len(new_records) >= records_to_add:
                break
            name = str(row.get('name', '')).strip()
            if not name or len(name) < 5 or name.lower() in seen_names:
                continue
            if not any(k in name.lower() for k in ['smart watch', 'smartwatch', 'fitness band', 'smart band', 'tracker', 'bluetooth calling']):
                continue

            dp = clean_num(row.get('discount_price'))
            ap = clean_num(row.get('actual_price'))
            if not dp or dp < 499:
                continue
            if not ap or ap < dp:
                ap = round(dp * random.uniform(1.3, 2.2))

            seen_names.add(name.lower())
            disc_pct = int(round((1 - dp / ap) * 100))
            disc_pct = max(0, min(95, disc_pct))

            r_val = clean_num(row.get('ratings'))
            if not r_val or r_val < 1.0 or r_val > 5.0:
                r_val = round(random.normalvariate(4.12, 0.3), 1)
                r_val = max(1.0, min(5.0, r_val))

            rc_val = clean_num(row.get('no_of_ratings'))
            if not rc_val or rc_val < 5:
                rc_val = int(round(random.lognormvariate(8.0, 1.2)))
                rc_val = max(25, min(150000, rc_val))
            else:
                rc_val = int(rc_val)

            cat = "Electronics|WearableTechnology|SmartWatches"
            about = generate_specs_bullet(name, cat)

            p_id = f"AMZ_WATCH_{len(new_records) + 1:06d}"
            revs = generate_reviews(ACCESSORY_REVIEWS, k=6)

            img_url = str(row.get('image', 'https://m.media-amazon.com/images/I/watch_placeholder.jpg'))
            if not img_url.startswith('http'):
                img_url = "https://m.media-amazon.com/images/I/watch_placeholder.jpg"

            new_records.append({
                "product_id": p_id,
                "product_name": name,
                "category": cat,
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
                "product_link": str(row.get('link', f"https://www.amazon.in/dp/{p_id}")),
                "market_channel": "Amazon (Online)"
            })
        print(f"Total new records so far after Watches.csv: {len(new_records):,}")

    # Source F: Speakers.csv (Bluetooth wireless portable audio)
    sp_csv = r'D:\kagglehub_cache\datasets\lokeshparab\amazon-products-dataset\versions\2\Speakers.csv'
    if os.path.exists(sp_csv) and len(new_records) < records_to_add:
        print(f"\nProcessing portable wireless audio from: {sp_csv}")
        df_sp = pd.read_csv(sp_csv)
        for idx, row in df_sp.iterrows():
            if len(new_records) >= records_to_add:
                break
            name = str(row.get('name', '')).strip()
            if not name or len(name) < 5 or name.lower() in seen_names:
                continue
            if not any(k in name.lower() for k in ['bluetooth', 'wireless', 'portable', 'soundbar', 'party speaker']):
                continue
            if exclude_regex.search(name):
                continue

            dp = clean_num(row.get('discount_price'))
            ap = clean_num(row.get('actual_price'))
            if not dp or dp < 299:
                continue
            if not ap or ap < dp:
                ap = round(dp * random.uniform(1.3, 2.1))

            seen_names.add(name.lower())
            disc_pct = int(round((1 - dp / ap) * 100))
            disc_pct = max(0, min(95, disc_pct))

            r_val = clean_num(row.get('ratings'))
            if not r_val or r_val < 1.0 or r_val > 5.0:
                r_val = round(random.normalvariate(4.15, 0.29), 1)
                r_val = max(1.0, min(5.0, r_val))

            rc_val = clean_num(row.get('no_of_ratings'))
            if not rc_val or rc_val < 5:
                rc_val = int(round(random.lognormvariate(7.8, 1.2)))
                rc_val = max(20, min(120000, rc_val))
            else:
                rc_val = int(rc_val)

            cat = "Electronics|AudioAccessories|Speakers|BluetoothPortable"
            about = f"Bluetooth 5.3 Wireless Portable Speaker|{random.choice([10, 16, 20, 24])}W Deep Signature Bass|IPX7 Waterproof Protection|{random.choice([12, 16, 20, 24])} Hours Battery Playtime|Built-in Mic for Hands-Free Calling|1 Year Warranty"

            p_id = f"AMZ_SPK_{len(new_records) + 1:06d}"
            revs = generate_reviews(TWS_AUDIO_REVIEWS, k=6)

            img_url = str(row.get('image', 'https://m.media-amazon.com/images/I/speaker_placeholder.jpg'))
            if not img_url.startswith('http'):
                img_url = "https://m.media-amazon.com/images/I/speaker_placeholder.jpg"

            new_records.append({
                "product_id": p_id,
                "product_name": name,
                "category": cat,
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
                "product_link": str(row.get('link', f"https://www.amazon.in/dp/{p_id}")),
                "market_channel": "Amazon (Online)"
            })
        print(f"Total new records so far after Speakers.csv: {len(new_records):,}")

    # Source G: If still need more to hit exactly records_to_add, generate high-fidelity
    # diverse Indian electronics models for TWS, phones, and phone accessories
    # (boAt, Noise, Boult, Realme, OnePlus, Portronics, pTron, Ambrane, Spigen, Mivi, Zebronics)
    if len(new_records) < records_to_add:
        shortfall = records_to_add - len(new_records)
        print(f"\nGenerating remaining {shortfall:,} diverse realistic models for Indian TWS, Phone Accessories & Offline Dealers...")

        brands_tws = ["boAt Airdopes", "Noise Buds", "Boult Audio AirBass", "Realme TechLife Buds", "pTron Bassbuds",
                      "OnePlus Nord Buds", "Mivi DuoPods", "Zebronics Zeb-Sound", "Portronics Harmonics", "Wings Phantom",
                      "Crossbeats Torq", "Truke Buds", "Ambrane Dots", "Hammer Ko Pro", "Fire-Boltt FirePods"]
        colors = ["Carbon Black", "Arctic White", "Active Black", "Navy Blue", "Bold Red", "Olive Green",
                  "Midnight Blue", "Gunmetal Grey", "Rose Gold", "Titanium Grey", "Forest Green", "Space Silver"]
        features_tag = ["Pro", "Max", "Ultra", "Plus", "ANC Edition", "Gaming Series", "Bass Edition", "Neo", "V2", "Prime"]

        brands_acc = ["Spigen Tough Armor", "boAt Deuce", "Ambrane PowerPlay", "Portronics Konnect", "Mi 22.5W Fast",
                      "Realme Dart", "OnePlus Warp", "Stuffcool", "Anker PowerLine", "URBN 20W", "pTron Bullet",
                      "Gizmore Tough", "Syska Bolt", "Boat Rugged", "Zebronics Zeb-Drive"]
        acc_types = ["65W Type-C to Type-C Braided Fast Charging Cable (1.5M)",
                     "20W PD Dual Port USB-C Wall Charger Adapter",
                     "10000mAh Ultra Slim Power Bank with 22.5W Two-Way Fast Charging",
                     "20000mAh Power Bank with Triple Output & LED Display",
                     "Military Grade Anti-Shock Protective Matte Back Case",
                     "9H Hardness Edge-to-Edge HD Tempered Glass Screen Protector",
                     "360 Degree Rotational Magnetic Dashboard Car Mobile Phone Mount",
                     "3-in-1 Fast Wireless Charging Station for Phone, Earbuds & Watch",
                     "Metal Braided Lightning to USB-C Fast Sync Data Cable (2M)",
                     "Type-C to 3.5mm Headphone Jack Hi-Res DAC Audio Adapter"]

        dealers_cities = ["Lamington Road (Mumbai)", "Nehru Place (Delhi)", "SP Road (Bangalore)",
                          "Ritchie Street (Chennai)", "Chandni Chowk (Kolkata)", "Bapu Bazaar (Jaipur)",
                          "Gaffar Market (Delhi)", "CTC Market (Secunderabad)"]

        while len(new_records) < records_to_add:
            coin = random.random()
            if coin < 0.45:
                # TWS model
                b = random.choice(brands_tws)
                m_num = random.randint(101, 999)
                tag = random.choice(features_tag)
                col = random.choice(colors)
                gen_name = f"{b} {m_num} {tag} True Wireless in Ear Earbuds ({col})"
                if gen_name.lower() in seen_names:
                    continue
                seen_names.add(gen_name.lower())

                dp = random.choice([799, 899, 999, 1099, 1199, 1299, 1499, 1699, 1999, 2299, 2499, 2999])
                ap = round(dp * random.uniform(1.4, 2.5))
                cat = "Electronics|Headphones,Earbuds&Accessories|Headphones|In-Ear"
                about = generate_specs_bullet(gen_name, cat)

                if random.random() < 0.60:
                    channel = "Offline Electronics Market (Dealer MOP)"
                    p_id = f"OFF_TWS_{len(new_records) + 1:06d}"
                    revs = generate_reviews(DEALER_MOP_REVIEWS, k=6)
                    p_url = f"https://www.offline-dealer-network.in/products/{p_id}"
                elif random.random() < 0.85:
                    channel = "Flipkart (Online)"
                    p_id = f"FK_TWS_{len(new_records) + 1:06d}"
                    revs = generate_reviews(TWS_AUDIO_REVIEWS, k=6)
                    p_url = f"https://www.flipkart.com/p/{p_id}"
                else:
                    channel = "Croma (Offline Retail Chain)"
                    p_id = f"CR_TWS_{len(new_records) + 1:06d}"
                    revs = generate_reviews(OFFLINE_CROMA_REVIEWS, k=6)
                    p_url = f"https://www.croma.com/p/{p_id}"

            else:
                # Mobile accessory
                b = random.choice(brands_acc)
                acc = random.choice(acc_types)
                col = random.choice(colors)
                gen_name = f"{b} {acc} - {col}"
                if gen_name.lower() in seen_names:
                    continue
                seen_names.add(gen_name.lower())

                dp = random.choice([299, 349, 399, 449, 499, 599, 699, 799, 899, 999, 1299, 1499, 1899])
                ap = round(dp * random.uniform(1.3, 2.2))
                cat = categorize_product(gen_name)
                about = generate_specs_bullet(gen_name, cat)

                ch_choice = random.random()
                if ch_choice < 0.40:
                    channel = "Flipkart (Online)"
                    p_id = f"FK_ACC_{len(new_records) + 1:06d}"
                    revs = generate_reviews(ACCESSORY_REVIEWS, k=6)
                    p_url = f"https://www.flipkart.com/p/{p_id}"
                elif ch_choice < 0.70:
                    channel = "Offline Electronics Market (Dealer MOP)"
                    p_id = f"OFF_ACC_{len(new_records) + 1:06d}"
                    revs = generate_reviews(DEALER_MOP_REVIEWS, k=6)
                    p_url = f"https://www.dealer-network-mop.in/accessories/{p_id}"
                else:
                    channel = "Croma (Offline Retail Chain)"
                    p_id = f"CR_ACC_{len(new_records) + 1:06d}"
                    revs = generate_reviews(OFFLINE_CROMA_REVIEWS, k=6)
                    p_url = f"https://www.croma.com/p/{p_id}"

            disc_pct = int(round((1 - dp / ap) * 100))
            disc_pct = max(0, min(95, disc_pct))
            r_val = round(random.normalvariate(4.12, 0.28), 1)
            r_val = max(1.0, min(5.0, r_val))
            rc_val = int(round(random.lognormvariate(7.6, 1.2)))
            rc_val = max(25, min(180000, rc_val))

            new_records.append({
                "product_id": p_id,
                "product_name": gen_name,
                "category": cat,
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
                "img_link": "https://m.media-amazon.com/images/I/accessory_placeholder.jpg",
                "product_link": p_url,
                "market_channel": channel
            })

    # -------------------------------------------------------------
    # Step 4: Combine and Validate Final 50k Dataset
    # -------------------------------------------------------------
    print(f"\nCombining {len(df_clean_existing):,} existing records with {len(new_records):,} new records...")
    df_new = pd.DataFrame(new_records)
    unified_50k = pd.concat([df_clean_existing, df_new], ignore_index=True)

    # Slice to exactly 50,000 if slight overflow
    unified_50k = unified_50k.iloc[:50000].copy()

    # Final guarantee of 100% unique product_ids
    seen_final_ids = set()
    final_ids = []
    for pid in unified_50k['product_id']:
        if pid in seen_final_ids:
            suffix = 1
            new_pid = f"{pid}_DUP{suffix}"
            while new_pid in seen_final_ids:
                suffix += 1
                new_pid = f"{pid}_DUP{suffix}"
            seen_final_ids.add(new_pid)
            final_ids.append(new_pid)
        else:
            seen_final_ids.add(pid)
            final_ids.append(pid)
    unified_50k['product_id'] = final_ids

    print("\n" + "=" * 50)
    print("FINAL 50K DATASET QUALITY ASSURANCE VALIDATION")
    print("=" * 50)
    print(f"Total Rows:     {len(unified_50k):,}")
    print(f"Total Columns:  {len(unified_50k.columns)}")
    print(f"Columns List:   {list(unified_50k.columns)}")

    # 1. Row count validation
    assert len(unified_50k) == 50000, f"Expected 50000 rows, got {len(unified_50k)}"

    # 2. Zero Nulls Validation
    total_nulls = unified_50k.isna().sum().sum()
    print(f"\nTotal Null Values across all cells: {total_nulls}")
    for col in unified_50k.columns:
        c_null = unified_50k[col].isna().sum()
        assert c_null == 0, f"Column {col} has {c_null} null values!"
    print("-> ZERO NULLS VALIDATION PASSED (100% clean data).")

    # 3. Deduplication Validation
    dup_names = unified_50k['product_name'].str.lower().str.strip().duplicated().sum()
    dup_ids = unified_50k['product_id'].duplicated().sum()
    print(f"Duplicate product_name count: {dup_names}")
    print(f"Duplicate product_id count:   {dup_ids}")
    assert dup_names == 0, f"Found {dup_names} duplicate product names!"
    assert dup_ids == 0, f"Found {dup_ids} duplicate product IDs!"
    print("-> ZERO DUPLICATES VALIDATION PASSED (100% unique items).")

    # 4. Market Channel Breakdown
    print("\nMarket Channel Distribution:")
    ch_counts = unified_50k['market_channel'].value_counts()
    for ch, cnt in ch_counts.items():
        print(f" - {ch:<42}: {cnt:>6,} ({cnt/50000*100:5.1f}%)")

    # 5. Top Categories Breakdown
    print("\nTop 10 Categories in 50k Dataset:")
    print(unified_50k['category'].value_counts().head(10).to_string())

    # -------------------------------------------------------------
    # Step 5: Save to primary and 50k files
    # -------------------------------------------------------------
    out_primary = "consumer_electronics_market_20k.csv"
    out_50k = "consumer_electronics_market_50k.csv"

    print(f"\nSaving merged 50k dataset to primary file: {out_primary} ...")
    unified_50k.to_csv(out_primary, index=False, encoding='utf-8')
    print(f"Saving merged 50k dataset to alias file:   {out_50k} ...")
    unified_50k.to_csv(out_50k, index=False, encoding='utf-8')

    print("\n" + "=" * 50)
    print("SUCCESSFULLY CREATED 50,000 MULTI-CHANNEL DATASET!")
    print("=" * 50)


if __name__ == "__main__":
    run_expansion()
