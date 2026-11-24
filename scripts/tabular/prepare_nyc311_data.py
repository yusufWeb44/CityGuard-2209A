import os, re, time, json, urllib.parse, urllib.request
import pandas as pd
# pyrefly: ignore[missing-import]
import numpy as np

RAW_DIR = "data/tabular/raw"
PROCESSED_DIR = "data/tabular/processed"
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

CATEGORY_QUERIES = {
    "Pothole": "complaint_type='Street Condition' AND descriptor='Pothole'",
    "Urban_Trash": "agency='DSNY'",
    "Water_Leakage": "complaint_type='Water System'",
    "Streetlight_Fault": "complaint_type='Street Light Condition'"
}

DUPLICATE_REGEX = re.compile(
    r"(duplicate|already\s+been\s+reported|previously\s+reported|same\s+condition|existing\s+service\s+request)",
    re.IGNORECASE
)

def fetch_nyc311_records(category, where_clause, limit=2500, retries=3):
    base_url = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"
    full_where = f"{where_clause} AND created_date >= '2023-01-01T00:00:00' AND created_date <= '2024-12-31T23:59:59'"
    params = {
        "$select": "unique_key,created_date,closed_date,complaint_type,descriptor,bbl,incident_address,cross_street_1,cross_street_2,intersection_street_1,intersection_street_2,latitude,longitude,resolution_description",
        "$where": full_where,
        "$limit": limit
    }
    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "CityGuard-Research/1.0"})
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=90) as response:
                data = json.loads(response.read().decode("utf-8"))
            df = pd.DataFrame(data)
            df["category"] = category
            return df
        except Exception as e:
            if attempt == retries:
                raise
            time.sleep(2)

def extract_admin_key(row):
    bbl = str(row.get("bbl", "")).strip()
    if bbl and bbl.lower() not in ("nan", "none", "") and len(bbl) >= 9:
        return f"BBL:{bbl}"
    addr = re.sub(r"\s+", " ", str(row.get("incident_address", "")).strip().upper())
    if addr and addr not in ("NAN", "NONE", ""):
        return f"ADDR:{addr}"
    is1 = re.sub(r"\s+", " ", str(row.get("intersection_street_1", "")).strip().upper())
    is2 = re.sub(r"\s+", " ", str(row.get("intersection_street_2", "")).strip().upper())
    if is1 and is2 and is1 not in ("NAN", "NONE", "") and is2 not in ("NAN", "NONE", ""):
        s = sorted([is1, is2])
        return f"INTER:{s[0]}&{s[1]}"
    cs1 = re.sub(r"\s+", " ", str(row.get("cross_street_1", "")).strip().upper())
    cs2 = re.sub(r"\s+", " ", str(row.get("cross_street_2", "")).strip().upper())
    if cs1 and cs2 and cs1 not in ("NAN", "NONE", "") and cs2 not in ("NAN", "NONE", ""):
        s = sorted([cs1, cs2])
        return f"CROSS:{s[0]}&{s[1]}"
    return None

def assign_true_incident_ids(df, start_id=1):
    raw_count = len(df)
    for col in ["latitude", "longitude", "created_date", "closed_date", "bbl", "incident_address", "cross_street_1", "cross_street_2", "intersection_street_1", "intersection_street_2", "resolution_description"]:
        if col not in df.columns:
            df[col] = np.nan
    df = df.dropna(subset=["latitude", "longitude", "created_date"]).copy()
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df[(df["latitude"].between(40.4, 41.0)) & (df["longitude"].between(-74.3, -73.6))]
    df["created_dt"] = pd.to_datetime(df["created_date"], errors="coerce")
    df["closed_dt"] = pd.to_datetime(df["closed_date"], errors="coerce")
    df = df.dropna(subset=["created_dt"]).copy()
    df["admin_key"] = df.apply(extract_admin_key, axis=1)
    df = df.dropna(subset=["admin_key"]).copy()
    df["is_official_dup_text"] = df["resolution_description"].fillna("").apply(lambda txt: bool(DUPLICATE_REGEX.search(str(txt))))

    curr_id = start_id
    grouped_frames = []
    for admin_key, group in df.groupby("admin_key", sort=False):
        group = group.sort_values("created_dt").copy()
        assigned_ids = []
        window_end = None
        for _, row in group.iterrows():
            c_time = row["created_dt"]
            e_time = row["closed_dt"] if pd.notnull(row["closed_dt"]) else (c_time + pd.Timedelta(hours=72))
            if (e_time - c_time).total_seconds() > 14 * 24 * 3600:
                e_time = c_time + pd.Timedelta(hours=72)
            if window_end is not None and c_time <= window_end:
                assigned_ids.append(f"INC_{curr_id:05d}")
                if e_time > window_end:
                    window_end = e_time
            else:
                curr_id += 1
                assigned_ids.append(f"INC_{curr_id:05d}")
                window_end = e_time
        group["True_Incident_ID"] = assigned_ids
        grouped_frames.append(group)

    cleaned_df = pd.concat(grouped_frames, ignore_index=True)
    cluster_counts = cleaned_df["True_Incident_ID"].value_counts()
    stats = {
        "raw": raw_count,
        "valid_clean": len(cleaned_df),
        "bbl_matched": int(cleaned_df["admin_key"].str.startswith("BBL:").sum()),
        "regex_dup_flag": int(cleaned_df["is_official_dup_text"].sum()),
        "unique_incidents": int(cleaned_df["True_Incident_ID"].nunique()),
        "dup_clusters": int((cluster_counts >= 2).sum())
    }
    return cleaned_df, curr_id + 1, stats

print("=" * 84)
print("CITYGUARD (IP-1.2): NYC 311 NON-CIRCULAR FILTERING & GROUND-TRUTH PIPELINE")
print("=" * 84)

all_cleaned, summary_rows, next_id = [], [], 1
for cat, query in CATEGORY_QUERIES.items():
    print(f"[INFO] Fetching & cleaning NYC 311 data for [{cat}]...")
    raw_df = fetch_nyc311_records(cat, query, limit=2500)
    cleaned_df, next_id, stats = assign_true_incident_ids(raw_df, start_id=next_id)
    all_cleaned.append(cleaned_df)
    summary_rows.append({
        "Category": cat, "Raw": stats["raw"], "Valid_Clean": stats["valid_clean"],
        "BBL_Keys": stats["bbl_matched"], "Regex_Dups": stats["regex_dup_flag"],
        "Unique_IDs": stats["unique_incidents"], "Multi_Clusters": stats["dup_clusters"]
    })

final_df = pd.concat(all_cleaned, ignore_index=True)
summary_df = pd.DataFrame(summary_rows)

print("\n--- [1/2] NON-CIRCULAR GROUND-TRUTH SUMMARY (ADMINISTRATIVE RULES ONLY) ---")
print(summary_df.to_string(index=False))

out_csv = os.path.join(PROCESSED_DIR, "nyc311_cleaned_candidates.csv")
final_df.to_csv(out_csv, index=False)

dup_ids = final_df["True_Incident_ID"].value_counts()
sample_dup_ids = dup_ids[dup_ids >= 2].head(3).index
sample_view = final_df[final_df["True_Incident_ID"].isin(sample_dup_ids)][["unique_key", "category", "admin_key", "created_date", "True_Incident_ID"]].head(8)

print("\n--- [2/2] VERIFIED DUPLICATE CLUSTERS SAMPLE (NO GPS USED) ---")
print(sample_view.to_string(index=False))
print(f"\n[SUCCESS] Saved {len(final_df)} validated records to: {out_csv}")
print("=" * 84)
