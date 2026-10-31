PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}
FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]
NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]
import pandas as pd
import numpy as np

def clean_shipments(raw,medians=None):
    clean=raw.copy()
    clean=clean.drop_duplicates()
    clean['service_type']=(
      clean['service_type'].str.strip().str.title()
    )
    clean['payment_mode']=clean['payment_mode'].fillna("Unknown")
    clean['declared_value_inr']=(
      clean['declared_value'].astype("string").str.replace("₹","",regex=False).str.replace(",","",regex=False)
    )
    clean=clean.drop(columns="declared_value")
    clean['declared_value_inr']=pd.to_numeric(clean['declared_value_inr'],errors="coerce")
    clean['dispatch_time']=(
      pd.to_datetime(clean['dispatch_time'])
    )
    clean.loc[clean['parcel_weight_kg'] > 100, 'parcel_weight_kg'] = np.nan
    columns_to_fill=[
      "distance_km",
    "parcel_weight_kg",
    "declared_value_inr"
    ]
    if medians is None:
       medians = {
    column: float(clean[column].median())
    for column in columns_to_fill
    }
    for column in columns_to_fill:
        clean[column] = pd.to_numeric(clean[column], errors="coerce").astype(float)
    for column in columns_to_fill:
        clean[column] = clean[column].fillna(medians[column])

    return clean, medians
def build_features(clean,hubs):
    features=clean.merge(hubs[["hub_id", "hub_city", "region", "hub_tier", "hub_daily_capacity"]]
                         ,on="hub_id",how="left")
    features["promised_hours"]=features["service_type"].map(PROMISED_HOURS)
    features["is_cod"]=(features["payment_mode"]=="COD").astype(int)
    features["is_weekend"]=((features["dispatch_time"].dt.dayofweek >4) & (features["dispatch_time"].dt.dayofweek <=6)).astype(int)
    features['dispatch_hour']=features['dispatch_time'].dt.hour.astype(int)
    features['log_declared_value']=np.log(features["declared_value_inr"])
    return features
