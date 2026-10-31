import pandas as pd
import numpy as np
import os
from src.shipment_prep import build_features , clean_shipments,FEATURES
raw=pd.read_csv("data/raw_shipments.csv")
hubs=pd.read_csv("data/hubs.csv")
clean,medians=clean_shipments(raw)
features=build_features(clean,hubs)

new_raw = pd.read_csv("data/new_shipments.csv")
new_clean, _ = clean_shipments(new_raw, medians=medians)
new_features = build_features(new_clean, hubs)

print("\nNew shipment feature null counts:")
print(new_features[FEATURES].isnull().sum())

print("\nTotal missing feature values:")
print(new_features[FEATURES].isnull().sum().sum())

features["is_delayed"]=(features["delivery_hours"]> features["promised_hours"]).astype(int)
nulls_in_features_and_target = features[
    FEATURES + ["delivery_hours", "is_delayed"]
].isna().sum().sum()
delayed_count = features["is_delayed"].sum()
delay_rate = features["is_delayed"].mean()
features.to_csv("data/modelling_table.csv",index=False)
raw_row=len(raw)
dedup=raw.drop_duplicates()
duplicate_rows_removed=len(raw)-len(dedup)
row_after_dedup=len(dedup)
missing_distance_km_before = dedup["distance_km"].isna().sum()

missing_parcel_weight_before = dedup["parcel_weight_kg"].isna().sum()

sentinel_weight_rows = (dedup["parcel_weight_kg"] > 100).sum()
missing_declared_value_before = dedup["declared_value"].isna().sum()
missing_payment_mode_before = dedup["payment_mode"].isna().sum()
service_type_spellings_before = dedup["service_type"].nunique()
service_type_spellings_after = clean["service_type"].nunique()
rows_after_merge=len(features)
hours = features["delivery_hours"]

hours_q1 = hours.quantile(0.25)
hours_q3 = hours.quantile(0.75)
iqr = hours_q3 - hours_q1

hours_lower_bound = hours_q1 - (1.5 * iqr)
hours_upper_bound = hours_q3 + (1.5 * iqr)
hours_outlier_count = (
    (hours < hours_lower_bound) |
    (hours > hours_upper_bound)
).sum()
declared_values = features["declared_value_inr"]
log_values = features["log_declared_value"]
declared_value_mean_inr = declared_values.mean()
declared_value_median_inr = declared_values.median()
log_declared_value_mean = log_values.mean()
log_declared_value_median = log_values.median()
checks = {
    "raw_rows": raw_row,
    "duplicate_rows_removed": duplicate_rows_removed,
    "rows_after_dedup": row_after_dedup,
    "missing_distance_km_before": missing_distance_km_before,
    "missing_parcel_weight_before": missing_parcel_weight_before,
    "sentinel_weight_rows": sentinel_weight_rows,
    "missing_declared_value_before": missing_declared_value_before,
    "median_distance_km": medians["distance_km"],
    "median_parcel_weight_kg": medians["parcel_weight_kg"],
    "median_declared_value_inr": medians["declared_value_inr"],
    "rows_after_hub_merge": rows_after_merge,
    "missing_payment_mode_before": missing_payment_mode_before,
    "service_type_spellings_before": service_type_spellings_before,
    "service_type_spellings_after": service_type_spellings_after,
    "nulls_in_features_and_target": nulls_in_features_and_target,
"delayed_count": delayed_count,
"delay_rate": round(delay_rate, 4),
"hours_q1": round(hours_q1, 4),
"hours_q3": round(hours_q3, 4),
"hours_lower_bound": round(hours_lower_bound, 4),
"hours_upper_bound": round(hours_upper_bound, 4),
"hours_outlier_count": hours_outlier_count,
"declared_value_mean_inr": round(declared_value_mean_inr, 4),
"declared_value_median_inr": round(declared_value_median_inr, 4),
"log_declared_value_mean": round(log_declared_value_mean, 4),
"log_declared_value_median": round(log_declared_value_median, 4),
}

os.makedirs("outputs", exist_ok=True)

checks_df = pd.DataFrame(
    list(checks.items()),
    columns=["metric", "value"]
)

checks_df.to_csv("outputs/part_a_checks.csv", index=False)
service_summary=features.groupby("service_type").agg(
    shipments=("is_delayed", "size"),
    delayed=("is_delayed", "sum"),
    delay_rate=("is_delayed","mean"),
    mean_hours=("delivery_hours","mean")
).reset_index()
service_summary=service_summary.rename(columns={"service_type":"group"})
service_summary["dimension"] = "service_type"
service_summary=service_summary[["dimension","group","shipments","delayed","delay_rate","mean_hours"]]
hub_summary=features.groupby("hub_tier").agg(
    shipments=("is_delayed", "size"),
    delayed=("is_delayed", "sum"),
    delay_rate=("is_delayed","mean"),
    mean_hours=("delivery_hours","mean")
).reset_index()
hub_summary=hub_summary.rename(columns={"hub_tier":"group"})
hub_summary["dimension"] = "hub_tier"
hub_summary=hub_summary[["dimension","group","shipments","delayed","delay_rate","mean_hours"]]
features["distance_band"]=pd.cut(features["distance_km"],bins=[0,200,400,600,2000],
                                 labels=["0-200", "201-400", "401-600", "601+"],right=True,include_lowest=True)
distance_summary = features.groupby("distance_band").agg(
    shipments=("is_delayed", "size"),
    delayed=("is_delayed", "sum"),
    delay_rate=("is_delayed", "mean"),
    mean_hours=("delivery_hours", "mean")
).reset_index()
distance_summary=distance_summary.rename(columns={"distance_band":"group"})
distance_summary["dimension"] = "distance_band"
distance_summary=distance_summary[["dimension","group","shipments","delayed","delay_rate","mean_hours"]]
pd.concat(
    [service_summary, hub_summary, distance_summary],
    ignore_index=True
)
delay_by_group = pd.concat(
    [service_summary, hub_summary, distance_summary],
    ignore_index=True
)

delay_by_group = delay_by_group.round(4)
delay_by_group.to_csv(
    "outputs/part_a_delay_by_group.csv",
    index=False
)
numeric_features = features.select_dtypes(include="number")
correlation_matrix = numeric_features.corr()

print(correlation_matrix)
correlation_matrix.round(4).to_csv(
    "outputs/part_a_correlations.csv",
    index=True
)
delivery_corr=correlation_matrix["delivery_hours"]
delivery_corr=delivery_corr.drop("delivery_hours")
abs_corr=abs(delivery_corr).sort_values(ascending=False)
