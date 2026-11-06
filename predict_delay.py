import pandas as pd
import numpy as np
import joblib

from src.shipment_prep import (clean_shipments,build_features,FEATURES)
raw_shipments = pd.read_csv("data/new_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")
impute_df = pd.read_csv("models/impute_values.csv")
medians = dict(zip(impute_df["column"], impute_df["median"]))
feature_order_df = pd.read_csv("models/feature_order.csv")
feature_order = feature_order_df["feature"].tolist()
threshold_df = pd.read_csv("models/delay_threshold.csv")
threshold = float(threshold_df.loc[0, "threshold"])
model = joblib.load("models/delay_model.pkl")
clean_new, _ = clean_shipments(raw_shipments, medians=medians)
features_new = build_features(clean_new, hubs)
X_new = features_new[feature_order]

probabilities = model.predict_proba(X_new)[:, 1]
delay_probabilities = np.round(probabilities, 4)
flagged = (delay_probabilities >= threshold).astype(int)
hub_segments = pd.read_csv("outputs/hub_segments.csv")
hub_segments = hub_segments[["hub_id", "segment"]]
results = features_new[["shipment_id", "hub_id"]].copy()
results = results.merge(hub_segments, on="hub_id", how="left", sort=False)
results["delay_probability"] = delay_probabilities
results["flagged"] = flagged

conditions = [
    (results["flagged"] == 1) & (results["segment"] == "Escalate"),
    (results["flagged"] == 1)]
choices = [
    "Escalate to hub manager",
    "Proactive customer call"
]
results["action"] = np.select(
    conditions,
    choices,
    default="No action"
)

results = results[
    [
        "shipment_id",
        "hub_id",
        "segment",
        "delay_probability",
        "flagged",
        "action"
    ]
]

results.to_csv("outputs/new_shipment_actions.csv", index=False)
print("Final shipment actions saved successfully!")
print("Total rows:", len(results))
print("Unique shipment IDs:", results["shipment_id"].nunique())
print("Missing values:", results.isnull().sum().sum())
print("\nAction distribution:")
print(results["action"].value_counts())

print("\nFlagged shipments:", results["flagged"].sum())