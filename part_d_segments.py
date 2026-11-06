import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

shipments=pd.read_csv("data/modelling_table.csv")

hub_profile=shipments.groupby("hub_id").agg(
    shipments=("hub_id","size"),
    mean_delivery_hours=("delivery_hours","mean"),
    delay_rate=("is_delayed","mean"),
    mean_distance_km=("distance_km","mean"),
    mean_n_stops=("n_stops","mean")
).reset_index()
hub_profile.to_csv("outputs/hub_profile.csv",index=False)
hub_profile = hub_profile.sort_values("hub_id").reset_index(drop=True)
X = hub_profile[["mean_delivery_hours","delay_rate","mean_distance_km","mean_n_stops"]]

scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)
clustering=KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)
clustering.fit(X_scaled)
cluster_labels=clustering.labels_
hub_profile["cluster"]=cluster_labels
hub_profile.to_csv("outputs/hub_profile.csv", index=False)
distances = pairwise_distances(X_scaled)
row_idx, col_idx = np.triu_indices(len(cluster_labels), k=1)
pair_distances = distances[row_idx, col_idx]
same_cluster = (
    cluster_labels[row_idx] == cluster_labels[col_idx]
)
within_distance = pair_distances[same_cluster].mean()
across_distance = pair_distances[~same_cluster].mean()
ratio = across_distance / within_distance
separation_results = []

for k in [2, 3, 4, 5]:
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    model.fit(X_scaled)
    model_labels = model.labels_
    row_idx, col_idx = np.triu_indices(len(model_labels), k=1)
    pair_distances = distances[row_idx, col_idx]
    same_cluster = (
    model_labels[row_idx] == model_labels[col_idx])
    model_within_distance = pair_distances[same_cluster].mean()
    model_across_distance = pair_distances[~same_cluster].mean()
    model_ratio = model_across_distance / model_within_distance
    
    cluster_sizes = sorted(
        np.bincount(model_labels).tolist(),
        reverse=True
    )

    separation_results.append({
        "k": k,
        "cluster_sizes": "-".join(map(str, cluster_sizes)),
        "within_distance": model_within_distance,
        "across_distance": model_across_distance,
        "ratio": model_ratio
    })

separation_df = pd.DataFrame(separation_results)
separation_df.to_csv("outputs/part_d_separation.csv", index=False)
cluster_count=hub_profile.groupby("cluster").size()
average_delay_rate = hub_profile.groupby("cluster")["delay_rate"].mean()
average_delivery_hours=hub_profile.groupby("cluster")["mean_delivery_hours"].mean()
average_distance_cluster=hub_profile.groupby("cluster")["mean_distance_km"].mean()
average_n_stop=hub_profile.groupby("cluster")["mean_n_stops"]

cluster_summary = hub_profile.groupby("cluster").agg(
    number_of_hubs=("hub_id", "size"),
    average_delay_rate=("delay_rate", "mean"),
    average_delivery_hours=("mean_delivery_hours", "mean"),
    average_distance_km=("mean_distance_km", "mean"),
    average_n_stops=("mean_n_stops", "mean")
)


segment_mapping = {
    2: "Standard",
    0: "Watch",
    1: "Escalate"
}
hub_profile["segment"] = hub_profile["cluster"].map(segment_mapping)
hubs = pd.read_csv("data/hubs.csv")
hub_segments = hub_profile.merge(
    hubs[["hub_id", "hub_city", "hub_tier"]],
    on="hub_id",
    how="left"
)
hub_segments = hub_segments[["hub_id", "segment", "hub_city", "hub_tier"]]
hub_segments.to_csv("outputs/hub_segments.csv",index=False)

segment_profile = hub_profile.groupby("segment").agg(
    hubs=("hub_id", "size"),
    mean_delivery_hours=("mean_delivery_hours", "mean"),
    delay_rate=("delay_rate", "mean"),
    mean_distance_km=("mean_distance_km", "mean"),
    mean_n_stops=("mean_n_stops", "mean")
).reset_index()
segment_profile.to_csv("outputs/segment_profile.csv",index=False)

tier_mapping = {
    1: "Standard",
    2: "Watch",
    3: "Escalate"
}
hub_segments["expected_segment"] = ( hub_segments["hub_tier"].map(tier_mapping))
mismatches = hub_segments[
    hub_segments["segment"] != hub_segments["expected_segment"]]
mismatches = mismatches[["hub_id", "segment", "hub_city", "hub_tier"]]
mismatches.to_csv( "outputs/tier_vs_segment_mismatch.csv",index=False)