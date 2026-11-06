import pandas as pd
import joblib
import os
import numpy as np

from sklearn.model_selection import train_test_split
from src.shipment_prep import FEATURES
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif

config = pd.read_csv("outputs/part_c_final_config.csv")
from src.shipment_prep import clean_shipments


selector_k = int(config.loc[0, "selector_k"])
n_estimators = int(config.loc[0, "n_estimators"])
max_depth = config.loc[0, "max_depth"]
min_samples_leaf = int(config.loc[0, "min_samples_leaf"])
threshold = float(config.loc[0, "threshold"])

if max_depth == "unlimited":
    max_depth = None

data = pd.read_csv("data/modelling_table.csv")

X = data[FEATURES]
y = data["is_delayed"]
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)
final_model = Pipeline([
    ("selector", SelectKBest(
        score_func=f_classif,
        k=selector_k
    )),
    ("model", RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    ))
])
final_model.fit(X_train, y_train)
os.makedirs("models", exist_ok=True)
joblib.dump(final_model, "models/delay_model.pkl")
threshold_df = pd.DataFrame({"threshold": [threshold]})
threshold_df.to_csv("models/delay_threshold.csv",index=False)

feature_order_df = pd.DataFrame({"feature": FEATURES})
feature_order_path = "models/feature_order.csv"
feature_order_df.to_csv(feature_order_path, index=False)

raw_shipments = pd.read_csv("data/raw_shipments.csv")
cleaned_shipments, medians = clean_shipments(raw_shipments)
impute_df = pd.DataFrame({
    "column": medians.keys(),
    "median": medians.values()
})
impute_df.to_csv("models/impute_values.csv",index=False)
loaded_model = joblib.load("models/delay_model.pkl")
original_probs = final_model.predict_proba(X_test)[:, 1]
reloaded_probs = loaded_model.predict_proba(X_test)[:, 1]
max_diff = np.max(np.abs(original_probs - reloaded_probs))
reload_check = pd.DataFrame({
    "metric": [
        "test_rows_scored",
        "reload_max_abs_probability_diff",
        "threshold"
    ],
    "value": [
        len(X_test),
        max_diff,
        threshold
    ]
})
reload_check.to_csv("outputs/part_d_reload_check.csv",index=False)
print(reload_check)