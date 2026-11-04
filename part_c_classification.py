from src.shipment_prep import FEATURES, NOISE_COLUMNS
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV
)
from sklearn.model_selection import ParameterGrid

data=pd.read_csv("data/modelling_table.csv")

X=data[FEATURES]
y=data["is_delayed"]
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
ly=len(y_val)
pred_val=np.zeros(ly)
baseline_acc=accuracy_score(y_val,pred_val)
baseline_pre=precision_score(y_val,pred_val,zero_division=0)
baseline_recall=recall_score(y_val,pred_val,zero_division=0)
baseline_f1=f1_score(y_val,pred_val,zero_division=0)
cm=confusion_matrix(y_val,pred_val,labels=[0,1])
tn=cm[0,0]
fp=cm[0,1]
tp=cm[1,1]
fn=cm[1,0]
pipeline=Pipeline([

    ("scaler",StandardScaler()),
    ("logr",LogisticRegression())
])
pipeline.fit(X_train,y_train)
y_pred_lr=pipeline.predict(X_val)
lr_acc=accuracy_score(y_val,y_pred_lr)
lr_pre=precision_score(y_val,y_pred_lr)
lr_recall=recall_score(y_val,y_pred_lr)
lr_f1=f1_score(y_val,y_pred_lr)
cm_lr=confusion_matrix(y_val,y_pred_lr,labels=[0,1])
tn_lr= cm_lr[0, 0]
fp_lr= cm_lr[0, 1]
fn_lr= cm_lr[1, 0]
tp_lr= cm_lr[1, 1]
tree_depth=[2, 3, 4, 5, 6, 8, 10]
f1_dc=[]
for i in tree_depth:
    dc=DecisionTreeClassifier(max_depth=i, criterion="gini",random_state=42 )
    dc.fit(X_train,y_train)
    y_pred_dc=dc.predict(X_val)
    dc_f1=f1_score(y_val,y_pred_dc)
    f1_dc.append(dc_f1)    
dc_f1=max(f1_dc)
best_index = f1_dc.index(dc_f1)
best_depth = tree_depth[best_index]
dc = DecisionTreeClassifier(max_depth=best_depth, criterion="gini", random_state=42)
dc.fit(X_train, y_train)

y_pred_dc = dc.predict(X_val)

dc_acc = accuracy_score(y_val, y_pred_dc)
dc_pre = precision_score(y_val, y_pred_dc, zero_division=0)
dc_recall = recall_score(y_val, y_pred_dc, zero_division=0)
dc_f1 = f1_score(y_val, y_pred_dc, zero_division=0)
cm_dc = confusion_matrix(y_val, y_pred_dc, labels=[0, 1])
tn_dc = cm_dc[0, 0]
fp_dc = cm_dc[0, 1]
fn_dc = cm_dc[1, 0]
tp_dc = cm_dc[1, 1]

rf=RandomForestClassifier(n_estimators=100,random_state=42)
rf.fit(X_train,y_train)
y_pred_rf=rf.predict(X_val)
rf_acc=accuracy_score(y_val,y_pred_rf)
rf_pre=precision_score(y_val,y_pred_rf)
rf_recall=recall_score(y_val,y_pred_rf)
rf_f1=f1_score(y_val,y_pred_rf)
cm_rf=confusion_matrix(y_val,y_pred_rf,labels=[0,1])
tn_rf= cm_rf[0, 0]
fp_rf= cm_rf[0, 1]
fn_rf= cm_rf[1, 0]
tp_rf= cm_rf[1, 1]
models = [
    {
        "model": "blind_all_on_time",
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "accuracy": baseline_acc,
        "precision": baseline_pre,
        "recall": baseline_recall,
        "f1": baseline_f1
    },
    {
        "model": "logistic_regression",
        "TN": tn_lr,
        "FP": fp_lr,
        "FN": fn_lr,
        "TP": tp_lr,
        "accuracy": lr_acc,
        "precision": lr_pre,
        "recall": lr_recall,
        "f1": lr_f1
    },
    {
        "model": f"decision_tree_depth_{best_depth}",
        "TN": tn_dc,
        "FP": fp_dc,
        "FN": fn_dc,
        "TP": tp_dc,
        "accuracy": dc_acc,
        "precision": dc_pre,
        "recall": dc_recall,
        "f1": dc_f1
    },
    {
        "model": "random_forest_default",
        "TN": tn_rf,
        "FP": fp_rf,
        "FN": fn_rf,
        "TP": tp_rf,
        "accuracy": rf_acc,
        "precision": rf_pre,
        "recall": rf_recall,
        "f1": rf_f1
    }
]
models_df = pd.DataFrame(models)
models_df.to_csv("outputs/part_c_models.csv", index=False)
forest=RandomForestClassifier(n_estimators=100,random_state=42)
forest.fit(X_train,y_train)
selector = SelectKBest(score_func=f_classif, k="all")
selector.fit(X_train,y_train)
f_scores = selector.scores_
f_ranks = pd.Series(f_scores).rank(ascending=False, method="min")
feature_df = pd.DataFrame({
    "feature": FEATURES,
    "f_score": f_scores,
    "f_rank": f_ranks
})
rf_importance = forest.feature_importances_
feature_df["rf_importance"] = rf_importance
feature_df["is_noise"] = feature_df["feature"].isin(NOISE_COLUMNS)
noise_importance = feature_df.loc[
    feature_df["is_noise"], "rf_importance"
].sum()

print("Noise importance share:", noise_importance)

feature_df = feature_df[
    ["feature", "f_score", "f_rank", "rf_importance", "is_noise"]
]
feature_df.to_csv(
    "outputs/part_c_feature_scores.csv",
    index=False
)
k_values = [5, 7, 10, 12, 17]
selectk_results = []

for k in k_values:
    pipelinek=Pipeline([
        ("selector",SelectKBest(score_func=f_classif,k=k)),
        ("rf",RandomForestClassifier(n_estimators=100,random_state=42))
    ])
    pipelinek.fit(X_train,y_train)
    y_pred_k=pipelinek.predict(X_val)
    f1_k=f1_score(y_val,y_pred_k)
    selectk_results.append({
        "k":k,
        "val_f1":f1_k
    })
selectk_df=pd.DataFrame(selectk_results)
selectk_df.to_csv("outputs/part_c_selectk.csv",index=False)

selector7 = SelectKBest(score_func=f_classif, k=7)
X_train_7 = selector7.fit_transform(X_train, y_train)
X_val_7 = selector7.transform(X_val)
param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [4, 6, 8],
    "min_samples_leaf": [1, 5, 10]
}
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5
)
grid_search.fit(X_train_7, y_train)

grid_val_pred = grid_search.best_estimator_.predict(X_val_7)
grid_val_f1 = f1_score(y_val, grid_val_pred)
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_grid,
    n_iter=9,
    cv=5,
    random_state=42
)
random_search.fit(X_train_7, y_train)
random_val_pred = random_search.best_estimator_.predict(X_val_7)
random_val_f1 = f1_score(y_val, random_val_pred)
default7 = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
default7.fit(X_train_7, y_train)
default_val_pred = default7.predict(X_val_7)
default_val_f1 = f1_score(y_val, default_val_pred)
tuning_results = [
    {
        "search": "grid_search",
        "candidates": len(grid_search.cv_results_["params"]),
        "cv_folds": grid_search.n_splits_,
        "total_fits": len(grid_search.cv_results_["params"]) * grid_search.n_splits_,
        "best_params": str(grid_search.best_params_),
        "best_cv_accuracy": grid_search.best_score_,
        "val_f1": grid_val_f1
    },
    {
        "search": "randomized_search",
        "candidates": len(random_search.cv_results_["params"]),
        "cv_folds": random_search.n_splits_,
        "total_fits": len(random_search.cv_results_["params"]) * random_search.n_splits_,
        "best_params": str(random_search.best_params_),
        "best_cv_accuracy": random_search.best_score_,
        "val_f1": random_val_f1
    },
    {
        "search": "default_forest_same_7_features",
        "candidates": 1,
        "cv_folds": 0,
        "total_fits": 1,
        "best_params": "defaults",
        "best_cv_accuracy": np.nan,
        "val_f1": default_val_f1
    }
]

tuning_df = pd.DataFrame(tuning_results)
tuning_df.to_csv(
    "outputs/part_c_tuning.csv",
    index=False
)
chosen_from = "default_forest_same_7_features"
chosen_params = {
    "n_estimators": 100,
    "max_depth": None,
    "min_samples_leaf": 1
}

final_model = Pipeline([
    ("selector", SelectKBest(score_func=f_classif, k=7)),
    ("model", RandomForestClassifier(
        n_estimators=chosen_params["n_estimators"],
        max_depth=chosen_params["max_depth"],
        min_samples_leaf=chosen_params["min_samples_leaf"],
        random_state=42
    ))
])
final_model.fit(X_train, y_train)

print("\nFinal Model Selected:", chosen_from)
print("Final Model Parameters:", chosen_params)
val_probs = final_model.predict_proba(X_val)[:, 1]

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

threshold_results = []

for threshold in thresholds:
    val_pred = (val_probs >= threshold).astype(int)
    cm = confusion_matrix(y_val, val_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    threshold_results.append({
        "threshold": threshold,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "accuracy": accuracy_score(y_val, val_pred),
        "precision": precision_score(y_val, val_pred, zero_division=0),
        "recall": recall_score(y_val, val_pred, zero_division=0),
        "f1": f1_score(y_val, val_pred, zero_division=0)
    })
threshold_df = pd.DataFrame(threshold_results)
threshold_df.to_csv(
    "outputs/part_c_thresholds.csv",
    index=False
)
eligible_thresholds = threshold_df[
    threshold_df["recall"] >= 0.80
]
if len(eligible_thresholds) > 0:
    chosen_threshold = eligible_thresholds["threshold"].max()
else:
    chosen_threshold = 0.1
test_probs = final_model.predict_proba(X_test)[:, 1]
test_pred = (test_probs >= chosen_threshold).astype(int)
test_cm = confusion_matrix(y_test, test_pred, labels=[0, 1])
tn_test, fp_test, fn_test, tp_test = test_cm.ravel()

test_results = [{
    "threshold": chosen_threshold,
    "TN": tn_test,
    "FP": fp_test,
    "FN": fn_test,
    "TP": tp_test,
    "accuracy": accuracy_score(y_test, test_pred),
    "precision": precision_score(y_test, test_pred, zero_division=0),
    "recall": recall_score(y_test, test_pred, zero_division=0),
    "f1": f1_score(y_test, test_pred, zero_division=0)
}]

test_df = pd.DataFrame(test_results)

test_df.to_csv(
    "outputs/part_c_final_test.csv",
    index=False
)
final_config = pd.DataFrame([{
    "selector_k": 7,
    "n_estimators": chosen_params["n_estimators"],
    "max_depth": (
        "unlimited" if chosen_params["max_depth"] is None
        else chosen_params["max_depth"]
    ),
    "min_samples_leaf": chosen_params["min_samples_leaf"],
    "threshold": chosen_threshold,
    "chosen_from": chosen_from
}])
final_config.to_csv(
    "outputs/part_c_final_config.csv",
    index=False
)


