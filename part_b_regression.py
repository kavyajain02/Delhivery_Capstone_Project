import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score

from src.shipment_prep import FEATURES
df=pd.read_csv("data/modelling_table.csv")
X=df[FEATURES]
y=df["delivery_hours"]
X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)
X_val,X_test,y_val,y_test=train_test_split(
    X_test,
    y_test,
    test_size=0.5,
    random_state=42
)
scaler=StandardScaler()
scaled_X_train=scaler.fit_transform(X_train)
scaled_X_test=scaler.transform(X_test)
scaled_X_val=scaler.transform(X_val)
X_train_mean=scaled_X_train.mean(axis=0)
X_test_mean=scaled_X_test.mean(axis=0)
abs_max_X_train=abs(X_train_mean).max()
abs_max_X_test=abs(X_test_mean).max()
train_rows=X_train.shape[0]
val_rows=X_val.shape[0]
test_rows=X_test.shape[0]
 
split={
   "train_rows":train_rows,
   "val_rows":val_rows,
   "test_rows":test_rows,
   "train_scaled_max_abs_mean":abs_max_X_train,
   "test_scaled_max_abs_mean":abs_max_X_test

}
split_df = pd.DataFrame(
    list(split.items()),
    columns=["metric", "value"]
)

split_df.to_csv("outputs/part_b_checks.csv",index=False)
y_train_mean=y_train.mean()
baseline_predictions = np.full(len(y_val), y_train_mean)
d=[1,2,3,4,6,8,12,None]
tree_result=[]
for i in d:
    model=DecisionTreeRegressor(max_depth=i,random_state=42)
    model.fit(X_train,y_train)
    y_pred_train=model.predict(X_train)
    y_pred_val=model.predict(X_val)
    rmse_train=np.sqrt(mean_squared_error(y_train,y_pred_train))
    rmse_val=np.sqrt(mean_squared_error(y_val,y_pred_val))
    max_depth=i
    rmse={
        "train_rmse":rmse_train,
        "val_rmse":rmse_val,
        "max_depth": "unlimited" if i is None else i    
        }
    tree_result.append(rmse)
tree_df=pd.DataFrame(tree_result)
tree_df.to_csv("outputs/part_b_tree_depth.csv",index=False)
best_tree=min(tree_result,key=lambda x:x["val_rmse"])
best_depth=best_tree["max_depth"]
print("best tree depth:",best_depth)
print("best validation score:",best_tree["val_rmse"])
pipeline=Pipeline([
         ("scaler",StandardScaler()),
         ("model",LinearRegression())
])
pipeline.fit(X_train,y_train)
y_pred=pipeline.predict(X_val)
lr_mae=mean_absolute_error(y_val,y_pred)
lr_mse=mean_squared_error(y_val,y_pred)
lr_rmse=np.sqrt(lr_mse)
lr_r2=r2_score(y_val,y_pred)

tree_model=DecisionTreeRegressor(
    max_depth=best_depth,
    random_state=42
)
tree_model.fit(X_train,y_train)
y_pred_dcr=tree_model.predict(X_val)
dcr_mae=mean_absolute_error(y_val,y_pred_dcr)
dcr_mse=mean_squared_error(y_val,y_pred_dcr)
dcr_rmse=np.sqrt(dcr_mse)
dcr_r2=r2_score(y_val,y_pred_dcr)

random_forest=RandomForestRegressor(n_estimators=100,random_state=42)
random_forest.fit(X_train,y_train)
y_pred_forest=random_forest.predict(X_val)
rf_mae=mean_absolute_error(y_val,y_pred_forest)
rf_mse=mean_squared_error(y_val,y_pred_forest)
rf_rmse=np.sqrt(rf_mse)
rf_r2=r2_score(y_val,y_pred_forest)

gbr=GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)
gbr.fit(X_train,y_train)
y_pred_gbr=gbr.predict(X_val)
gbr_mae=mean_absolute_error(y_val,y_pred_gbr)
gbr_mse=mean_squared_error(y_val,y_pred_gbr)
gbr_rmse=np.sqrt(gbr_mse)
gbr_r2=r2_score(y_val,y_pred_gbr)

baseline_mae = mean_absolute_error(y_val, baseline_predictions)
baseline_mse = mean_squared_error(y_val, baseline_predictions)
baseline_rmse = np.sqrt(baseline_mse)
baseline_r2 = r2_score(y_val, baseline_predictions)

metrics_data = [
    {
        "model": "mean_baseline",
        "split": "validation",
        "MAE": baseline_mae,
        "MSE": baseline_mse,
        "RMSE": baseline_rmse,
        "R2": baseline_r2
    },
    {
        "model": "linear_regression",
        "split": "validation",
        "MAE": lr_mae,
        "MSE": lr_mse,
        "RMSE": lr_rmse,
        "R2": lr_r2
    },
    {
        "model": "decision_tree",
        "split": "val",
        "MAE": dcr_mae,
        "MSE": dcr_mse,
        "RMSE": dcr_rmse,
        "R2": dcr_r2
    },
    {
        "model": "random_forest",
        "split": "val",
        "MAE": rf_mae,
        "MSE": rf_mse,
        "RMSE": rf_rmse,
        "R2": rf_r2
    },
    {
        "model": "gradient_boosting",
        "split": "val",
        "MAE": gbr_mae,
        "MSE": gbr_mse,
        "RMSE": gbr_rmse,
        "R2": gbr_r2
    }
]

metrics_df = pd.DataFrame(metrics_data)
metrics_df.to_csv("outputs/part_b_metrics.csv", index=False)
learning_rate=[0.01,0.1,1.0]
gblr=[]
for i in learning_rate:
    gbr_1=GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=i,
        random_state=42
    )
    gbr_1.fit(X_train,y_train)
    y_pred_gbr_1=gbr_1.predict(X_val)
    val_rmse_gbr1=np.sqrt(mean_squared_error(y_val,y_pred_gbr_1))
    val_r2=r2_score(y_val,y_pred_gbr_1)
    gblr.append(
        {
            "learning_rate":i,
            "val_rmse":val_rmse_gbr1,
            "val_r2":val_r2
        }
    )
gbdf=pd.DataFrame(gblr)
gbdf.to_csv("outputs/part_b_gb_learning_rate.csv",index=False)

X_train_val=pd.concat([X_train,X_val])
y_train_val=pd.concat([y_train,y_val])
model_ranking = [
    ("linear_regression", lr_rmse),
    ("decision_tree", dcr_rmse),
    ("random_forest", rf_rmse),
    ("gradient_boosting", gbr_rmse)
]

model_ranking = sorted(model_ranking, key=lambda x: x[1])

kf=KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
top_models = [
    ("linear_regression", pipeline),
    ("gradient_boosting", gbr)
]
cv_results = []
for name, model in top_models:
    scores = cross_val_score(
        model,
        X_train_val,
        y_train_val,
        cv=kf
    )
    cv_results.append({
        "model": name,
        "mean_r2": scores.mean(),
        "std_r2": scores.std()
    })

cv_df = pd.DataFrame(cv_results)
cv_df.to_csv("outputs/part_b_cv.csv", index=False)
final_model = pipeline
test_predictions = final_model.predict(X_test)
test_mae = mean_absolute_error(y_test, test_predictions)
test_mse = mean_squared_error(y_test, test_predictions)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test, test_predictions)

test_result = pd.DataFrame([{
    "model": "linear_regression",
    "split": "test",
    "MAE": test_mae,
    "MSE": test_mse,
    "RMSE": test_rmse,
    "R2": test_r2
}])

metrics_df = pd.concat([metrics_df, test_result], ignore_index=True)
metrics_df.to_csv("outputs/part_b_metrics.csv", index=False)
