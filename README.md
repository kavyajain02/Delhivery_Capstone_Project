# Delhivery_Capstone_Project

# Part A : Data Cleaning & EDA

# Key Finding

1. Service Type: Express had the highest delay rate at 48.97% ie 119 out of 243 shipments , compared with standard and economy at 20.56% and 8.57%
2. The delay rate increases by 40% from Tier 1 to Tier 3 as it is 4% in Tier1 and 44% in Tier 3 a rise of 40% while Tier 2 had 21.5%
3. The delay rate increased with shipment distance, from 2.19% for shipments travelling 0–200 km to 85.23% for shipments travelling over 600 km. The intermediate bands had delay rates of 13.68% and 43.64%.
4. the 3 columns most strongly correlated with delivery hours excluding delivery_hours are
   promised_hours 0.837153
   distance_km 0.401345
   n_stops 0.324674
5. The original declared_value_inr had a mean of ₹1,533.0542 and a median of ₹1,121.50, suggesting a right-skewed distribution. After log transformation, the mean (7.0277) and median (7.0224) became much closer, indicating reduced skewness and a more closer value

# Part B:

# Model Results(Regression)

# Baseline

The baseline model got an R2 score of around -0.021 on the validation data. It predicts the average delivery time for every shipment, so it does not consider any shipment features. The negative score shows that this simple approach performs slightly worse than predicting it on the validation set average.

# Model comparison

Metric after training and compareing four models using the validation data.
Model , Validation RMSE (hours)
Linear Regression , 4.8912
Gradient Boosting ,4.9165
Random Forest ,5.5059
Decision Tree , 6.2796

After observing Linear Regression gave the lowest validation RMSE of 4.8912 hours, so I selected it as the final model. Gradient Boosting performed a little lower compared to Linear Regression.

# Decision tree depth

I tested different tree depths and saw how they affected the model.
At depth 1, the training RMSE was 11.6334 and the validation RMSE was 12.1324. Both errors are high, which shows that the tree is too simple and is underfitting the data, so we do not choose that.Depth 6 gave the best validation RMSE of 6.2796.When it reached no depth limit, the training RMSE became 0, but the validation RMSE increased to 7.8368. This shows overfitting because the tree performs extremely well on training data but not as well on unseen data.

# Gradient Boosting learning rate

I tested three learning rates: 0.01, 0.1 and 1.0.

Learning rate , Validation RMSE ,Validation R2_score
0.01 ,10.0018 , 0.7127
0.1 ,4.9165 ,0.9306
1.0 ,7.3701 , 0.8440

The learning rate 0.1 gave the best result. When it reached 0.01, the model learned too slowly with 100 estimators and finally with 1.0, the updates were too fast and the model went above the useful improvements. So, 0.1 gave a best balance.

# Cross-validation and final model

i have implemented 5-fold cross-validation to compare the two best models on the combined training and validation data.

Model , Mean R2_score ,Standard deviation
Linear Regression ,0.9261 ,0.0048
Gradient Boosting ,0.9353 ,0.0045|

Gradient Boosting had a slightly higher cross-validation R2_score, while Linear Regression had the lower validation RMSE. hence i chose Linear Regression as the final model

# Final test results

Testing the Linear Regression model

Metric ,Result
MAE , 3.5671 hours
MSE ,20.4624
RMSE , 4.5235 hours
R2_score , 0.9493

The final test R2_score was 0.9493, which means the model explains about 94.93% of the variation in delivery time. It also performed well on data that was not trained on which means it fits well

# PART C:

# Model results (classification)

# 1. Train, validation, and test split

The modelling dataset was divided into training, validation, and test sets used `random_state=42`.

Dataset ,Rows ,Delayed shipments
Training: 840 ,195
Validation: 180 ,42
Test: 180 ,41
Total: 1200 278

Stratification was used to preserve the proportion of delayed and on-time shipments across the datasets.

# 2. Baseline and model comparison

The baseline model predicts every shipment as on time. Although it achieves reasonable accuracy because most shipments are on time, it fails to identify delayed shipments.

Model ,Accuracy , Precision , Recall , F1-score
Baseline : 0.7667 , 0.0000 , 0.0000 , 0.0000
Logistic Regression : 0.9056 , 0.7778 , 0.8333 , 0.8046
Decision Tree (depth 6) : — , — , — , 0.7470
Random Forest (17 features) : 0.9111 , 0.8421 , 0.7619 , 0.8000

The baseline demonstrates why accuracy alone is insufficient for this classification problem. Recall and F1-score are important because delayed shipments needs to be identified .

# 3. Feature importance and noise analysis

Feature selection was performed using `SelectKBest` with the ANOVA F-test, fitted only on the training data. Random Forest feature importances were also calculated.
The seven highest-ranked features by F-score were:

1. `distance_km`
2. `hub_tier`
3. `hub_daily_capacity`
4. `n_stops`
5. `promised_hours`
6. `rain_flag`
7. `sale_event_flag`

The 5 noise columns were `label_print_batch`, `scanner_build_no`, `tape_roll_no`, `sticker_serial_mod`, and `shelf_slot_no`.

Together, these five columns accounted for approximately 22.52% of total Random Forest feature importance. All 5 ranked outside the top seven features by ANOVA F-score. This suggests that they contribute some importance to the tree-based model despite having relatively weak individual relationships with the target.

# 4. SelectKBest feature selection

Random Forest models were evaluated using different numbers of selected features. Validation F1-score was used for comparison.

Number of selected features (k) , Validation F1-score
5 , 0.6988
7 , 0.8333
10 , 0.8140
12 , 0.8049
17 , 0.8000

Using seven features produced the highest validation F1-score of approximately 0.8333. Therefore, seven features were selected for model comparison and tuning experiments.

# 5. Hyperparameter tuning

Grid Search and Randomized Search were performed using the seven selected features and five-fold cross-validation.

Experiment , Candidates , CV folds , Total fits , Best CV accuracy , Validation F1-score
Grid Search : 27 , 5 , 135 , 0.8952 , 0.7750
Randomized Search : 9 , 5 , 45 , 0.8905 , 0.7901
Default Random Forest (7 features) : 1 , — , 1 , — , 0.8333

Grid Search selected `max_depth=6`, `min_samples_leaf=1`, and `n_estimators=100` and Randomized Search selected `max_depth=6`, `min_samples_leaf=1`, and `n_estimators=50`.However, the default Random Forest using the same seven features achieved the highest validation F1-score. Neither tuning search improved validation F1-score, so the default Random Forest was selected for the final model. This comparison illustrates that hyperparameter tuning does not necessarily improve performance on a separate validation set.

# 6. Threshold selection

The final model was trained on the training set, and predicted probabilities were evaluated on the validation set at thresholds from 0.1 to 0.9.The rule stated mentioned to choose the highest threshold value with validation recall of at least 0.80.
At a threshold of 0.5, validation results were:

Metric , Validation result
Accuracy , 0.9111
Precision , 0.7955
Recall , 0.8333
F1-score , 0.8140

A threshold of 0.6 produced a slightly higher validation F1-score of approximately 0.8205, but recall fell to 0.7619 but this does not satisfy the minimum recall requirement, the selected threshold was 0.5.

# 7. Final test evaluation

After model selection and threshold selection were completed, the final model was evaluated on the held-out test set.

Metric Final test result
Threshold ,0.5
Accuracy , 0.8667
Precision , 0.6889
Recall , 0.7561
F1-score , 0.7209

The test confusion matrix was:
Predicted on time , Predicted delayed
Actually on time 125 , 14
Actually delayed 10 , 31

The model correctly identified 31 delayed shipments and missed 10 delayed shipments. It also alerted 14 on-time shipments as delayed.The test F1-score was lower than the validation F1-score. This indicates that validation performance did not fully generalize to the held-out test set. Since the test set contains only 41 delayed shipments, individual prediction changes can have a noticeable effect on the metrics.The test set was used for final evaluation only and was not used to select features, tune hyperparameters, or choose the classification threshold.

# 8. Final model configuration

The model configuration is stored in `outputs/part_c_final_config.csv`.

Parameter :Selected value
Model :Random Forest Classifier
Feature selection :SelectKBest using ANOVA F-test
Number of selected features : 7
Number of estimators :100
Maximum depth :None
Minimum samples per leaf :1
Random state : 42
Classification threshold :0.5
Selected experiment : `default_forest_same_7_features`

# 9. Output files

The classification pipeline generates the following result files:

`outputs/part_c_models.csv` — baseline and initial model comparison.
`outputs/part_c_feature_scores.csv` — ANOVA F-scores, ranks, Random Forest importances, and noise-column indicators.
`outputs/part_c_selectk.csv` — validation F1-scores for different values of k.
`outputs/part_c_tuning.csv` — Grid Search, Randomized Search, and default model comparison.
`outputs/part_c_thresholds.csv` — validation metrics across classification thresholds.
`outputs/part_c_final_test.csv` — final test-set metrics at the selected threshold.
`outputs/part_c_final_config.csv` — selected model configuration.

# 10. Conclusion

The experiments showed that feature selection and classification-threshold selection influenced the model's validation performance. Seven features gave the best validation f1_score among the tested feature counts, and the default Random Forest outperformed both tuning searches on validation F1-score.
The threshold was selected using the required minimum-recall rule rather than simply maximizing validation F1-score. The final test F1-score was 0.7209, with recall of 0.7561. These results provide a realistic estimate of the model's performance on unseen shipments, although the relatively small number of delayed test means that the reported metrics should be interpreted with some caution.
