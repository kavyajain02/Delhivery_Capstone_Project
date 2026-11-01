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
 Gradient Boosting  ,4.9165 
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
 0.1  ,4.9165  ,0.9306 
 1.0  ,7.3701 , 0.8440 

The learning rate 0.1 gave the best result. When it reached  0.01, the model learned too slowly with 100 estimators and finally with 1.0, the updates were too fast and the model went above the  useful improvements. So, 0.1 gave a best balance.

# Cross-validation and final model
 i have implemented 5-fold cross-validation to compare the two best models on the combined training and validation data.

Model , Mean R2_score ,Standard deviation 
Linear Regression ,0.9261 ,0.0048 
Gradient Boosting ,0.9353 ,0.0045|

Gradient Boosting had a slightly higher cross-validation R2_score, while Linear Regression had the lower validation RMSE. hence i  chose Linear Regression as the final model 

# Final test results
Testing the Linear Regression model 

Metric ,Result 
 MAE , 3.5671 hours 
 MSE ,20.4624 
 RMSE , 4.5235 hours 
 R2_score , 0.9493 

The final test R2_score was 0.9493, which means the model explains about 94.93% of the variation in delivery time. It also performed well on data that was not trained on which means it fits well
