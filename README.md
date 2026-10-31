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
