# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_csv('c:\Users\DELL\Desktop\oasis (intern)\car datatask3.csv')  # Replace 'car_data.csv' with your dataset file

# Data preprocessing
# Assuming 'brand', 'horsepower', 'mileage', and other relevant columns exist
# Handle missing values, encode categorical variables, scale numerical features, etc.

# Split data into features and target variable
X = data[['brand', 'horsepower', 'mileage', 'feature1', 'feature2', ...]]  # Include relevant features
y = data['price']  # Assuming 'price' is the target variable

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)  # You can tune hyperparameters

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Optionally, you can save the model for future use
# import joblib
# joblib.dump(model, 'car_price_prediction_model.pkl')
