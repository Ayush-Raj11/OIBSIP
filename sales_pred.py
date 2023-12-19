# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset (replace 'sales_data.csv' with your dataset)
data = pd.read_csv('sales_data.csv')

# Data Preprocessing
# Perform necessary preprocessing steps (handling missing values, encoding, etc.)

# Split data into features and target variable
X = data[['advertising_spend', 'target_audience', 'marketing_platform', ...]]  # Include relevant features
y = data['sales']  # Assuming 'sales' column contains sales figures

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Predict future sales based on new data
# new_data = ...  # Prepare new data for prediction
# future_sales = model.predict(new_data)
# print(f"Predicted future sales: {future_sales}")
