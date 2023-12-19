import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# Load the dataset
data = pd.read_csv('                 ')

# Display basic info about the dataset
print(data.head())
print(data.info())
# Check for missing values
print(data.isnull().sum())

# Summary statistics
print(data.describe())

# Visualizations
plt.figure(figsize=(10, 6))
sns.lineplot(x='Date', y='UnemploymentRate', data=data)
plt.title('Unemployment Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate')
plt.show()

# Explore correlations
correlation = data.corr()
sns.heatmap(correlation, annot=True)
plt.title('Correlation Heatmap')
plt.show()
# Prepare data for modeling
X = data[['Feature1', 'Feature2', ...]]  # Add relevant features
y = data['UnemploymentRate']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('R-squared:', r2_score(y_test, y_pred))
