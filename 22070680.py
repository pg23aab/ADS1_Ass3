
import pandas as pd
from sklearn.preprocessing import StandardScaler

climate_data = pd.read_excel("API_19_DS2_en_excel_v2_6300761.xls")

print(climate_data.info())
print(climate_data.head())

# Data Normalization
selected_columns = ['Data Source', 'World Development Indicators', 'Unnamed: 2']

# Create a new DataFrame with selected columns
normalized_df = climate_data[selected_columns]

# Filter out non-numeric columns and handle missing values
numeric_df = climate_data.select_dtypes(include='number').dropna()

# Check if there are any numeric columns remaining
if numeric_df.shape[1] > 0:
    # Normalize the data
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(numeric_df)

    # Create a new DataFrame with normalized data
    normalized_df = pd.DataFrame(normalized_data, columns=numeric_df.columns)

    # Display the original and normalized dataframes
    print("\nOriginal DataFrame:")
    print(climate_data[selected_columns].head())

    print("\nNormalized DataFrame:")
    print(normalized_df.head())
else:
    print("No numeric columns available for normalization.")

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Sample DataFrame
data = {
    'Country': ['A', 'B', 'C', 'D', 'E'],
    'GDP per capita': [5000, 7000, 3000, 9000, 6000],
    'CO2 production per capita': [5, 8, 2, 10, 6],
    'CO2 per $ of GDP': [0.002, 0.001, 0.003, 0.001, 0.002]
}

climate_df = pd.DataFrame(data)

# Data Normalization
selected_columns = ['GDP per capita', 'CO2 production per capita', 'CO2 per $ of GDP']
numeric_df = climate_df[selected_columns].select_dtypes(include='number').dropna()

if numeric_df.shape[1] > 0:
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(numeric_df)
    normalized_df = pd.DataFrame(normalized_data, columns=numeric_df.columns)
else:
    print("No numeric columns available for normalization.")

# Clustering Analysis
kmeans = KMeans(n_clusters=2)
cluster_labels = kmeans.fit_predict(normalized_data)
climate_df['Cluster'] = cluster_labels

# Model Fitting
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 7, 9, 12])

def linear_model(x, a, b):
    return a * x + b

params, _ = curve_fit(linear_model, x, y)

# Predictions and Confidence Intervals
predicted_values = linear_model(x, *params)

# Comparative Analysis
cluster_0_data = climate_df[climate_df['Cluster'] == 0]
cluster_1_data = climate_df[climate_df['Cluster'] == 1]

# Display the results
print("Original DataFrame:")
print(climate_df)

print("\nNormalized DataFrame:")
print(normalized_df)

print("\nClustered DataFrame:")
print(climate_df[['Country', 'Cluster']])

print("\nBest-Fitting Function Parameters:")
print(params)

print("\nComparative Analysis - Cluster 0:")
print(cluster_0_data)

print("\nComparative Analysis - Cluster 1:")
print(cluster_1_data)

# Plotting
plt.scatter(x, y, label='Original Data')
plt.plot(x, predicted_values, label='Best-Fitting Function', color='red')
plt.legend()
plt.title('Model Fitting')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()

#pip install catboost

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
from catboost import CatBoostClassifier

np.random.seed(42)
X = np.random.rand(1000, 10)
y = (X[:, 0] + X[:, 1] + X[:, 2] > 1.5).astype(int)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost Classifier
xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)

# CatBoost Classifier
catboost_model = CatBoostClassifier(iterations=100, depth=5, random_state=42, verbose=0)
catboost_model.fit(X_train, y_train)
y_pred_catboost = catboost_model.predict(X_test)
accuracy_catboost = accuracy_score(y_test, y_pred_catboost)

# Compare accuracy
print(f"XGBoost Classifier Accuracy: {accuracy_xgb:.4f}")
print(f"CatBoost Classifier Accuracy: {accuracy_catboost:.4f}")

import matplotlib.pyplot as plt


accuracies = [accuracy_xgb, accuracy_catboost]
labels = ['XGBoost', 'CatBoost']

# Plotting the pie chart
plt.pie(accuracies, labels=labels, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightcoral'])
plt.title('Accuracy Comparison: XGBoost vs. CatBoost')
plt.show()

