import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error,accuracy_score,f1_score,recall_score
# Generate random data for demonstration (replace with your actual dataset)
from google.colab import files
uploaded = files.upload()
df=pd.read_csv('kidney_disease.csv')
np.random.seed(42)
num_samples = int(input("enter the comparision"))
serum_creatinine = np.random.uniform(0.5, 2.0, num_samples)
age = np.random.randint(20, 80, num_samples)
egfr = 141 * (np.minimum(serum_creatinine / 0.7, 1) ** -0.329) * (np.maximum(serum_creatinine / 0.7, 1) ** -1.209) * (0.993 ** age) * 1.018

# Create a DataFrame to hold the data
data = pd.DataFrame({'SerumCreatinine': serum_creatinine, 'Age': age, 'eGFR': egfr})

# Access the 'SerumCreatinine' and 'Age' columns as features (X) and the 'eGFR' column as the target (y)
X = data[['SerumCreatinine', 'Age']]
y_true = data['eGFR']

# Initialize and fit the linear regression model
model = RandomForestRegressor()
model.fit(X, y_true)

# Coefficients (slope) and intercept of the linear regression model
#coefficients = model.coef_
#intercept = model.intercept_

#print("Coefficients:", coefficients)
#print("Intercept:", intercept)

# Function to calculate eGFR based on serum creatinine and age
def calculate_egfr(serum_creatinine, age):
    return model.predict([[serum_creatinine, age]])[0]

# Lists to store predicted eGFR and actual eGFR for evaluation
y_pred = []
for index, row in data.iterrows():
    serum_creatinine_value = row['SerumCreatinine']
    age_value = row['Age']
    predicted_egfr = calculate_egfr(serum_creatinine_value, age_value)
    y_pred.append(predicted_egfr)

    # Check kidney health status based on predicted eGFR
    if predicted_egfr > 60:
        print(f"Data point {index+1}: Predicted eGFR: {predicted_egfr} - Kidney normal")
    elif 15 <= predicted_egfr <= 60:
        print(f"Data point {index+1}: Predicted eGFR: {predicted_egfr} - Kidney disease detected")
    else:
        print(f"Data point {index+1}: Predicted eGFR: {predicted_egfr} - Kidney failure")

# Calculate evaluation metrics
mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("R-squared (R2) Score:", r2)

# Plot a graph of metrics
metric_names = ['MSE', 'MAE', 'R2']
metric_values = [mse, mae, r2]
import matplotlib.pyplot as plt
plt.bar(metric_names, metric_values)
plt.xlabel('Metric')
plt.ylabel('Value')
plt.title('Regression Model Evaluation Metrics')
plt.show()
#import matplotlib.pyplot as plt

# Assuming you have calculated the MSE, MAE, and R2 values and stored them in 'metric_values' list
metric_names = ['MSE', 'MAE', 'R2']
metric_values = [mse, mae, r2]

plt.plot(metric_names, metric_values, marker='o', linestyle='-', color='blue')
plt.xlabel('Metric')
plt.ylabel('Value')
plt.title('Regression Model Evaluation Metrics')
plt.show()


