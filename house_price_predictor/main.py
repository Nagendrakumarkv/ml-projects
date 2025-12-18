import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# ==========================================
# STEP 1: Generate Dummy Data
# ==========================================
# In a real project, you would load this from a CSV file.
# Here, we create data: 'Size' (sq ft) and 'Price' ($)
print("1. Generating Data...")

data = {
    'Size': [650, 785, 1200, 1100, 1400, 1800, 1550, 900, 2100, 2400],
    'Price': [150000, 185000, 280000, 260000, 320000, 410000, 360000, 220000, 490000, 550000]
}

df = pd.DataFrame(data)

# Features (X) = The input data (Size)
# Target (y) = What we want to predict (Price)
X = df[['Size']] 
y = df['Price']

# ==========================================
# STEP 2: Split Data
# ==========================================
# We hide 20% of data to test the model later. 
# We don't want the model to see this data during training.
print("2. Splitting Data into Train and Test sets...")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# STEP 3: Train the Model
# ==========================================
# We use Linear Regression (finding the best fit line)
print("3. Training the Model...")

model = LinearRegression()
model.fit(X_train, y_train)

print("   Training complete!")

# ==========================================
# STEP 4: Evaluation
# ==========================================
# Let's see how well it predicts on the hidden 'Test' data
predictions = model.predict(X_test)

print("\n--- Model Evaluation ---")
# Compare predicted prices vs actual prices
for i in range(len(predictions)):
    print(f"Actual: ${y_test.iloc[i]} | Predicted: ${predictions[i]:.2f}")

# Calculate Error (Average difference between actual and predicted)
mae = mean_absolute_error(y_test, predictions)
print(f"Mean Absolute Error: ${mae:.2f}")

# ==========================================
# STEP 5: Live Prediction
# ==========================================
# Predict price for a user input (e.g., a 2000 sq ft house)
new_house_size = [[2000]]
predicted_price = model.predict(new_house_size)
print(f"\nPrediction for a 2000 sq ft house: ${predicted_price[0]:.2f}")

# ==========================================
# STEP 6: Visualization (Optional)
# ==========================================
# This will popup a chart showing the data and the prediction line
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', label='Prediction Line')
plt.xlabel('Size (sq ft)')
plt.ylabel('Price ($)')
plt.title('Housing Price Prediction')
plt.legend()
plt.show()