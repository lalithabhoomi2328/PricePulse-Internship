
"""
PricePulse: AI-Based Dynamic Pricing and Revenue Optimization
Industry: E-commerce and Retail
"""

import pandas as pd
import numpy as np
import os
import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

os.makedirs("outputs", exist_ok=True)

# 1. Generate E-commerce Dataset
np.random.seed(42)
n = 3000

data = pd.DataFrame({
    "product_id": range(1, n + 1),
    "base_price": np.random.uniform(100, 2000, n),
    "competitor_price": np.random.uniform(100, 2000, n),
    "stock": np.random.randint(5, 500, n),
    "demand": np.random.randint(10, 500, n),
    "weather_score": np.random.uniform(0, 1, n),
    "driver_availability": np.random.uniform(0, 1, n),
    "season": np.random.randint(1, 5, n)
})

data["sales"] = (
    data["demand"] * 0.5
    + data["stock"] * 0.08
    - data["base_price"] * 0.04
    + data["weather_score"] * 30
    + data["driver_availability"] * 25
    + np.random.normal(0, 15, n)
).clip(1, None)

data.to_csv("outputs/pricepulse_dataset.csv", index=False)

# 2. Train Machine Learning Model
features = [
    "base_price", "competitor_price", "stock",
    "demand", "weather_score",
    "driver_availability", "season"
]

X = data[features]
y = data["sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("MAE:", round(mean_absolute_error(y_test, predictions), 2))
print("R2 Score:", round(r2_score(y_test, predictions), 3))

with open("outputs/pricepulse_model.pkl", "wb") as file:
    pickle.dump(model, file)

# 3. Dynamic Pricing Recommendation
def recommend_price(base_price, competitor_price, stock, demand):
    if demand > 300 and stock < 100:
        price = base_price * 1.10
    elif demand < 100 or stock > 300:
        price = base_price * 0.90
    else:
        price = base_price

    price = min(price, competitor_price * 1.15)
    return round(max(price, base_price * 0.70), 2)

data["recommended_price"] = data.apply(
    lambda row: recommend_price(
        row["base_price"],
        row["competitor_price"],
        row["stock"],
        row["demand"]
    ),
    axis=1
)

data[[
    "product_id", "base_price",
    "competitor_price", "stock",
    "demand", "recommended_price"
]].to_csv("outputs/price_recommendations.csv", index=False)

# 4. Q-Learning Pricing Simulation
states = 3
actions = 3
q_table = np.zeros((states, actions))

learning_rate = 0.1
discount = 0.9
epsilon = 0.1

for episode in range(1000):
    state = np.random.randint(states)

    for step in range(20):
        if np.random.random() < epsilon:
            action = np.random.randint(actions)
        else:
            action = np.argmax(q_table[state])

        reward = [5, 10, 3][action] - abs(state - action) * 2
        next_state = np.random.randint(states)

        q_table[state, action] += learning_rate * (
            reward + discount * np.max(q_table[next_state])
            - q_table[state, action]
        )

        state = next_state

np.savetxt("outputs/q_learning_policy.csv", q_table, delimiter=",")

# 5. Demand Forecast
forecast = data["sales"].tail(14).reset_index(drop=True)
forecast.to_csv(
    "outputs/demand_forecast.csv",
    index=False
)

# 6. Save Model Evaluation
with open("outputs/model_evaluation.txt", "w") as file:
    file.write("PricePulse Model Evaluation\n")
    file.write(f"MAE: {mean_absolute_error(y_test, predictions):.2f}\n")
    file.write(f"R2 Score: {r2_score(y_test, predictions):.3f}\n")

print("\nPricePulse Project Completed Successfully!")
print("Dataset, trained model, pricing recommendations,")
print("Q-learning policy and demand forecast saved in outputs/")
      
