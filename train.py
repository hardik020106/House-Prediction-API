from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#training a model

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f'R2 Score: {r2_score(y_test, y_pred)}')
print(f'Mean Absolute Error: ${mean_absolute_error(y_test, y_pred)*100000:.2f}') 

joblib.dump(model, 'house_price_model.joblib')
joblib.dump(list(X.columns), 'feature_names.joblib')