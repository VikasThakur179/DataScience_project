from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

diabetes = load_diabetes()
x, y = diabetes.data, diabetes.target

print(f'Dataset shape: {x.shape}')
print(f'Features: {diabetes.feature_names}')
print(f'Target range: {y.min():.1f} to {y.max():.1f}')

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=42)

print(f'Training samples:{x_train.shape[0]}')
print(f'Test samples: {x_test.shape[0]}')

# train random forest regressor
model=RandomForestRegressor(n_estimators=100)
model.fit(x_train,y_train)

# let's evaluate our model
y_pred = model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)   #tell us what percentage of varience in disease progression our model.
                                # Anything above 0.4 is pretty good for this dataset

print(f'mean squared error: {mse:.2f}')
print(f'R2 score: {r2:.3f}')

# let's save our trained model

os.makedirs('models', exist_ok=True)
with open('models/diabetes_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print('Model trained and saved successfully!')
