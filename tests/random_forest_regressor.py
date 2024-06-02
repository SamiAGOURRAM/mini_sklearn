import warnings
warnings.filterwarnings("ignore")

import time
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor as skRandomForestRegressor
from ensemble.random_forest import RandomForestRegressor

# Load the California housing dataset
X, y = fetch_california_housing(return_X_y=True)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
print(X_train.shape)

# Setting parameters
n_estimators = 20
criterion = 'mse'
max_depth = 5
min_samples_split = 2
max_features = 'auto'
bootstrap = True
random_state = 0

# Training your random forest regressor
start_time = time.time()
my_reg = RandomForestRegressor(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                               min_samples_split=min_samples_split, max_features=max_features,
                               bootstrap=bootstrap, random_state=random_state).fit(X_train, y_train)
time_my_reg = time.time() - start_time
print("Time taken to fit my random forest regressor:", time_my_reg, "seconds")

# Training scikit-learn's random forest regressor
start_time = time.time()
sk_reg = skRandomForestRegressor(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                                 min_samples_split=min_samples_split, max_features=max_features,
                                 bootstrap=bootstrap, random_state=random_state).fit(X_train, y_train)
time_sk_reg = time.time() - start_time
print("Time taken to fit scikit-learn's random forest regressor:", time_sk_reg, "seconds")

# Calculating mean squared error
mse_your_model = mean_squared_error(y_test, my_reg.predict(X_test))
mse_sklearn_model = mean_squared_error(y_test, sk_reg.predict(X_test))

# Printing the mean squared error
print("Mean Squared Error of my random forest regressor:", mse_your_model)
print("Mean Squared Error of scikit-learn's random forest regressor:", mse_sklearn_model)
