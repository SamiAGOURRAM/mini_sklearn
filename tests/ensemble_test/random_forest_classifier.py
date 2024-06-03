import warnings
warnings.filterwarnings("ignore")

import time
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier as skRandomForestClassifier
from ensemble.random_forest import RandomForestClassifier

# Load the breast cancer dataset
X, y = load_breast_cancer(return_X_y=True)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# Setting parameters
n_estimators = 100
criterion = 'gini'
max_depth = None
min_samples_split = 2
max_features = 'auto'
bootstrap = True
random_state = 0

# Training your random forest classifier
start_time = time.time()
my_clf = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                                min_samples_split=min_samples_split, max_features=max_features,
                                bootstrap=bootstrap, random_state=random_state).fit(X_train, y_train)
time_my_clf = time.time() - start_time
print("Time taken to fit my random forest classifier:", time_my_clf, "seconds")

# Training scikit-learn's random forest classifier
start_time = time.time()
sk_clf = skRandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                                  min_samples_split=min_samples_split, max_features=max_features,
                                  bootstrap=bootstrap, random_state=random_state).fit(X_train, y_train)
time_sk_clf = time.time() - start_time
print("Time taken to fit scikit-learn's random forest classifier:", time_sk_clf, "seconds")

# Calculating accuracy scores
accuracy_score_your_model = accuracy_score(y_test, my_clf.predict(X_test))
accuracy_score_sklearn_model = accuracy_score(y_test, sk_clf.predict(X_test))

# Printing the accuracy scores
print("Accuracy score of my random forest classifier:", accuracy_score_your_model)
print("Accuracy score of scikit-learn's random forest classifier:", accuracy_score_sklearn_model)

print("--------------------------------------------------------------------------------")

# Additional test with different parameters
n_estimators = 50
max_depth = 10

# Training your random forest classifier with new parameters
start_time = time.time()
my_clf = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                                min_samples_split=min_samples_split, max_features=max_features,
                                bootstrap=bootstrap, random_state=random_state).fit(X_train, y_train)
time_my_clf = time.time() - start_time
print("Time taken to fit my random forest classifier with max_depth=10:", time_my_clf, "seconds")

# Training scikit-learn's random forest classifier with new parameters
start_time = time.time()
sk_clf = skRandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                                  min_samples_split=min_samples_split, max_features=max_features,
                                  bootstrap=bootstrap, random_state=random_state).fit(X_train, y_train)
time_sk_clf = time.time() - start_time
print("Time taken to fit scikit-learn's random forest classifier with max_depth=10:", time_sk_clf, "seconds")

# Calculating accuracy scores with new parameters
accuracy_score_your_model = accuracy_score(y_test, my_clf.predict(X_test))
accuracy_score_sklearn_model = accuracy_score(y_test, sk_clf.predict(X_test))

# Printing the accuracy scores with new parameters
print("Accuracy score of my random forest classifier with max_depth=10:", accuracy_score_your_model)
print("Accuracy score of scikit-learn's random forest classifier with max_depth=10:", accuracy_score_sklearn_model)
