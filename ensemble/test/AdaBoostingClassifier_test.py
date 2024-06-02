import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from metrics.classification import accuracy_score 
from sklearn.ensemble import AdaBoostClassifier as skAdaBoostClassifier
from ensemble.boosting import AdaBoostClassifier

X, y = load_breast_cancer(return_X_y=True)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# Training my AdaBoost classifier
my_clf = AdaBoostClassifier(n_estimators=50, random_state=0).fit(X_train, y_train)

# Training sklearn's AdaBoost classifier
sk_clf = skAdaBoostClassifier(random_state=0, n_estimators=50).fit(X_train, y_train)

# Calculating accuracy scores
accuracy_score_your_model = accuracy_score(y_test, my_clf.predict(X_test))
accuracy_score_sklearn_model = accuracy_score(y_test, sk_clf.predict(X_test))

# Printing the accuracy scores
print("Accuracy score of my AdaBoost classifier:", accuracy_score_your_model)
print("Accuracy score of sklearn's AdaBoost classifier:", accuracy_score_sklearn_model)
