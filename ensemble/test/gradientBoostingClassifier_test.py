from sklearn.ensemble import GradientBoostingClassifier as SklearnGradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from metrics.classification import accuracy_score

from ensemble.boosting import GradientBoostingClassifier

# Load the breast cancer dataset
X, y = load_breast_cancer(return_X_y=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train your GradientBoostingClassifier
my_clf = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, max_depth=3, random_state=0)
my_clf.fit(X_train, y_train)

# Train scikit-learn's GradientBoostingClassifier
sk_clf = SklearnGradientBoostingClassifier(learning_rate=0.1, n_estimators=100, max_depth=3, random_state=0)
sk_clf.fit(X_train, y_train)

# Make predictions
y_pred_my_clf = my_clf.predict(X_test)
y_pred_sk_clf = sk_clf.predict(X_test)

# Calculate accuracy
accuracy_my_clf = accuracy_score(y_test, y_pred_my_clf)
accuracy_sk_clf = accuracy_score(y_test, y_pred_sk_clf)

print("Accuracy of my GradientBoostingClassifier:", accuracy_my_clf)
print("Accuracy of scikit-learn's GradientBoostingClassifier:", accuracy_sk_clf)
