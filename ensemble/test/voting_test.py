import numpy as np
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.ensemble import VotingClassifier as skVotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from ensemble.voting import VotingClassifier, VotingRegressor
from sklearn.ensemble import VotingRegressor as skVotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from metrics.regression import r2_score

# soft voting
X, y = load_iris(return_X_y = True)
clf1 = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=15000, random_state=0)
clf2 = RandomForestClassifier(n_estimators=100, random_state=0)
eclf1 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2)]).fit(X, y)

# Print accuracy of your model
print("My Soft Voting Classifier Accuracy:", accuracy_score(y, eclf1.predict(X)))

# Print accuracy of sklearn's model
eclf2 = skVotingClassifier(estimators=[('lr', clf1), ('rf', clf2)]).fit(X, y)
print("Sklearn's Soft Voting Classifier Accuracy:", accuracy_score(y, eclf2.predict(X)))

prob1 = eclf1.transform(X)
prob2 = eclf2.transform(X)
assert np.allclose(prob1, prob2)
pred1 = eclf1.predict(X)
pred2 = eclf2.predict(X)
assert np.allclose(pred1, pred2)

# hard voting
X, y = load_iris(return_X_y = True)
clf1 = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=15000, random_state=0)
clf2 = RandomForestClassifier(n_estimators=100, random_state=0)
eclf1 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2)], voting='soft').fit(X, y)

# Print accuracy of your model
print("My Hard Voting Classifier Accuracy:", accuracy_score(y, eclf1.predict(X)))

# Print accuracy of sklearn's model
eclf2 = skVotingClassifier(estimators=[('lr', clf1), ('rf', clf2)],
                           voting='soft', flatten_transform=False).fit(X, y)
print("Sklearn's Hard Voting Classifier Accuracy:", accuracy_score(y, eclf2.predict(X)))

prob1 = eclf1.transform(X)
prob2 = eclf2.transform(X)
assert np.allclose(prob1, prob2)
prob1 = eclf1.predict_proba(X)
prob2 = eclf2.predict_proba(X)
assert np.allclose(prob1, prob2)
pred1 = eclf1.predict(X)
pred2 = eclf2.predict(X)
assert np.array_equal(pred1, pred2)

###########################################################################

X, y = fetch_california_housing(return_X_y=True)
clf1 = LinearRegression()
clf2 = RandomForestRegressor(n_estimators=100, random_state=0)
eclf1 = VotingRegressor(estimators=[('lr', clf1), ('rf', clf2)]).fit(X, y)
eclf2 = skVotingRegressor(estimators=[('lr', clf1), ('rf', clf2)]).fit(X, y)

prob1 = eclf1.transform(X)
prob2 = eclf2.transform(X)
assert np.allclose(prob1, prob2)
pred1 = eclf1.predict(X)
pred2 = eclf2.predict(X)
assert np.allclose(pred1, pred2)

print("Voting Regressor R2 Score (My Model):", r2_score(y, eclf1.predict(X)))
print("Voting Regressor R2 Score (Sklearn's Model):", r2_score(y, eclf2.predict(X)))

print("--------------------------------------------------------")
print("Test passed!")