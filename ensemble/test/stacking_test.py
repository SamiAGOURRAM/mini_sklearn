import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier as skStackingClassifier
from ensemble.stacking import StackingClassifier, StackingRegressor
from sklearn.ensemble import StackingRegressor as skStackingRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor as skStackingRegressor

# Stacking classifier

X, y = load_iris(return_X_y=True)
clf1 = StackingClassifier(estimators=[RandomForestClassifier(random_state=0),
                                      GradientBoostingClassifier(random_state=0),
                                      SVC(random_state=0, probability=True)],
                          final_estimator=LogisticRegression(random_state=0)).fit(X, y)
clf2 = skStackingClassifier(estimators=[("rf", RandomForestClassifier(random_state=0)),
                                        ("gbdt", GradientBoostingClassifier(random_state=0)),
                                        ("svc", SVC(random_state=0, probability=True))],
                            final_estimator=LogisticRegression(random_state=0)).fit(X, y)
trans1 = clf1.transform(X)
trans2 = clf2.transform(X)
assert np.allclose(trans1, trans2)
pred1 = clf1.predict(X)
pred2 = clf2.predict(X)
assert np.array_equal(pred1, pred2)
prob1 = clf1.predict_proba(X)
prob2 = clf2.predict_proba(X)
assert np.allclose(prob1, prob2)



# Stacking regressor
X, y = fetch_california_housing(return_X_y=True)
reg1 = StackingRegressor(estimators=[RandomForestRegressor(random_state=0),
                                     GradientBoostingRegressor(random_state=0),
                                     SVR()],
                         final_estimator=Ridge(random_state=0)).fit(X, y)
reg2 = skStackingRegressor(estimators=[("rf", RandomForestRegressor(random_state=0)),
                                       ("gbdt", GradientBoostingRegressor(random_state=0)),
                                       ("svr", SVR())],
                           final_estimator=Ridge(random_state=0)).fit(X, y)
trans1 = reg1.transform(X)
trans2 = reg2.transform(X)
assert np.allclose(trans1, trans2)
pred1 = reg1.predict(X)
pred2 = reg2.predict(X)
assert np.allclose(pred1, pred2)


