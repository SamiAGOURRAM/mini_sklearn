from base.base import BaseEstimator, ClassifierMixin, RegressorMixin
import numpy as np


class BaseDecisionTree(BaseEstimator):
    def __init__(self, criterion, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth if max_depth is not None else np.inf
        self.min_samples_split = min_samples_split
        self.depth = None
        self.criterion = criterion

    def apply(self, X):
        pred = np.zeros(X.shape[0], dtype=int)
        for i in range(X.shape[0]):
            cur_node = 0
            while self.tree_[cur_node].left_child != -1:
                if X[i][self.tree_[cur_node].feature] <= self.tree_[cur_node].threshold:
                    cur_node = self.tree_[cur_node].left_child
                else:
                    cur_node = self.tree_[cur_node].right_child
            pred[i] = cur_node
        return pred
    
    @property
    def feature_importances_(self):
        importances = np.zeros(self.n_features)
        for node in self.tree_:
            if node.left_child != -1:
                left_child = self.tree_[node.left_child]
                right_child = self.tree_[node.right_child]
                importances[node.feature] += (node.n_node * node.impurity
                                              - left_child.n_node * left_child.impurity
                                              - right_child.n_node * right_child.impurity)
        return importances / np.sum(importances)


class TreeNode():
    def __init__(self):
        self.left_child = -1
        self.right_child = -1
        self.feature = None
        self.threshold = None
        self.impurity = None
        self.n_node = None
        self.value = None


class DecisionTreeClassifier(ClassifierMixin, BaseDecisionTree):
    
    def __init__(self, criterion='gini', max_depth=None, min_samples_split=2):
        if criterion not in ['gini', 'entropy']:
            raise ValueError("The criterion for decision tree classifiers should be either 'gini' or 'criterion'.")
         
        super().__init__(criterion, max_depth, min_samples_split)

    def _gini(self, y_cnt):
        prob = y_cnt / np.sum(y_cnt)
        return 1 - np.sum(np.square(prob))
    
    def _entropy(self, y_cnt):
        prob = y_cnt / np.sum(y_cnt)
        prob = prob[prob > 0]
        return -np.sum(prob * np.log2(prob))

    def _build_tree(self, X, y, cur_depth, parent, is_left):
        if self.criterion == 'gini':
            criterion_func = self._gini
        else:
            criterion_func = self._entropy

        if cur_depth == self.max_depth or np.unique(y).shape[0] == 1 or len(y) < self.min_samples_split:
            cur_node = TreeNode()
            cur_node.n_node = X.shape[0]
            cur_node.value = np.bincount(y, minlength=self.n_classes_)
            cur_node.impurity = criterion_func(cur_node.value)
            cur_id = len(self.tree_)
            self.tree_.append(cur_node)
            if parent is not None:
                if is_left:
                    self.tree_[parent].left_child = cur_id
                else:
                    self.tree_[parent].right_child = cur_id

            
            if self.depth is None:
                self.depth = cur_depth
            else:
                self.depth = max(cur_depth, self.depth)
            return
        
        best_improvement = -np.inf
        best_feature = None
        best_threshold = None
        best_left_ind = None
        best_right_ind = None
        y_cnt = np.bincount(y, minlength=self.n_classes_)
        for i in range(X.shape[1]):
            ind = np.argsort(X[:, i])
            y_cnt_left = np.bincount([], minlength=self.n_classes_)
            y_cnt_right = y_cnt.copy()
            n_left = 0
            n_right = X.shape[0]
            for j in range(ind.shape[0] - 1):
                y_cnt_left[y[ind[j]]] += 1
                y_cnt_right[y[ind[j]]] -= 1
                n_left += 1
                n_right -= 1
                if j + 1 < ind.shape[0] - 1 and np.isclose(X[ind[j], i], X[ind[j + 1], i]):
                    continue
                cur_improvement = -n_left * criterion_func(y_cnt_left) - n_right * criterion_func(y_cnt_right)
                if cur_improvement > best_improvement:
                    best_improvement = cur_improvement
                    best_feature = i
                    best_threshold = X[ind[j], i]
                    best_left_ind = ind[:j + 1]
                    best_right_ind = ind[j + 1:]
        cur_node = TreeNode()
        cur_node.feature = best_feature
        cur_node.threshold = best_threshold
        cur_node.n_node = X.shape[0]
        cur_node.value = y_cnt
        cur_node.impurity = criterion_func(y_cnt)
        cur_id = len(self.tree_)
        self.tree_.append(cur_node)
        if parent is not None:
            if is_left:
                self.tree_[parent].left_child = cur_id
            else:
                self.tree_[parent].right_child = cur_id
        if cur_depth < self.max_depth:
            self._build_tree(X[best_left_ind], y[best_left_ind], cur_depth + 1, cur_id, True)
            self._build_tree(X[best_right_ind], y[best_right_ind], cur_depth + 1, cur_id, False)

    def fit(self, X, y):
        self.n_features = X.shape[1]
        self.classes_, y_train = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        self.tree_ = []
        self._build_tree(X, y_train, 0, None, None)

        self._is_fitted = True
        return self

    def predict_proba(self, X):
        pred = self.apply(X)
        prob = np.array([self.tree_[p].value for p in pred])
        return prob / np.sum(prob, axis=1)[:, np.newaxis]

    def predict(self, X):
        pred = self.apply(X)
        return np.array([self.classes_[np.argmax(self.tree_[p].value)] for p in pred])


class DecisionTreeRegressor(RegressorMixin, BaseDecisionTree):

    def __init__(self, criterion='mse', max_depth=None, min_samples_split=2):
        if criterion not in ['mse', 'friedman_mse']:
            raise ValueError("The criterion for decision tree classifiers should be either 'mse' or 'friedman_mse'.")
        
        super().__init__(criterion, max_depth, min_samples_split)

    
    def fit(self, X, y):
        self.n_features = X.shape[1]
        self.tree_ = []
        self._build_tree(X, y, 0, None, None)

        self._is_fitted = True
        return self
    
    
    def _build_tree(self, X, y, cur_depth, parent, is_left):
        if cur_depth == self.max_depth or np.unique(y).shape[0] == 1 or len(y) < self.min_samples_split:
            cur_node = TreeNode()
            cur_node.impurity = np.mean(np.square(y)) - np.square(np.mean(y))
            cur_node.n_node = X.shape[0]
            cur_node.value = np.mean(y)
            cur_id = len(self.tree_)
            self.tree_.append(cur_node)
            if parent is not None:
                if is_left:
                    self.tree_[parent].left_child = cur_id
                else:
                    self.tree_[parent].right_child = cur_id

            if self.depth is None:
                self.depth = cur_depth
            else:
                self.depth = max(cur_depth, self.depth)
                
            return
        best_improvement = -np.inf
        best_feature = None
        best_threshold = None
        best_left_ind = None
        best_right_ind = None
        sum_total = np.sum(y)
        for i in range(X.shape[1]):
            sum_left = 0
            sum_right = sum_total
            n_left = 0
            n_right = X.shape[0]
            ind = np.argsort(X[:, i])
            for j in range(ind.shape[0] - 1):
                sum_left += y[ind[j]]
                sum_right -= y[ind[j]]
                n_left += 1
                n_right -= 1
                if j + 1 < ind.shape[0] - 1 and np.isclose(X[ind[j], i], X[ind[j + 1], i]):
                    continue
                cur_improvement = self._calc_improvement(sum_left, n_left, sum_right, n_right)
                if cur_improvement > best_improvement:
                    best_improvement = cur_improvement
                    best_feature = i
                    best_threshold = X[ind[j], i]
                    best_left_ind = ind[:j + 1]
                    best_right_ind = ind[j + 1:]
        cur_node = TreeNode()
        cur_node.feature = best_feature
        cur_node.threshold = best_threshold
        cur_node.impurity = np.mean(np.square(y)) - np.square(np.mean(y))
        cur_node.n_node = X.shape[0]
        cur_node.value = np.mean(y)
        cur_id = len(self.tree_)
        self.tree_.append(cur_node)
        if parent is not None:
            if is_left:
                self.tree_[parent].left_child = cur_id
            else:
                self.tree_[parent].right_child = cur_id
        if cur_depth < self.max_depth:
            self._build_tree(X[best_left_ind], y[best_left_ind], cur_depth + 1, cur_id, True)
            self._build_tree(X[best_right_ind], y[best_right_ind], cur_depth + 1, cur_id, False)

    
    def _calc_improvement(self, sum_left, n_left, sum_right, n_right):
        if self.criterion == 'mse':
            return sum_left * sum_left / n_left + sum_right * sum_right / n_right
        elif self.criterion == 'friedman_mse':
            return n_left * n_right * np.square(sum_left / n_left - sum_right / n_right)
    

    def predict(self, X):
        pred = self.apply(X)
        return np.array([self.tree_[p].value for p in pred])