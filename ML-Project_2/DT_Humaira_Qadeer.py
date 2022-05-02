from collections import Counter
import pandas as pd
import numpy as np
import seaborn as sns
# https://towardsdatascience.com/implementing-a-decision-tree-from-scratch-f5358ff9c4bb

from sklearn import datasets

from sklearn.model_selection import train_test_split


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None


class DecisionTreeModel:

    def __init__(self, max_depth=100, criterion='gini', min_samples_split=2, impurity_stopping_threshold=1):
        self.max_depth = max_depth
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.impurity_stopping_threshold = impurity_stopping_threshold
        self.root = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        # TODO
        # call the _fit method which builds the tree
        # using preprocessesed data from fit method
        # which takes in the data
        # and the categorical variable as an array
        # in y converted to numerical values
        # as parameters
        X = X.to_numpy()
        y_convert = catToInt(y)
        self._fit(X, y_convert)
        # end TODO
        print("Done fitting")

    def predict(self, X: pd.DataFrame):
        # TODO
        # call the predict method
        # method that takes in flattened data
        # and calls on the _predict method which
        # proceeds to predict data based upon previous data
        X = X.to_numpy()
        predictions = self._predict(X)
        return predictions
        # end TODO

    def _fit(self, X, y):
        # builds data with numerical and categorical features
        self.root = self._build_tree(X, y)

    def _predict(self, X):
        # traverses the tree to make predictions based on the
        # previous data which is determined to be the most viable option
        predictions = [self._traverse_tree(x, self.root) for x in X]
        return np.array(predictions)

    def _is_finished(self, depth):
        # TODO: for graduate students only, add another stopping criteria
        # modify the signature of the method if needed
        if (depth >= self.max_depth
            or self.n_class_labels == 1
                or self.n_samples < self.min_samples_split):
            return True
        # end TODO
        return False

    def _is_homogenous_enough(self):
        # TODO: for graduate students only
        result = False
        # end TODO
        return result

    def _build_tree(self, X, y, depth=0):
        self.n_samples, self.n_features = X.shape
        self.n_class_labels = len(np.unique(y))

        # stopping criteria
        if self._is_finished(depth):
            most_common_Label = np.argmax(np.bincount(y))
            return Node(value=most_common_Label)

        # get best split
        rnd_feats = np.random.choice(
            self.n_features, self.n_features, replace=False)
        best_feat, best_thresh = self._best_split(X, y, rnd_feats)

        # grow children recursively
        left_idx, right_idx = self._create_split(X[:, best_feat], best_thresh)
        left_child = self._build_tree(X[left_idx, :], y[left_idx], depth + 1)
        right_child = self._build_tree(
            X[right_idx, :], y[right_idx], depth + 1)
        return Node(best_feat, best_thresh, left_child, right_child)

    def _gini(self, y):
        # TODO
        # gini = 0 specifies that the node is pure (training instances belong to the same class)
        gini = 0
        # compute gini score using the ratio of class instances among each iteration of nodes
        proportions = np.bincount(y) / len(y)
        gini = 1 - np.sum([p**2 for p in proportions if p > 0])
        # end TODO
        return gini

    def _entropy(self, y):
        # TODO: the following won't work if y is not integer
        # make it work for the cases where y is a categorical variable
        # calculates average of class instances among each iteration of nodes
        proportions = np.bincount(y) / len(y)
        entropy = -np.sum([p * np.log2(p) for p in proportions if p > 0])

        # end TODO
        return entropy

    def _create_split(self, X, thresh):
        left_idx = np.argwhere(X <= thresh).flatten()
        right_idx = np.argwhere(X > thresh).flatten()
        return left_idx, right_idx

    def _information_gain(self, X, y, thresh):
        # TODO: fix the code so it can switch between the two criterion: gini and entropy
        # create conditional statements to check input of criterion
        # if criterion is entropy
        if self.criterion == "entropy":

            parent_loss = self._entropy(y)
            left_idx, right_idx = self._create_split(X, thresh)
            n, n_left, n_right = len(y), len(left_idx), len(right_idx)

            if n_left == 0 or n_right == 0:
                return 0

            child_loss = (
                n_left / n) * self._entropy(y[left_idx]) + (n_right / n) * self._entropy(y[right_idx])
            # end TODO
            return parent_loss - child_loss
        # else if the criterion is gini
        elif self.criterion == "gini":
            parent_loss = self._gini(y)
            left_idx, right_idx = self._create_split(X, thresh)
            n, n_left, n_right = len(y), len(left_idx), len(right_idx)

            if n_left == 0 or n_right == 0:
                return 0

            child_loss = (
                n_left / n) * self._gini(y[left_idx]) + (n_right / n) * self._gini(y[right_idx])
            # end TODO
            return parent_loss - child_loss

    def _best_split(self, X, y, features):
        '''TODO: add comments here
        1. while building the tree, we must compute the best split at the current stage
        To do that we must first:
        1. initialize split as a tuple that returns the best feature and
           threshold after 
        2. loop through every combination in information gain and compare the result
           to previous iterations

        '''
        split = {'score': -1, 'feat': None, 'thresh': None}

        for feat in features:
            X_feat = X[:, feat]
            thresholds = np.unique(X_feat)
            for thresh in thresholds:
                score = self._information_gain(X_feat, y, thresh)

                if score > split['score']:
                    split['score'] = score
                    split['feat'] = feat
                    split['thresh'] = thresh

        return split['feat'], split['thresh']

    def _traverse_tree(self, x, node):
        '''TODO: add some comments here
        1. recursively traverses tree and compares the node features and 
           threshold values to the current samples values
        2. returns left node if the feature is less than or equal to threshold
           otherwise returns right node
        3. base case is when we reach a leaf node, we return the valuie of the node
        '''
        if node.is_leaf():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


class RandomForestModel(object):

    def __init__(self, n_estimators):
        # TODO:
        # constructor for random forest class
        # which takes in the decision tree models
        # and number of trees needed for estimation as parameters
        self.num_trees = n_estimators
        self.models = []
        # end TODO


    def fit(self, X: pd.DataFrame, y: pd.Series):
        # TODO:
        # fit model for random forest takes in Decision tree models
        # with random subsets of features at each node
        self.models = []

        for i in range(self.num_trees):
            tree = DecisionTreeModel(max_depth=10)

            n_samples = X.shape[0]
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_sample, y_sample = X.iloc[indices], y.iloc[indices]

            tree.fit(X_sample, y_sample)
            self.models.append(tree)
        # end TODO


    def predict(self, X: pd.DataFrame):
        # TODO:
        # makes predictions for each tree by comparing the frequency
        # of labels that appear in each tree
        tree_predictions = np.array([tree.predict(X) for tree in self.models])
        tree_predictions = np.swapaxes(tree_predictions, 0, 1)
        y_pred = [most_common_Label(tree_pred)
                  for tree_pred in tree_predictions]
        return np.asarray(y_pred)
        # end TODO




def most_common_Label(y):
    # freturns the label for each tree based on the frequency
    # of that labels occurrence
    counter = Counter(y)
    most_common = counter.most_common(1)[0][0]
    return most_common



def accuracy_score(y_true, y_pred):
    # returns the performance measurement of model
    # based on the ratio of correct predictions
    y_true = catToInt(y_true)
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


def classification_report(y_test, y_pred):
    # calculate precision, recall, f1-score
    # TODO:
    matrix = confusion_matrix(y_test, y_pred)
    # ratoo of the positive predictions
    precision = matrix[0][0] / (matrix[0][0] + matrix[0][1])
    # ratio of positive predictions detected by false negatives
    recall = matrix[0][0] / (matrix[0][0] + matrix[1][1])
    # returns harmonic mean of precision and recall
    # favors calssifiers that have similar precision and recall
    f1 = 2 * (precision * recall)/(precision + recall)
    print(f"Precision {precision} \nRecall {recall}, \nF1-Score {f1}")
    result = np.array([precision, recall, f1])
    # end TODO
    return(result)


def confusion_matrix(y_test, y_pred):
 # return the 2x2 matrix
    # TODO:
    # compares actual class and predictive class
    # to determine the amount of classes that were classified
    # correctly or incorrectl
    converted = y_test.to_numpy()

    a, b = np.unique(converted, return_counts=True)
    y_copy = []
    for i in range(len(converted)):
        val = np.where(a == converted[i])
        y_copy.append(val[0][0])
    y_test = np.asarray(y_copy)

    false_positive = 0
    false_negative = 0

    true_positive = 0
    true_negative = 0

    for y_test, y_pred in zip(y_test, y_pred):

        if y_pred == y_test:
            if y_pred == 1:
                true_positive += 1
            else:
                true_negative += 1
        else:
            if y_pred == 1:
                false_positive += 1
            else:
                false_negative += 1
    confusion_matrix = [
        [true_positive, false_positive],
        [false_negative, true_negative],
    ]
    confusion_matrix = np.array(confusion_matrix)

    # end TODO
    return (confusion_matrix)


def _test():

    df = pd.read_csv('wine-tasting.csv')

    #X = df.drop(['diagnosis'], axis=1).to_numpy()
    #y = df['diagnosis'].apply(lambda x: 0 if x == 'M' else 1).to_numpy()

    X = df.drop(['diagnosis'], axis=1)
    y = df['diagnosis'].apply(lambda x: 0 if x == 'M' else 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    clf = DecisionTreeModel(max_depth=10)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("Accuracy:", acc)

def catToInt(cat_variable):

    # helper function catToInt which take a categorical variable
    #  as a paramater and returns the values converted into numeric type
    convert = cat_variable.to_numpy()
    values, counts = np.unique(convert, return_counts=True)
    y_convert = []
    for i in range(len(convert)):
        val = np.where(values == convert[i])
        y_convert.append(val[0][0])
    y_convert = np.asarray(y_convert)
    return y_convert


if __name__ == "__main__":
    _test()
