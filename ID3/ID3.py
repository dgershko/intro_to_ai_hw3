import math
import numpy as np
from DecisonTree import Leaf, Question, DecisionNode, class_counts
from utils import *

"""
Make the imports of python packages needed
"""


class ID3:
    def __init__(self, label_names: list, min_for_pruning=0, target_attribute='diagnosis'):
        self.label_names = label_names
        self.target_attribute = target_attribute
        self.tree_root = None
        self.used_features = set()
        self.min_for_pruning = min_for_pruning

    @staticmethod
    def entropy(rows: np.ndarray, labels: np.ndarray):
        """
        Calculate the entropy of a distribution for the classes probability values.
        :param rows: array of samples
        :param labels: rows data labels.
        :return: entropy value.
        """
        # 3% faster lmao
        n_labels = len(labels)
        if n_labels <= 1:
            return 0
        _, counts = np.unique(labels, return_counts=True)
        probs = counts[np.nonzero(counts)] / n_labels
        tmp_entropy = - np.sum(probs * np.log2(probs))
        return tmp_entropy

        counts = class_counts(rows, labels)
        num_of_objects = np.shape(rows)[0]
        # p(label)  = counts[label] / num_of_objects
        p_label = list(map(lambda x: counts[x] / num_of_objects, set(labels)))
        entropy = -np.sum([label_prob * math.log2(label_prob) if label_prob else 0 for label_prob in p_label])
        return entropy

    def info_gain(self, left, left_labels, right, right_labels, current_uncertainty):
        """
        Calculate the information gain, as the uncertainty of the starting node, minus the weighted impurity of
        two child nodes.
        :param left: the left child rows.
        :param left_labels: the left child labels.
        :param right: the right child rows.
        :param right_labels: the right child labels.
        :param current_uncertainty: the current uncertainty of the current node
        :return: the info gain for splitting the current node into the two children left and right.
        """
        assert (len(left) == len(left_labels)) and (len(right) == len(right_labels)), \
            'The split of current node is not right, rows size should be equal to labels size.'

        info_gain_value = 0.0
        total_size = len(left) + len(right)
        l_entropy, r_entropy = self.entropy(left, left_labels), self.entropy(right, right_labels)
        info_gain_value = current_uncertainty - (len(left) * l_entropy + len(right) * r_entropy) / total_size

        return info_gain_value

    def partition(self, rows, labels, question: Question, current_uncertainty):
        """
        Partitions the rows by the question.
        :param rows: array of samples
        :param labels: rows data labels.
        :param question: an instance of the Question which we will use to partition the data.
        :param current_uncertainty: the current uncertainty of the current node
        :return: Tuple of (gain, true_rows, true_labels, false_rows, false_labels)
        """
        gain, true_rows, true_labels, false_rows, false_labels = 0, [], [], [], []
        assert len(rows) == len(labels), 'Rows size should be equal to labels size.'
        
        for row, label in zip(rows, labels):
            if question.match(row):
                true_rows.append(row)
                true_labels.append(label)
            else:
                false_rows.append(row)
                false_labels.append(label)
        
        gain = self.info_gain(true_rows, true_labels, false_rows, false_labels, current_uncertainty)
        
        return gain, true_rows, true_labels, false_rows, false_labels

    def find_best_split(self, rows, labels):
        """
        Find the best question to ask by iterating over every feature / value and calculating the information gain.
        :param rows: array of samples
        :param labels: rows data labels.
        :return: Tuple of (best_gain, best_question, best_true_rows, best_true_labels, best_false_rows, best_false_labels)
        """
        best_gain = - math.inf  # keep track of the best information gain
        best_question = None  # keep train of the feature / value that produced it
        best_false_rows, best_false_labels = None, None
        best_true_rows, best_true_labels = None, None
        current_uncertainty = self.entropy(rows, labels)
        for feature_index in range(np.shape(rows)[1]):
            feature_values = np.sort([example[feature_index] for example in rows])
            thresholds = []
            for example_pair in range(len(feature_values) - 1):
                thresholds.append(0.5 * (feature_values[example_pair] + feature_values[example_pair + 1]))
            for threshold in thresholds:
                current_question = Question(self.label_names[feature_index], feature_index, threshold)
                gain, true_rows, true_labels, false_rows, false_labels = self.partition(rows, labels, current_question, current_uncertainty)
                if gain >= best_gain:
                    best_gain = gain
                    best_question = current_question
                    best_true_rows, best_true_labels, best_false_rows, best_false_labels = true_rows, true_labels, false_rows, false_labels

        return best_gain, best_question, best_true_rows, best_true_labels, best_false_rows, best_false_labels

    def build_tree(self, rows, labels):
        """
        Build the decision Tree in recursion.
        :param rows: array of samples
        :param labels: rows data labels.
        :return: a Question node, This records the best feature / value to ask at this point, depending on the answer.
                or leaf if we have to prune this branch (in which cases ?)

        """
        if len(labels) == 0:
            print("sanity check")
            return None

        if len(rows) <= self.min_for_pruning or len(set(labels)) == 1:
            # print(f"Leaf > {len(labels)} labels with value {labels[0]}")
            return Leaf(rows, labels)
        

        best_question = None
        true_branch, false_branch = None, None
        
        best_gain, best_question, best_true_rows, best_true_labels, best_false_rows, best_false_labels = self.find_best_split(rows, labels)
        # if all labels are the same - LEAF
        
        true_branch = self.build_tree(best_true_rows, best_true_labels)
        false_branch = self.build_tree(best_false_rows, best_false_labels)
        # print(f"question chosen: {best_question}")

        return DecisionNode(best_question, true_branch, false_branch)

    def fit(self, x_train, y_train):
        """
        Trains the ID3 model. By building the tree.
        :param x_train: A labeled training data.
        :param y_train: training data labels.
        """
        # Hello Python SKLearn
        # I've come to talk with you again
        self.tree_root = self.build_tree(x_train, y_train)

    def predict_sample(self, row, node: DecisionNode or Leaf = None):
        """
        Predict the most likely class for single sample in subtree of the given node.
        :param row: vector of shape (1,D).
        :return: The row prediction.
        """
        if node is None:
            node = self.tree_root
            
        prediction = None

        if isinstance(node, Leaf):
            # TODO: I hate this
            if len(set(node.predictions.values())) == len(node.predictions.values()):
                return 'M'
            return max(node.predictions, key=node.predictions.get)

        if node.question.match(row):
            prediction = self.predict_sample(row, node.true_branch)
        else:
            prediction = self.predict_sample(row, node.false_branch)

        return prediction

    def predict(self, rows):
        """
        Predict the most likely class for each sample in a given vector.
        :param rows: vector of shape (N,D) where N is the number of samples.
        :return: A vector of shape (N,) containing the predicted classes.
        """
        y_pred = [self.predict_sample(row) for row in rows]
        return np.array(y_pred)
