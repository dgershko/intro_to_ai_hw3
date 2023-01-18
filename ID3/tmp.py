from pprint import pprint
import numpy as np
from utils import load_data_set, get_dataset_split
from DecisonTree import Leaf, Question, DecisionNode, class_counts
import math
from utils import *

attrs, train, test = load_data_set("ID3")
a, b, c, d = get_dataset_split(train, test, attrs[0])
# pprint(class_counts(a, b))
# pprint(train["diagnosis"])
# pprint(a)
# pprint(class_counts(train, train["diagnosis"]))
# print(train["diagnosis"])
# pprint(class_counts(train, train["diagnosis"]))


def entropy(rows: np.ndarray, labels: np.ndarray):
    """
    Calculate the entropy of a distribution for the classes probability values.
    :param rows: array of samples
    :param labels: rows data labels.
    :return: entropy value.
    """
    counts = class_counts(rows, labels)
    # print(counts)
    num_of_objects = np.shape(rows)[0]
    # p(label)  = counts[label] / num_of_objects
    p_label = list(map(lambda x: counts[x] / num_of_objects, set(labels)))
    entropy = -np.sum([label_prob * math.log2(label_prob) if label_prob else 0 for label_prob in p_label])
    
    return entropy

def info_gain(left, left_labels, right, right_labels, current_uncertainty):
    assert (len(left) == len(left_labels)) and (len(right) == len(right_labels)), \
        'The split of current node is not right, rows size should be equal to labels size.'

    info_gain_value = 0.0
    total_size = len(left) + len(right)
    l_entropy, r_entropy = entropy(left, left_labels), entropy(right, right_labels)
    info_gain_value = current_uncertainty - (len(left) * l_entropy + len(right) * r_entropy) / total_size

    return info_gain_value

def partition(rows, labels, question: Question, current_uncertainty):
    gain, true_rows, true_labels, false_rows, false_labels = 0, [], [], [], []
    assert len(rows) == len(labels), 'Rows size should be equal to labels size.'
    
    for row, label in zip(rows, labels):
        if question.match(row):
            true_rows.append(row)
            true_labels.append(label)
        else:
            false_rows.append(row)
            false_labels.append(label)
    
    gain = info_gain(true_rows, true_labels, false_rows, false_labels, current_uncertainty)
    
    return gain, true_rows, true_labels, false_rows, false_labels

def find_best_split(rows, labels):
    """
    Find the best question to ask by iterating over every feature / value and calculating the information gain.
    :param rows: array of samples
    :param labels: rows data labels.
    :return: Tuple of (best_gain, best_question, best_true_rows, best_true_labels, best_false_rows, best_false_labels)
    """
    # TODO:
    #   - For each feature of the dataset, build a proper question to partition the dataset using this feature.
    #   - find the best feature to split the data. (using the `partition` method)
    best_gain = - math.inf  # keep track of the best information gain
    best_question = None  # keep train of the feature / value that produced it
    best_false_rows, best_false_labels = None, None
    best_true_rows, best_true_labels = None, None
    current_uncertainty = entropy(rows, labels)
    for feature_index in range(np.shape(rows)[1]):
        feature_values = np.sort([example[feature_index] for example in rows])
        thresholds = []
        for example_pair in range(len(feature_values) - 1):
            thresholds.append(0.5 * (feature_values[example_pair] + feature_values[example_pair + 1]))
        for threshold in thresholds:
            current_question = Question("IDK", feature_index, threshold)
            gain, true_rows, true_labels, false_rows, false_labels = partition(rows, labels, current_question, current_uncertainty)
            if gain > best_gain:
                best_gain = gain
                best_question = current_question
                best_true_rows, best_true_labels, best_false_rows, best_false_labels = true_rows, true_labels, false_rows, false_labels

    return best_gain, best_question, best_true_rows, best_true_labels, best_false_rows, best_false_labels

best_gain, best_question, best_true_rows, best_true_labels, best_false_rows, best_false_labels = find_best_split(a, b)
print(best_gain)
print(best_question)
# print(entropy(train[1:], train[1]))
# print(entropy(a, b))
# max_gain = {}
# for i in range(10, 70, 1):
#     question = Question("radius_mean", 0, i / 2)
#     gain, true_rows, true_labels, false_rows, false_labels = partition(a, b, question, entropy(a, b))
#     # print(partition(a, b, question, entropy(a, b)))
#     print(gain)
#     max_gain[i] = gain
# print(f"max gain: {max_gain}")