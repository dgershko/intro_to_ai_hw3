from ID3 import ID3
from utils import *

"""
Make the imports of python packages needed
"""

"""
========================================================================
========================================================================
                              Experiments 
========================================================================
========================================================================
"""
target_attribute = 'diagnosis'


# ========================================================================
def basic_experiment(x_train, y_train, x_test, y_test, attribute_names, formatted_print=False):
    """
    Use ID3 model, to train on the training dataset and evaluating the accuracy in the test set.
    """
    id3 = ID3(attribute_names)
    id3.fit(x_train, y_train)
    acc = accuracy(id3.predict(x_test), y_test)

    assert acc > 0.9, 'you should get an accuracy of at least 90% for the full ID3 decision tree'
    print(f'Test Accuracy: {acc * 100:.2f}%' if formatted_print else acc)


# ========================================================================

def best_m_test(x_train, y_train, x_test, y_test, attribute_names, min_for_pruning):
    """
        Test the pruning for the best M value we have got from the cross validation experiment.
        :param: best_m: the value of M with the highest mean accuracy across folds
        :return: acc: the accuracy value of ID3 decision tree instance that using the best_m as the pruning parameter.
    """
    id3 = ID3(attribute_names, min_for_pruning=min_for_pruning)
    id3.fit(x_train, y_train)
    acc = accuracy(id3.predict(x_test), y_test)
    return acc

class fit_and_predict():
    def __init__(self, x_train, y_train, x_test, y_test, attribute_names):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.attribute_names = attribute_names

    def __call__(self, M):
        id3 = ID3(self.attribute_names, min_for_pruning=M)
        id3.fit(self.x_train, self.y_train)
        acc = accuracy(id3.predict(self.x_test), self.y_test)
        # print(f"acc: {acc}, m = {M}")
        return acc

def cross_validation_experiment(plot_graph):
    attributes_names, train_dataset, test_dataset = load_data_set('ID3')
    # M_values = list(range(1, 50, 9))
    M_values = [1, 10, 25]
    n_split = 5
    k_fold = KFold(n_splits=n_split, shuffle=True, random_state=random_gen)
    split = create_train_validation_split(train_dataset, k_fold, 0.2)
    M_accuracy = {m: 0 for m in M_values}

    # import multiprocessing
    # with multiprocessing.Pool(processes=12) as pool:
    #     m_acc_per_split = []
    #     for data_split in split:
    #         print("new split started")
    #         x_train, y_train, x_test, y_test = get_dataset_split(data_split[0], data_split[1], target_attribute)
    #         fit_and_predict_obj = fit_and_predict(x_train, y_train, x_test, y_test, attributes_names)
    #         m_acc_per_split.append(pool.map(fit_and_predict_obj, M_values))
    # m_acc_per_split = np.array(m_acc_per_split).mean(axis=0)
    # print(m_acc_per_split)
    
    for data_split in split:
        x_train, y_train, x_test, y_test = get_dataset_split(data_split[0], data_split[1], target_attribute)
        for M in M_values:
            id3 = ID3(attributes_names, min_for_pruning=M)
            id3.fit(x_train, y_train)
            M_accuracy[M] += accuracy(id3.predict(x_test), y_test) / n_split
            # print(f"M: {M}, new acc: {M_accuracy[M]}")
    util_plot_graph(M_values, M_accuracy.values(), "M value", "M accuracy")
    return max(M_accuracy, key=M_accuracy.get)
    

# ========================================================================
if __name__ == '__main__':
    attributes_names, train_dataset, test_dataset = load_data_set('ID3')
    data_split = get_dataset_split(train_dataset, test_dataset, target_attribute)
    """
    Usages helper:
    (*) To get the results in “informal” or nicely printable string representation of an object
        modify the call "utils.set_formatted_values(value=False)" from False to True and run it
    """
    formatted_print = True
    basic_experiment(*data_split, attributes_names, formatted_print)
    """
       cross validation experiment
       (*) To run the cross validation experiment over the  M pruning hyper-parameter 
           uncomment below code and run it
           modify the value from False to True to plot the experiment result
    """
    plot_graphs = True
    best_m = cross_validation_experiment(plot_graph=plot_graphs)
    print(f'best_m = {best_m}')

    """
        pruning experiment, run with the best parameter
        (*) To run the experiment uncomment below code and run it
    """
    acc = best_m_test(*data_split, attributes_names, min_for_pruning=best_m)
    print(f'Test Accuracy: {acc * 100:.2f}%' if formatted_print else acc)
    # assert acc > 0.95, 'you should get an accuracy of at least 95% for the pruned ID3 decision tree'
