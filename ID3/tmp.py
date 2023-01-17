from pprint import pprint
import numpy as np
from utils import load_data_set, get_dataset_split
from DecisonTree import class_counts

attrs, train, test = load_data_set("ID3")
a, b, c, d = get_dataset_split(train, test, attrs[0])
# pprint(class_counts(a, b))
# pprint(train["diagnosis"])
# pprint(a)
pprint(class_counts(train, train["diagnosis"]))
# print(train["diagnosis"])
# pprint(class_counts(train, train["diagnosis"]))

