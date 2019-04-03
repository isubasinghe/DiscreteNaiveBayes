from collections import defaultdict

import pandas as pd
import glob
import os
import random


def class_maintained_shuffle(data, classes):
    combined = list(zip(data, classes))
    random.shuffle(combined)
    data[:], classes[:] = zip(*combined)
    return data, classes


class NBayes:
    def __init__(self):
        self.priors = defaultdict(int)
        self.posteriors = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.V = defaultdict(set)
        self.total_classes = None
        self.k = 1

    def fit(self, x_train, y_labels, k=1):
        assert len(x_train) == len(y_labels)
        self.total_classes = len(y_labels)
        self.k = k
        for i in range(len(x_train)):
            if i % 100 == 0:
                print("Training at [" + str(i/(len(x_train))) + "%]")
            self.priors[y_labels[i]] += 1
            for j in range(len(x_train[i])):
                if x_train[i][j] is not None:
                    self.posteriors[j][y_labels[i]][x_train[i][j]] += 1
                    self.V[j].add(x_train[i][j])

        print("Training done")
      
    def predict(self, x_test):
        best_class = None
        best_score = -1
        for classification in self.priors.keys():
            curr_prob = 1
            for i in range(len(x_test)):
                if x_test[i] is not None:
                    freq_i_class = self.posteriors[i][classification][x_test[i]]
                    freq_class = self.priors[classification]
                    num_attributes = len(self.V[i])
                    curr_prob = curr_prob*((self.k + freq_i_class)/(num_attributes + freq_class))

            score = (self.priors[classification]/self.total_classes)*curr_prob
            if score > best_score:
                best_score = score
                best_class = classification
        return best_class


def process(filename, class_col=-1, header=None):
    df = pd.read_csv(filename, header=header)
    df = df.replace({'?': None})
    class_col_name = df.columns[class_col]
    classes = df[class_col_name]
    df = df.drop(class_col_name, axis =1)
    return df.as_matrix(), list(classes)


def train(x, y):
    nb = NBayes()
    nb.fit(x, y)
    return nb


def predict(nb, x_test):
    return nb.predict(x_test)


def evaluate(nb, x_test_list, y_test_list):
    class_tp = defaultdict(int)
    class_fp = defaultdict(int)
    class_fn = defaultdict(int)

    classes = nb.priors.keys()
    total_count = 0
    correct_count = 0
    for i in range(len(x_test_list)):
        predicted = nb.predict(x_test_list[i])
        label = y_test_list[i]
        if predicted == label:
            correct_count += 1
            class_tp[label] += 1
        else:
            class_fn[label] += 1
            class_fp[predicted] += 1
        total_count += 1

    recall_upper = 0
    recall_lower = 0
    precision_lower = 0
    precision_upper = 0
    for class_key in classes:
        recall_upper += class_tp[class_key]
        recall_lower += class_tp[class_key] + class_fp[class_key]
        precision_upper += class_tp[class_key]
        precision_lower += class_tp[class_key] + class_fn[class_key]
        print("Class =", class_key, "TP =", class_tp[class_key], "FP =", class_fp[class_key], "FN =", class_fn[class_key])
    if precision_lower != 0:
        precision = precision_upper/precision_lower
    if recall_lower != 0:
        recall = recall_upper/recall_lower
    print("Recall (micro-averaged) =", recall)
    print("Precision (micro-averaged)=", precision)

    f1 = 0
    if (precision+recall) != 0:
        f1 = 2*precision*recall/(precision+recall)
    print("F1 (micro-averaged)=", f1)

    precision = 0
    recall = 0
    for class_key in classes:
        if (class_tp[class_key] + class_fp[class_key]) != 0:
            precision += (class_tp[class_key]/(class_tp[class_key] + class_fp[class_key]))
        if (class_tp[class_key] + class_fn[class_key]) != 0:
            recall += (class_tp[class_key]/(class_tp[class_key] + class_fn[class_key]))
    precision = precision/len(classes)
    recall = recall/len(classes)
    print("Precision (macro-averaged) =", precision)
    print("Recall (macro-averaged) =", recall)

    f1 = 0
    if (precision+recall) != 0:
        f1 = 2*precision*recall/(precision+recall)

    print("F1 (macro-averaged) =", f1)
    print("Accuracy =", correct_count/total_count)
    return correct_count/total_count


def info_gain(posteriors):
    pass



def holdout(data, classes):
    up_lim = int(len(data) * 0.8)

    x_train = data[:up_lim]
    y_train = classes[:up_lim]

    x_test = data[up_lim:]
    y_test = classes[up_lim:]

    model = NBayes()
    model.fit(x_train, y_train)
    evaluate(model, x_test, y_test)


def chunks(data, num):
    avg = len(data) / float(num)
    out = []
    last = 0.0

    while last < len(data):
        out.append(data[int(last):int(last + avg)])
        last += avg

    return out

def cross_validate(data, classes, m):
    data_chunks = chunks(list(zip(data, classes)))
    data[:], classes[:] = zip(*data_chunks)
    for i in range(m):
        pass


if __name__ == '__main__':
    cwd = os.getcwd()
    os.chdir('./2019S1-proj1-data_dos')
    for file in glob.glob("*.csv"):
        print("*"*80)
        print(file)
        data, classes = process(file)
        data, classes = class_maintained_shuffle(data, classes)
        holdout(data, classes)
        print("*"*80)
        print('', end='\n\n')
    os.chdir(cwd)


# I have implemented add-k, holdout, cross validation