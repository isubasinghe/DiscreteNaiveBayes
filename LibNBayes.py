from collections import defaultdict

import pandas as pd


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
    print(df.head(4))
    return df.as_matrix(), list(classes)


def train(x, y):
    nb = NBayes()
    nb.fit(x, y)
    return nb


if __name__ == '__main__':
    data, classes = process('./2019S1-proj1-data_dos/mushroom.csv')
    up_lim = int(len(data)*0.8)

    x_train = data[:up_lim]
    y_train = classes[:up_lim]

    x_test = data[up_lim:]
    y_test = classes[up_lim:]

    model = train(x_train, y_train)

