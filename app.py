from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from collections import defaultdict
import re
import dill as pickle

app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('base.html')


@app.route('/', methods=['POST'])
def main_script():
    query = request.form['text']
    # Read pickle file for model
    nb = pickle.load(open("nb_model.pickle", "rb"))
    labels = pickle.load(open("labels.pickle", "rb"))

    # Read ufo file using pandas dataframe
    '''ufo = pd.read_csv("Reports.csv", error_bad_lines=False, sep=',')
    print(ufo.head())
    print(ufo.info())
    print(ufo.isnull().sum())

    # Get unique labels to use for dataset
    train_labels = ufo['shape']
    print(np.unique(train_labels))
    train_data = ufo['comments']
    nb = NaiveBayes(np.unique(train_labels)) 
    print("---------------- Training In Progress --------------------")

    nb.train(train_data, train_labels)  
    print('----------------- Training Completed ---------------------')
    
    test_data = train_data
    test_labels = train_labels

    print("Number of Test Examples: ", len(test_data))  
    print("Number of Test Labels: ", len(test_labels))  
    pclasses = nb.test(test_data)  # get predictions for test set

    # check how many predcitions actually match original test labels
    test_acc = np.sum(pclasses == test_labels) / float(test_labels.shape[0])

    print("Test Set Examples: ", test_labels.shape[0])  
    print("Test Set Accuracy: ", test_acc * 100, "%")  
    pickle.dump(nb, open("nb_model.pickle", "wb"))'''

    """# Testing model code:
    ufo = pd.read_csv("Reports.csv", error_bad_lines=False, sep=',')
    test_labels = ufo['shape']
    print(np.unique(test_labels))
    test_data = ufo['comments']
    pclasses = nb.test(test_data)  # get predictions for test set
    test_acc = np.sum(pclasses == test_labels) / float(test_labels.shape[0])
    print("Test Set Examples: ", test_labels.shape[0])  # Outputs : Test Set Examples:  78392
    print("Test Set Accuracy: ", test_acc * 100, "%")  # Outputs : Test Set Accuracy:  42.56429227472191 %"""

    new_query = [query]
    p_query = nb.test(new_query)
    prob_query = nb.getExampleProb(query)
    print(labels)
    print(p_query)
    print(max(prob_query))
    print(prob_query)
    return render_template('stats.html', query=query, labels=labels, p_query=p_query, p_query_max=(max(prob_query)), len=len(labels), prob_query=prob_query)


def preprocess_string(str_arg):
    # Remove extra spaces, punctuation, and make everything lowercase
    cleaned_str = re.sub('[^a-z\s]+', ' ', str_arg, flags=re.IGNORECASE)  # removed anything thats not a-z
    cleaned_str = re.sub('(\s+)', ' ', cleaned_str)  # remove extra spaces
    cleaned_str = cleaned_str.lower()  # convert whats left to lower case
    return cleaned_str


class NaiveBayes:

    def __init__(self, unique_classes):

        self.classes = unique_classes

    def addToBow(self, example, dict_index):
        if isinstance(example, np.ndarray): example = example[0]

        for token_word in example.split():

            self.bow_dicts[dict_index][token_word] += 1

    def train(self, dataset, labels):
        self.examples = dataset
        self.labels = labels
        self.bow_dicts = np.array([defaultdict(lambda: 0) for index in range(self.classes.shape[0])])

        if not isinstance(self.examples, np.ndarray): self.examples = np.array(self.examples)
        if not isinstance(self.labels, np.ndarray): self.labels = np.array(self.labels)

        for cat_index, cat in enumerate(self.classes):
            all_cat_examples = self.examples[self.labels == cat]

            cleaned_examples = [preprocess_string(cat_example) for cat_example in all_cat_examples]

            cleaned_examples = pd.DataFrame(data=cleaned_examples)

            np.apply_along_axis(self.addToBow, 1, cleaned_examples, cat_index)

        prob_classes = np.empty(self.classes.shape[0])
        all_words = []
        cat_word_counts = np.empty(self.classes.shape[0])
        for cat_index, cat in enumerate(self.classes):
            prob_classes[cat_index] = np.sum(self.labels == cat) / float(self.labels.shape[0])

            count = list(self.bow_dicts[cat_index].values())
            cat_word_counts[cat_index] = np.sum(
                np.array(list(self.bow_dicts[cat_index].values()))) + 1

            all_words += self.bow_dicts[cat_index].keys()

        self.vocab = np.unique(np.array(all_words))
        self.vocab_length = self.vocab.shape[0]

        denoms = np.array(
            [cat_word_counts[cat_index] + self.vocab_length + 1 for cat_index, cat in enumerate(self.classes)])

        self.cats_info = [(self.bow_dicts[cat_index], prob_classes[cat_index], denoms[cat_index]) for cat_index, cat in
                          enumerate(self.classes)]
        self.cats_info = np.array(self.cats_info)

    def getExampleProb(self, test_example):

        likelihood_prob = np.zeros(self.classes.shape[0])

        for cat_index, cat in enumerate(self.classes):

            for test_token in test_example.split():
                test_token_counts = self.cats_info[cat_index][0].get(test_token, 0) + 1
                test_token_prob = test_token_counts / float(self.cats_info[cat_index][2])
                likelihood_prob[cat_index] += np.log(test_token_prob)

        post_prob = np.empty(self.classes.shape[0])
        for cat_index, cat in enumerate(self.classes):
            post_prob[cat_index] = likelihood_prob[cat_index] + np.log(self.cats_info[cat_index][1])

        return post_prob

    def test(self, test_set):
        predictions = []
        for example in test_set:
            cleaned_example = preprocess_string(example)
            post_prob = self.getExampleProb(cleaned_example)
            predictions.append(self.classes[np.argmax(post_prob)])

        return np.array(predictions)


if __name__ == '__main__':
    app.run()
