from __future__ import division

import math
import os
import numpy as np

from collections import defaultdict
import string

import tkinter
import tkinter.filedialog as fd
import tokenizer
import matplotlib.pyplot as plt

# Global class labels.
POS_LABEL = 'pos'
NEG_LABEL = 'neg'

root = tkinter.Tk()
root.withdraw()
file = fd.askdirectory(parent=root,title='Choose a directory')
# Path to dataset
PATH_TO_DATA = file
# e.g. r"c:\path\to\large_movie_review_dataset", etc.
TRAIN_DIR = PATH_TO_DATA + '/train'
TEST_DIR = PATH_TO_DATA + '/test'



def tokenize_doc(doc):
    """
    Tokenize a document and return its bag-of-words representation.
    doc - a string representing a document.
    returns a dictionary mapping each word to the number of times it appears in doc.
    """
    doc = doc.translate(str.maketrans('', '', string.punctuation))
    token_list = list(tokenizer.tokenize (doc))
    tokens = []
    for token in token_list:
        tokens.append(token.txt)

    del tokens[0], tokens[-1]
    seen_tokens = []
    final_dict = []
    numbers = []
    for i in tokens:
        if i not in seen_tokens:
            seen_tokens.append(i)
            numbers.append(1)
        else:
            numbers[seen_tokens.index(i)] = numbers[seen_tokens.index(i)] + 1
    for s in range(len(seen_tokens)):
        final_dict.append({seen_tokens[s]: numbers[s]})

    return final_dict

class NaiveBayes:
    """A Naive Bayes model for text classification."""

    def __init__(self):
        # Vocabulary is a set that stores every word seen in the training data
        self.vocab = set()

        # class_total_doc_counts is a dictionary that maps a class (i.e., pos/neg) to
        # the number of documents in the trainning set of that class
        self.class_total_doc_counts = { POS_LABEL: 0.0,
                                        NEG_LABEL: 0.0 }

        # class_total_word_counts is a dictionary that maps a class (i.e., pos/neg) to
        # the number of words in the training set in documents of that class
        self.class_total_word_counts = { POS_LABEL: 0.0,
                                         NEG_LABEL: 0.0 }

        # class_word_counts is a dictionary of dictionaries. It maps a class (i.e.,
        # pos/neg) to a dictionary of word counts. For example:
        #    self.class_word_counts[POS_LABEL]['awesome']
        # stores the number of times the word 'awesome' appears in documents
        # of the positive class in the training documents.
        self.class_word_counts = { POS_LABEL: defaultdict(float),
                                   NEG_LABEL: defaultdict(float) }


    def train_model(self, num_docs=None):
        """
        This function processes the entire training set using the global PATH
        variable above.  It makes use of the tokenize_doc and update_model
        functions you will implement.

        num_docs: set this to e.g. 10 to train on only 10 docs from each category.
        """

        if num_docs is not None:
            print("Limiting to only %s docs per clas" % num_docs)

        #pos_path = os.path.join(TRAIN_DIR, POS_LABEL)
        #neg_path = os.path.join(TRAIN_DIR, NEG_LABEL)
        pos_path = TRAIN_DIR + '/' + POS_LABEL
        neg_path = TRAIN_DIR + '/' + NEG_LABEL
        print("Starting training with paths %s and %s" % (pos_path, neg_path))
        for (p, label) in [ (pos_path, POS_LABEL), (neg_path, NEG_LABEL) ]:
            filenames = os.listdir(p)
            if num_docs is not None: filenames = filenames[:num_docs]
            for f in filenames:
                with open(os.path.join(p,f),'r', encoding="utf8") as doc:
                    content = doc.read()
                    self.tokenize_and_update_model(content, label)
        self.report_statistics_after_training()

    def report_statistics_after_training(self):
        """
        Report a number of statistics after training.
        """

        print("REPORTING CORPUS STATISTICS")
        print("NUMBER OF DOCUMENTS IN POSITIVE CLASS:", self.class_total_doc_counts[POS_LABEL])
        print("NUMBER OF DOCUMENTS IN NEGATIVE CLASS:", self.class_total_doc_counts[NEG_LABEL])
        print("NUMBER OF TOKENS IN POSITIVE CLASS:", self.class_total_word_counts[POS_LABEL])
        print("NUMBER OF TOKENS IN NEGATIVE CLASS:", self.class_total_word_counts[NEG_LABEL])
        print("VOCABULARY SIZE: NUMBER OF UNIQUE WORDTYPES IN TRAINING CORPUS:", len(self.vocab))

    def update_model(self, bow, label):
        """
        Update internal statistics given a document represented as a bag-of-words
        bow - a map from words to their counts
        label - the class of the document whose bag-of-words representation was input
        This function doesn't return anything but should update a number of internal
        statistics. Specifically, it updates:
          - the internal map the counts, per class, how many times each word was
            seen (self.class_word_counts)
          - the number of words seen for each class (self.class_total_word_counts)
          - the vocabulary seen so far (self.vocab)
          - the number of documents seen of each class (self.class_total_doc_counts)
        """

        self.class_total_doc_counts
        self.class_total_word_counts[label] += 1.0
        words = []
        for w in bow:
            i = list(w.keys())[0]
            words.append(i)
            if i not in self.class_word_counts[label].keys():
                self.class_word_counts[label][i] = list(w.values())[0]
            elif i in self.class_word_counts[label].keys():
                t = self.class_word_counts[label][i]
                self.class_word_counts[label][i] = t+1.0
            self.class_total_word_counts[label] += list(w.values())[0]
        self.vocab = self.vocab.union(set(words))
        self.class_total_doc_counts[label] += 1.0



    def tokenize_and_update_model(self, doc, label):
        """
        Tokenizes a document doc and updates internal count statistics.
        doc - a string representing a document.
        label - the sentiment of the document (either postive or negative)
        stop_word - a boolean flag indicating whether to stop word or not
        """
        bow = tokenize_doc(doc)
        self.update_model(bow, label)

    def top_n(self, label, n):
        """
        Returns the most frequent n tokens for documents with class 'label'.
        """
        vs = []
        ks = []
        for k, v in list(self.class_word_counts[label].items()):
            ks.append(k)
            vs.append(v)
        
        n_most = []
        while n != 0:
            max = 0
            max_idx = 0
            for i in range(len(vs)):
                if vs[i] > max:
                    max = vs[i]
                    max_idx = i

            n_most.append((ks[max_idx], vs[max_idx]))
            vs.remove(vs[max_idx])
            ks.remove(ks[max_idx])
            n -= 1

        return n_most


    def p_word_given_label(self, word, label):
        """
        Returns the probability of word given label (i.e., P(word|label))
        according to this NB model.
        """
        p = self.class_word_counts[label][word] / self.class_total_word_counts[label]
        return p

    def p_word_given_label_and_psuedocount(self, word, label, alpha):

        """
        Returns the probability of word given label wrt psuedo counts.
        alpha - psuedocount parameter
        """
        #p = (self.class_word_counts[label][word] + alpha) / (self.class_total_word_counts[label] * (1 + alpha))
        p = (self.class_word_counts[label][word] + alpha) / (self.class_total_word_counts[label] + (len(self.vocab) * alpha))
        return p

    def log_likelihood(self, bow, label, alpha):
        """
        Computes the log likelihood of a set of words give a label and psuedocount.
        bow - a bag of words (i.e., a tokenized document)
        label - either the positive or negative label
        alpha - float; psuedocount parameter
        """
        prob = 0
        for w in bow:
            word = list(w.keys())[0]
            val = list(w.values())[0]
            prob += math.log(self.p_word_given_label_and_psuedocount(word, label, alpha)) * val
        return prob

    def log_prior(self, label):
        """
        Returns a float representing the fraction of training documents
        that are of class 'label'.
        """
        res = math.log(self.class_total_doc_counts[label] / sum(self.class_total_doc_counts.values()))
        return res

    def unnormalized_log_posterior(self, bow, label, alpha):
        """
        alpha - psuedocount parameter
        bow - a bag of words (i.e., a tokenized document)
        Computes the unnormalized log posterior (of doc being of class 'label').
        """
        unnorm_log = self.log_likelihood(bow, label, alpha) + self.log_prior(label)
        return unnorm_log

    def classify(self, bow, alpha):
        """
        alpha - psuedocount parameter.
        bow - a bag of words (i.e., a tokenized document)

        Compares the unnormalized log posterior for doc for both the positive
        and negative classes and returns the either POS_LABEL or NEG_LABEL
        (depending on which resulted in the higher unnormalized log posterior).
        """
        pos_val = self.unnormalized_log_posterior(bow, POS_LABEL, alpha)
        neg_val = self.unnormalized_log_posterior(bow, NEG_LABEL, alpha)
        if pos_val > neg_val:
            return POS_LABEL
        else:
            return NEG_LABEL

    def likelihood_ratio(self, word, alpha):
        """
        alpha - psuedocount parameter.
        Returns the ratio of P(word|pos) to P(word|neg).
        """
        p_pos = self.p_word_given_label_and_psuedocount(word, POS_LABEL, alpha)
        p_neg = self.p_word_given_label_and_psuedocount(word, NEG_LABEL, alpha)
        lr = p_pos/p_neg
        return lr

    def evaluate_classifier_accuracy(self, alpha, num_docs):
        """
        alpha - psuedocount parameter.
        This function should go through the test data, classify each instance and
        compute the accuracy of the classifier (the fraction of classifications
        the classifier gets right.
        """
        correct = 0
        total = 0
        pos_path = os.path.join(TEST_DIR, POS_LABEL)
        neg_path = os.path.join(TEST_DIR, NEG_LABEL)
        for (p, label) in [ (pos_path, POS_LABEL), (neg_path, NEG_LABEL) ]:
            filenames = os.listdir(p)
            if num_docs is not None: filenames = filenames[num_docs:2*num_docs]
            for f in filenames:
                with open(os.path.join(p,f),'r') as doc:
                    try:
                        content = doc.read()
                        bow = tokenize_doc(content)
                        if self.classify(bow, alpha) == label:
                            correct += 1
                        total +=1
                    except:
                        pass

        return correct / total

def produce_results():
    # PRELIMINARIES
     d1 = "this sample doc has   words that  repeat repeat"
     bow = tokenize_doc(d1)
     assert bow[0]['this'] == 1
     assert bow[1]['sample'] == 1
     assert bow[2]['doc'] == 1
     assert bow[3]['has'] == 1
     assert bow[4]['words'] == 1
     assert bow[5]['that'] == 1
     assert bow[6]['repeat'] == 2
     print('')
     print('[done.]')

     print("vocabulary size: " + str(len(nb.vocab)))
     print( '')

    # print "TOP 10 WORDS FOR CLASS " + POS_LABEL + " :"
     for tok, count in nb.top_n(POS_LABEL, 10):
         print('', tok, count)
     print('')

    # print "TOP 10 WORDS FOR CLASS " + NEG_LABEL + " :"
     for tok, count in nb.top_n(NEG_LABEL, 10):
         print('', tok, count)
     print('')

def plot_psuedocount_vs_accuracy(psuedocounts, accuracies):
    """
    A function to plot psuedocounts vs. accuries. You may want to modify this function
    to enhance your plot.
    """
    plt.plot(psuedocounts, accuracies)
    plt.xlabel('Psuedocount Parameter')
    plt.ylabel('Accuracy (%)')
    plt.title('Psuedocount Parameter vs. Accuracy Experiment')
    plt.show()

if __name__ == '__main__':
    nb = NaiveBayes()
    nb.train_model()
    accs = []
    accuracies = [nb.evaluate_classifier_accuracy(i, 20) for i in range(1, 10)]
    #print("Best pseudocount: " + str(np.argmax(accuracies) + 1))
    #print("Best accuracy: " + str(max(accuracies)))
    plot_psuedocount_vs_accuracy(range(1, 10), accuracies)
    produce_results()
