from probabilities import *


class NaiveBayesSpamFilter:
    def __init__(self, vocab_size):
        '''
        probabilities are initialized and need to be calculated in the train method
        the size of the vocabulary in the training set is provided
        '''        
        self.p_word_given_spam = dict()
        self.p_word_given_ham = dict()
        self.p_spam = 0
        self.p_ham = 0
        self.vocab_size = vocab_size
        

    def train(self, X_paths_spam, X_paths_ham):
        '''
        -- Inputs
        paths to known spam and ham emails
        '''
        self.p_word_given_spam = compute_p_word_given_class(X_paths_spam, self.vocab_size)
        self.p_word_given_ham = compute_p_word_given_class(X_paths_ham, self.vocab_size)
        self.p_spam = compute_p_class(len(X_paths_spam), len(X_paths_ham))
        self.p_ham = compute_p_class(len(X_paths_ham), len(X_paths_spam))
    

    def predict(self, X_instance_path):
        '''
        -- Input
        path to input email
        -- Ouput
        classification label 
        '''
        p_spam_given_input = compute_p_class_given_input(X_instance_path, self.p_word_given_spam, self.p_spam)        
        p_ham_given_input = compute_p_class_given_input(X_instance_path, self.p_word_given_ham, self.p_ham)
        
        if p_spam_given_input > p_ham_given_input:
            return 1
        else:
            return 0
        
        
    def evaluate(self, X_paths, ground_truth_class):
        '''
        -- Inputs
        paths to emails that have the same ground truth class
        the ground truth class
        -- Ouput
        accuracy score
        '''
        gt = 1 if ground_truth_class == 'spam' else 0

        count = 0
        for path in X_paths:
            if self.predict(path) == gt:
                count += 1

        return float(count) / len(X_paths)