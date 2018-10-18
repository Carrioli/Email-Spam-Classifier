################################################
#
# Damiano Carrioli
# Impementation of email spam classifier based 
# on Naive Bayes assumption
#
################################################

import json
from utils import *
from naive_bayes import NaiveBayesSpamFilter


if __name__ == '__main__':

    # data locations
    root = 'data/'
    train_path_spam = root + 'spam'
    train_path_ham = root + 'easy_ham'
    test_path_spam = root + 'spam_2'
    test_path_ham = root + 'hard_ham_2'


    # paths to spam and ham emails used to train and eval
    X_train_spam = get_data_paths(train_path_spam)  
    X_train_ham = get_data_paths(train_path_ham)
    X_test_spam = get_data_paths(test_path_spam) 
    X_test_ham = get_data_paths(test_path_ham) 


    print('preprocessing')
    vocab = set()
    for path in X_train_spam + X_train_ham:
        message = open_file(path)
        words = get_words(message)
        vocab = vocab.union(set(words))
    vocab_size = len(vocab)

    
    print('training')
    # naive bayes (no hyperparameters)
    nb = NaiveBayesSpamFilter(vocab_size)
    nb.train(X_train_spam, X_train_ham)
    

    print('testing')
    # eval
    spam_acc = nb.evaluate(X_test_spam, 'spam')
    ham_acc = nb.evaluate(X_test_ham, 'ham')
        
    print('test accuracies:', 'spam emails', '{0:.3f}'.format(spam_acc) + ',', \
          'ham emails', '{0:.3f}'.format(ham_acc))

    # save
    json.dump([spam_acc, ham_acc], open('test_accuracies.json', 'w'))