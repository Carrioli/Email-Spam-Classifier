from utils import get_words, open_file

def compute_p_word_given_class(data_paths, vocab_size):
    """
    Return a dictionary of word probabilities, P(word | class). All datapaths belong to the same class.
    Incorporate Laplacian Smoothing with k=1 here. p_word_given_class should include the probability of UNKNOWN_WORD, 
        any word that doesn't appear in the training set
    """
    p_word_given_class = dict()
    # compute number of words in the given class
    class_size = 0
    for path in data_paths:
        message = open_file(path)
        words = get_words(message)
        class_size += len(words)

    # add elements to dictionary
    for path in data_paths:
        message = open_file(path)
        words = get_words(message)
        for word in words:
            if word in p_word_given_class:
                p_word_given_class[word] += 1 / (class_size + vocab_size + 1)
            else:
                p_word_given_class[word] = 2 / (class_size + vocab_size + 1)
    p_word_given_class['UNKNOWN_WORD'] = 1 / (class_size + vocab_size + 1)
    return p_word_given_class


def compute_p_class(n_samples_this_class, n_samples_other_class):
    """
    Return P(class)
    Incorporate Laplacian Smoothing with k=1 here.
    """
    p_class = (n_samples_this_class + 1) / (n_samples_this_class + n_samples_other_class + 2)
    return p_class


def compute_p_class_given_input(input_path, p_word_given_class, p_class):
    """
    Return P(class | input).
    """
    message = open_file(input_path)
    words = get_words(message)

    p_class_given_input = 0.0
    for word in words:
        if(word in p_word_given_class):
            p_class_given_input += ln(p_word_given_class[word] * p_class)
        else:
            p_class_given_input += ln(p_word_given_class['UNKNOWN_WORD'] * p_class)

    return p_class_given_input

def ln(x):
    n = 10000000.0
    return n * ((x ** (1 / n)) - 1)