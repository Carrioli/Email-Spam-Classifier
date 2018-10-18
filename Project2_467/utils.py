###########################################################
# DO NOT EDIT ANYTHING
###########################################################

import nltk
from nltk.tokenize import wordpunct_tokenize
import os

f = open('data/stopwords.txt')
stopwords = [w.strip() for w in f.readlines() if w != '\n']
f.close()


def get_words(message):
    """
    Extracts words from an message.
    """
    all_words = wordpunct_tokenize(message.replace('=\\n', '').lower())
    
    # remove the stopwords
    msg_words = [word for word in all_words if word not in stopwords and len(word) > 2]
    
    return msg_words


def open_file(file_name):
    """
    Opens file, returns it as a single string.
    """
    content = ''
    found_start = False
    with open(file_name, 'r', encoding='Latin-1') as handle:
        for line in handle:
            if line == '\n':
                found_start = True
            if found_start:
                content += line
    return content


def get_data_paths(path):
    """
    Gets data paths inside the given folder. You dont need to use this method.
    """   
    file_list = os.listdir(path)
    
    filtered_list = []
    for filename in file_list:
        f_split = filename.split('.')
        if len(f_split) == 2:
            if len(f_split[1]) == 32:
                filtered_list.append(path + '/' + filename)
    return filtered_list