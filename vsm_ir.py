import math
from bs4 import BeautifulSoup
import nltk
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter
from collections import OrderedDict
from numpy.linalg import norm

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
dictionary = OrderedDict()
appearances = {}
df = {}
tf = {}
weights = {}
docs = []
docs_length = {}
vsm = {}


def count_words_in_record(record):
    global appearances
    ps = PorterStemmer()
    stop_words = stopwords.words('english') + ['.', "'"]
    record_num = record.findNext('RECORDNUM').text.lstrip('0').strip()
    title = record.findNext('TITLE').text
    summary = record.findNext('ABSTRACT').text if record.findNext('ABSTRACT') is not None else record. \
        findNext('EXTRACT').text
    stemmed_words = [ps.stem(w) for w in word_tokenize(title) + word_tokenize(summary)]
    word_tokens = [w for w in stemmed_words if w not in stop_words]
    docs_length[record_num] = len(word_tokens)
    for w in word_tokens:
        if w not in dictionary:
            dictionary[w] = set(record_num)
        else:
            dictionary[w].add(record_num)
    counts = Counter([w for w in word_tokens if not w.lower() in stop_words])
    for word in counts:
        appearances[(record_num, word)] = counts[word]
    return counts, record_num


def process_record(record):
    global docs
    global dictionary
    word_counts, record_num = count_words_in_record(record)
    max_occurrences = max(word_counts.values())
    docs.append(record_num)
    for word in word_counts:
        tf[(record_num, word)] = word_counts[word] / max_occurrences
        if word in df:
            df[word] += 1
        else:
            df[word] = 1


def get_tf_df(file):
    bs_data = BeautifulSoup(file.read(), "xml")
    records = bs_data.find_all('RECORD')
    for record in records:
        process_record(record)


def get_idf(num_of_docs):
    idf = {}
    for term in df:
        idf[term] = math.log2(num_of_docs / df[term])
    return idf


def get_weights(idf):
    for term in dictionary:
        for doc in docs:
            if (doc, term) in tf:
                weights[(doc, term)] = tf[(doc, term)] * idf[term]
            else:
                weights[(doc, term)] = 0
    return weights


def create_index(directory):
    global weights, docs_length, appearances, tf
    for file_name in os.listdir(directory):
        if file_name.endswith(".xml"):
            file = open(directory + '/' + file_name, 'r')
            get_tf_df(file)
            file.close()
    idf = get_idf(len(docs))
    weights = get_weights(idf)
    file = open(directory + "/vsm_inverted_index.json", "w")
    file.write("{\n\tdocs_dict: {\n")
    docs_left = len(docs)
    for doc in docs:
        file.write("\t\t" + str(doc) + ": {\n")
        file.write("\t\t\tlength: " + str(docs_length[doc]) + ",\n")
        file.write("\t\t\tnorm: " + str(norm([weights[(doc, word)] for word in dictionary])) + '\n')
        if docs_left == 1:
            file.write("\t\t}\n")
        else:
            file.write("\t\t},\n")
        docs_left -= 1
    file.write("\t},\n")
    file.write("\twords_dict: {\n")
    words_left = len(dictionary)
    for word in dictionary:
        file.write("\t\t" + word + ": {\n")
        closed = True
        for doc in docs:
            if (doc, word) in appearances:
                if not closed:
                    file.write("\t\t\t},\n")
                file.write("\t\t\t" + str(doc) + ": {\n")
                file.write("\t\t\t\tcount: " + str(appearances[(doc, word)]) + ",\n")
                file.write("\t\t\t\ttf: " + str(tf[doc, word]) + ",\n")
                file.write("\t\t\t\ttf_idf: " + str(weights[doc, word]) + "\n")
                closed = False
        file.write("\t\t\t}\n")
        if words_left == 1:
            file.write("\t\t}\n")
        else:
            file.write("\t\t},\n")
        words_left -= 1
    file.write("\t},\n")
    file.write("\taverage document length: " + str(sum(docs_length.values())/len(docs_length)) + '\n')
    file.write("\t}\n")
    file.write("}\n")


create_index("/Users/amitsharon/Documents/GitHub/Information-Retrieval")
