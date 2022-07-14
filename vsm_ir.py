import math

from bs4 import BeautifulSoup
import nltk
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter
from collections import OrderedDict

nltk.download('stopwords')
nltk.download('punkt')
dictionary = OrderedDict()
df = {}
tf = {}
weights = {}
docs = []
docs_length = {}
vsm = {}


def count_words_in_record(record):
    ps = PorterStemmer()
    stop_words = stopwords.words('english') + ['.', "'"]
    record_num = record.findNext('RECORDNUM').text
    title = record.findNext('TITLE').text
    summary = record.findNext('ABSTRACT').text if record.findNext('ABSTRACT') is not None else record. \
        findNext('EXTRACT').text
    stemmed_words = [ps.stem(w) for w in word_tokenize(title) + word_tokenize(summary)]
    docs_length[record_num] = len(stemmed_words)
    word_tokens = [w for w in stemmed_words if w not in stop_words]
    for w in word_tokens:
        if w not in dictionary:
            dictionary[w] = set(record_num)
        else:
            dictionary[w].add(record_num)
    counts = Counter([w for w in word_tokens if not w.lower() in stop_words])
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
    global weights
    for file_name in os.listdir(directory):
        if file_name.endswith(".xml"):
            file = open(directory + '/' + file_name, 'r')
            get_tf_df(file)
            file.close()
    idf = get_idf(len(docs))
    weights = get_weights(idf)
    for doc in docs:
        vsm[doc] = []
        for word in dictionary:
            vsm[doc].append(weights[(doc, word)])


create_index("/Users/amitsharon/Documents/GitHub/Information-Retrieval")
print("breakpoint")
