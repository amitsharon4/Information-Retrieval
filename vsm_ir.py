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
import sys
import numpy as np
import json

#PART 1
#create inverted index

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
    stop_words = stopwords.words('english') + ['.', "'", ".", "?", "!", "[", "]", "(", ")"]
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
    json_data = {}
    docs_dict = {}
    global weights, docs_length, appearances, tf
    for file_name in os.listdir(directory):
        if file_name.endswith(".xml"):
            file = open(directory + '/' + file_name, 'r')
            get_tf_df(file)
            file.close()
    idf = get_idf(len(docs))
    weights = get_weights(idf)
    for doc in docs:
        docs_dict[doc] = {
            "length": docs_length[doc],
            "norm": norm([weights[(doc, word)] for word in dictionary])
        }
    words_dict = {word: {} for word in dictionary}
    for word in dictionary:
        for doc in docs:
            if (doc, word) in appearances:
                words_dict[word][doc] = {
                    "count": appearances[(doc, word)],
                    "tf": tf[(doc, word)],
                    "tf_idf": weights[(doc, word)]
                }
    json_data["docs_dict"] = docs_dict
    json_data["words_dict"] = words_dict
    json_data["avg_doc_length"] = sum(docs_length.values())/len(docs_length)
    with open('vsm_inverted_index.json', 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)

#create_index("C:/Users/galem/PycharmProjects/information_retrival")


#PART 2
#Query

#BM25
def bm25(fixed_query,docs_dict,words_dict,avgdl,N):
    dict_res={}
    b=0.75
    k1=1.2
    idf_query=idf_bm25_query(fixed_query, N, words_dict)
    for doc in docs_dict.keys():
        D =int(docs_dict[doc]["length"])#doc length
        tf_doc_query=tf_bm25(doc,fixed_query,words_dict)
        for_cal=(1-b+b*(D/avgdl))*k1
        k1_1=k1+1
        res=calc_sum_bm25(idf_query,tf_doc_query,for_cal,k1_1)

        if res > 0:
            dict_res[doc] = res
        sorted_dict = {k: v for k, v in sorted(dict_res.items(), key=lambda item: item[1])}

    return sorted_dict

def calc_sum_bm25(idf_query,tf_doc_query,for_cal,k1_1):
    res=0
    for i in range(len(idf_query)):
        res+=(idf_query[i]*((tf_doc_query[i]*k1_1)/(tf_doc_query[i]+for_cal)))
    return res

def tf_bm25(doc,query,words_dict):
    tf_query=[]
    for word in query:
        if words_dict[word] != None and  words_dict[word][doc] != None :
            tf = float(words_dict[word][doc]["tf"])
        else:
            tf=0
        tf_query.append(tf)
    return tf_query


def idf_bm25_query(fixed_query,N,words_dict):
    idf_query=[]
    for q_i in fixed_query:
        n_qi=len(words_dict.get(q_i))
        idf_qi=math.log(1+ (N-n_qi+0.5)/(n_qi+0.5))
        idf_query.append(idf_qi)
    return idf_query


#TF - IDF
#create vector similarity for query
def tfidf_query(query_dict,words_dict,N): #create vector for query
    #dict_res={}
    res=[] #vector query
    max_val_query=query_dict[max(query_dict, key=query_dict.get)]
    for key in query_dict.keys(): #normal tf + calc tf*idf
        val=query_dict[key]
        if words_dict.get(key) != None:
            idf=math.log2(N/len(words_dict.get(key)))
        else:
            continue #the word not in corpus so we move on
        #dict_res[key] = (val/max_val_query)*idf
        res.append((val/max_val_query)*idf)
    return res

def tfidf_doc(query_dict,words_dict,doc): #create relevant vector for doc
    vector_doc=[]
    for q in query_dict.keys():
        if q not in words_dict.keys():
            continue
        else:
            if doc not in words_dict[q]:
                continue
            else:
                vector_doc.append(float(words_dict[q][doc]["tf_idf"]))
    return vector_doc



#calcualte cosin_similarity for doc and query
def tfidf(fixed_query, docs_dict, words_dict,N):
    dict_res={}
    query_dict=dict(Counter(fixed_query)) #dict of num of appearance of word in dict
    #print(query_dict)
    vector_query=tfidf_query(query_dict,words_dict,N) #create vector similarity for query
    vecotr_query_norm=norm(vector_query)
    for doc in docs_dict.keys():
        vector_doc=tfidf_doc(query_dict,words_dict,doc)
        vector_doc_norm=float(docs_dict[doc]["norm"])
        print(vector_doc)
        print(vector_query)
        cosin_similarity=np.dot(vector_doc,vector_query)/(vector_doc_norm*vecotr_query_norm)
        if cosin_similarity > 0:
            dict_res[doc]= cosin_similarity
    sorted_dict={k: v for k, v in sorted(dict_res.items(), key=lambda item: item[1])}
    return sorted_dict

#stem query
def process_query(question):
    q=question.lower()
    ps = PorterStemmer()
    stop_words = stopwords.words('english') + ['.', "'", ".", "?", "!"]
    stemmed_words = [ps.stem(w) for w in word_tokenize(q)]
    word_tokens = [w for w in stemmed_words if w not in stop_words]
    print(word_tokens)
    return word_tokens

def query(ranking,question,index_path):
    try:
        jason_inverted = open(index_path, "r")
    except:
        print("can't open json file")
        exit()
    print("here")
    dict_inverted = json.load(jason_inverted)
    print("next")
    docs_dict=dict_inverted["docs_dict"]
    words_dict=dict_inverted["words_dict"]
    avgdl=float(dict_inverted["avg_doc_length"])

    jason_inverted.close()

    fixed_query = process_query(question) #stem query as a list
    N = len(docs_dict) #number of doc in corpus

    if ranking == "bm25":
        relevant_doc=bm25(fixed_query,docs_dict,words_dict,avgdl,N)
    elif ranking == "tfidf":
        relevant_doc=tfidf(fixed_query,docs_dict,words_dict,N)
    else:
        print("Worng argument for ranking")
        exit()

    res = open("ranked_query_docs.txt","w")
    for doc in relevant_doc.keys():
        res.write(doc+"\n")
    res.close


#Main
def main(mode, path, ranking, question):
    if mode == "create_index":
        create_index(path)
    if mode == "query":
        query(ranking,question,path)
"""
#Main
if len(sys.argv) == 1:
    print("Wrong number of arguments")
    exit()
if sys.argv[1] == "create_index":
    if len(sys.argv) != 3:
        print("Wrong number of arguments for create mode")
        exit()
    else:
        create_index(sys.argv[2])
        exit()
if sys.argv[1] == "query":
    if len(sys.argv) != 5:
        print("Wrong number of arguments for question mode")
        exit()
    else:
        ranking = sys.argv[2]
        index_path = sys.argv[3]
        question = sys.argv[4]
        query(ranking,question,index_path)
        exit()

"""