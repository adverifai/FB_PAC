import pandas as pd
import numpy as np
import gensim
from gensim import models
from utils import save_file
from utils import model_test
from utils import model_test_cross
from utils import load_file
from pre_processing import pre_process
import json
from pprint import pprint


def create_topic_models_lda(processed_docs, model_name):
    # creating a dictionary of all tokens in all documents
    dictionary = gensim.corpora.Dictionary(processed_docs)
    save_file('models/LDAdict_'+model_name+'.pickle', dictionary)
    # dictionary.filter_extremes(no_below=100, no_above=0.5, keep_n=10000) # if we want to filter the corpus
    print("Log: dictionary is created and saved.")

    # creating bag of words and tf-idf corpora
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    tf_idf = models.TfidfModel(bow_corpus)
    corpus_tf_idf = tf_idf[bow_corpus]

    # creating LDA model using bag of words
    lda_model_bow = gensim.models.LdaMulticore(bow_corpus, id2word=dictionary, num_topics=5, minimum_probability=0.0)
    save_file('models/LDAbow_'+model_name+'.pickle', lda_model_bow)
    print("Log: lda model [bog] is created and saved.")
    for idx, topic in lda_model_bow.print_topics(-1):
        print('Topic: {} | Words: {}'.format(idx, topic))

    # creating LDA model using tf-idf
    lda_model_tf_idf = gensim.models.LdaMulticore(corpus_tf_idf, id2word=dictionary, num_topics=5, minimum_probability=0.0)
    save_file('models/LDAtfidf_'+model_name+'.pickle', lda_model_tf_idf)
    print("Log: lda model [tf-idf] is created and saved.")
    for idx, topic in lda_model_tf_idf.print_topics(-1):
        print('Topic: {} | Word: {}'.format(idx, topic))

    return lda_model_bow, lda_model_tf_idf, dictionary


def print_doc_topics(doc, lda_model, dictionary):
    """ given a sample document, trained LDA model and its corresponding dictionary, this method prints the topics of the
    documents and a score associated with each topic"""
    print("\n")
    bow_vector = dictionary.doc2bow(pre_process(doc))
    for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
        print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 5)))


def docs_to_topics_vector(docs, lda_model, dictionary):
    """ given a list of documents and a trained topic mode, this method return the topic vector
        representation of all documents"""
    docs_topics_vectors = []
    for doc in docs:
        bow_vector = dictionary.doc2bow(pre_process(doc))
        docs_topics_vectors.append(lda_model[bow_vector])
    return docs_topics_vectors


def read_seeds_data():
    all_docs = []
    docs_labels = []
    with open('data/seeds.json') as f:
        data = json.load(f)
        try:
            for item in data["not_political"]:
                all_docs.append(pre_process(item))
                docs_labels.append(0)
            for item in data["political"]:
                all_docs.append(pre_process(item))
                docs_labels.append(1)
        except:
            print("Error in reading data.")
    return all_docs, docs_labels


def read_test_train():
    train_path = "fbpac-ads-en-US-train.csv"
    test_path = "fbpac-ads-en-US-test.csv"
    # data_path = "data/limited_sample.csv"
    data_train = pd.read_csv(train_path, error_bad_lines=False)
    data_test = pd.read_csv(test_path, error_bad_lines=False)

    # pre processing all the documents [title:04 + message:05]
    processed_docs = []

    for index, row in data_train.iterrows():
        try:
            processed_record = pre_process(row[4] + " " + row[5])
            processed_docs.append(processed_record)
        except:
            print("Error in pre-processing: " + str(index))
    for index, row in data_test.iterrows():
        try:
            processed_record = pre_process(row[4] + " " + row[5])
            processed_docs.append(processed_record)
        except:
            print("Error in pre-processing: " + str(index))

    print("Log: pre processing is done.")
    return processed_docs


def read_main_data():
    data_path = "data/fbpac-ads-en-US.csv"
    # data_path = "data/limited_sample.csv"
    data = pd.read_csv(data_path, error_bad_lines=False)

    # pre processing all the documents [title:04 + message:05]
    processed_docs = []

    # printing unique list of advertisers
    advertisers = data.iloc[:, 16].unique()
    np.savetxt('data/advertisers.txt', advertisers, fmt='%s')

    for index, row in data.iterrows():
        try:
            processed_record = pre_process(row[4] + " " + row[5])
            processed_docs.append(processed_record)
        except:
            print("Error in pre-processing: " + str(index))
    print("Log: pre processing is done.")
    return processed_docs

all_docs = read_test_train()
lda_model_bow, lda_model_tf_idf, dictionary = create_topic_models_lda(all_docs, "fbpac")


all_docs = read_main_data()
lda_model_bow, lda_model_tf_idf, dictionary = create_topic_models_lda(all_docs, "fbpac")

all_docs, docs_labels = read_seeds_data()
lda_model_bow, lda_model_tf_idf, dictionary = create_topic_models_lda(all_docs, "seeds")

all_docs_vectors = []
all_docs_labels = []
for i in range(len(all_docs)):
    try:
        bow = dictionary.doc2bow(all_docs[i])
        all_docs_vectors.append(lda_model_bow[bow])
        all_docs_labels.append(docs_labels[i])
    except Exception as e:
        print(e)
        print("Error in computing document's vector")
# converting the 3d array to a 2d array to be used in sklearn
n, nx, ny = np.array(all_docs_vectors).shape
d2_all_docs = np.array(all_docs_vectors).reshape((n, nx * ny))
model_test(d2_all_docs, np.array(all_docs_labels))
model_test_cross(d2_all_docs, np.array(all_docs_labels))
