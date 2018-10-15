import pandas as pd
import gensim
from gensim import models
from utils import save_file
from pre_processing import pre_process


def create_topic_models():
    data_path = "data/fbpac-ads-en-US.csv"
    # data_path = "data/limited_sample.csv"
    data = pd.read_csv(data_path, error_bad_lines=False)

    # pre processing all the documents [title:04 + message:05]
    processed_docs = []
    for index, row in data.iterrows():
        try:
            processed_record = pre_process(row[4] + " " + row[5])
            processed_docs.append(processed_record)
        except:
            print("Error in pre-processing: " + str(index))
    print("Log: pre processing is done.")

    # creating a dictionary of all tokens in all documents
    dictionary = gensim.corpora.Dictionary(processed_docs)
    save_file('models/LDAdict.pickle', dictionary)
    dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=100000)
    print("Log: dictionary is created and saved.")

    # creating bag of words and tf-idf corpora
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    tf_idf = models.TfidfModel(bow_corpus)
    corpus_tf_idf = tf_idf[bow_corpus]

    # creating LDA model using bag of words
    lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=2, id2word=dictionary, passes=2, workers=4)
    save_file('models/LDAbow.pickle', lda_model)
    print("Log: lda model [bog] is created and saved.")
    for idx, topic in lda_model.print_topics(-1):
        print('Topic: {} | Words: {}'.format(idx, topic))

    # creating LDA model using tf-idf
    lda_model_tf_idf = gensim.models.LdaMulticore(corpus_tf_idf, num_topics=2, id2word=dictionary, passes=2, workers=4)
    save_file('models/LDAtfidf.pickle', lda_model)
    print("Log: lda model [tf-idf] is created and saved.")
    for idx, topic in lda_model_tf_idf.print_topics(-1):
        print('Topic: {} | Word: {}'.format(idx, topic))

    # test_topic_model('How a Pentagon deal became an identity crisis for Google', lda_model, dictionary)


def test_topic_model(doc, lda_model, dictionary):
    print("\n")
    bow_vector = dictionary.doc2bow(pre_process(doc))
    for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
        print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 5)))


create_topic_models()
